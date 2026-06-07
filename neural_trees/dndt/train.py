"""Train & evaluate DNDT, reproducing the paper's tabular setup.

The DNDT paper (Yang et al., 2018) reports accuracy on small UCI datasets and
compares against a sklearn decision tree and a neural network. Here we
reproduce that comparison on Iris (the canonical 4-feature / 3-class set) and
allow any sklearn-style dataset via ``--dataset``.

Usage
-----
    python train.py --dataset iris --n_cut 1 --epochs 1000
    python train.py --dataset wine --n_cut 1 --epochs 2000
    python train.py --dataset breast_cancer --n_cut 1 --feature_topk 6
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from dndt import DNDT

DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choices: {list(DATASETS)}")
    data = DATASETS[name]()
    return data.data, data.target


def train_dndt(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    n_cut: int,
    temperature: float,
    epochs: int,
    lr: float,
    feature_subset: list[int] | None,
    seed: int,
) -> dict[str, float]:
    torch.manual_seed(seed)
    n_features = X_tr.shape[1]
    n_classes = int(np.max(np.concatenate([y_tr, y_te])) + 1)

    model = DNDT(
        n_features=n_features,
        n_classes=n_classes,
        n_cut=n_cut,
        temperature=temperature,
        feature_subset=feature_subset,
        seed=seed,
    )

    Xtr_t = torch.tensor(X_tr, dtype=torch.float32)
    ytr_t = torch.tensor(y_tr, dtype=torch.long)
    Xte_t = torch.tensor(X_te, dtype=torch.float32)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(Xtr_t)
        loss = loss_fn(logits, ytr_t)
        loss.backward()
        opt.step()
    wall = time.time() - t0

    model.eval()
    y_pred = model.predict(Xte_t).numpy()
    return {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "f1_macro": float(f1_score(y_te, y_pred, average="macro")),
        "n_leaves": int((n_cut + 1) ** model.n_features),
        "wall_s": round(wall, 2),
        "final_loss": float(loss.item()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Reproduce DNDT results.")
    p.add_argument("--dataset", default="iris", choices=list(DATASETS))
    p.add_argument("--n_cut", type=int, default=1, help="cut points per feature")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--test_fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--feature_topk",
        type=int,
        default=0,
        help="If >0, keep only the top-k features by mutual information "
        "(keeps leaf count tractable on high-dim data).",
    )
    p.add_argument("--out", default="../results")
    args = p.parse_args()

    X, y = load_dataset(args.dataset)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_fraction, random_state=args.seed, stratify=y
    )

    # Standardise — soft binning is scale-sensitive.
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

    # Optional feature subset for high-dimensional data.
    feature_subset: list[int] | None = None
    if args.feature_topk and args.feature_topk < X.shape[1]:
        mi = mutual_info_classif(X_tr, y_tr, random_state=args.seed)
        feature_subset = sorted(np.argsort(mi)[::-1][: args.feature_topk].tolist())

    print(f"\n=== DNDT on {args.dataset} ===")
    print(f"train={X_tr.shape[0]} test={X_te.shape[0]} features={X.shape[1]} "
          f"classes={int(np.max(y) + 1)}")
    if feature_subset is not None:
        print(f"feature subset (top-{args.feature_topk} by MI): {feature_subset}")

    dndt_res = train_dndt(
        X_tr, y_tr, X_te, y_te,
        n_cut=args.n_cut,
        temperature=args.temperature,
        epochs=args.epochs,
        lr=args.lr,
        feature_subset=feature_subset,
        seed=args.seed,
    )

    # Baselines from the paper's comparison: a CART decision tree and (proxy) RF.
    dt = DecisionTreeClassifier(random_state=args.seed).fit(X_tr, y_tr)
    rf = RandomForestClassifier(n_estimators=100, random_state=args.seed).fit(X_tr, y_tr)
    baselines = {
        "sklearn_decision_tree": float(accuracy_score(y_te, dt.predict(X_te))),
        "sklearn_random_forest": float(accuracy_score(y_te, rf.predict(X_te))),
    }

    print("\n--- Results ---")
    print(f"DNDT     accuracy: {dndt_res['accuracy']:.4f}  "
          f"f1_macro: {dndt_res['f1_macro']:.4f}  "
          f"leaves: {dndt_res['n_leaves']}  ({dndt_res['wall_s']}s)")
    print(f"CART DT  accuracy: {baselines['sklearn_decision_tree']:.4f}")
    print(f"RF       accuracy: {baselines['sklearn_random_forest']:.4f}")

    out_dir = Path(__file__).resolve().parent / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "dataset": args.dataset,
        "config": vars(args),
        "feature_subset": feature_subset,
        "dndt": dndt_res,
        "baselines": baselines,
    }
    out_path = out_dir / f"dndt_{args.dataset}.json"
    out_path.write_text(json.dumps(record, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
