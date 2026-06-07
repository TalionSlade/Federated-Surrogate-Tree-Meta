"""Train & evaluate NODE, mirroring the DNDT harness.

NODE (Popov et al., ICLR 2020) is designed as a strong deep baseline for
tabular data, competitive with GBDTs. We evaluate on the same small UCI sets
used for the DNDT reproduction plus sklearn DT / RF baselines, with mini-batch
Adam training and early-ish stopping by epochs.

Usage
-----
    python train.py --dataset iris
    python train.py --dataset wine --num_trees 128 --depth 4
    python train.py --dataset breast_cancer --num_layers 2 --epochs 150
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.tree import DecisionTreeClassifier

from node import NODEClassifier

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


def train_node(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    num_layers: int,
    num_trees: int,
    depth: int,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> dict[str, float]:
    torch.manual_seed(seed)
    n_features = X_tr.shape[1]
    n_classes = int(np.max(np.concatenate([y_tr, y_te])) + 1)

    model = NODEClassifier(
        in_features=n_features,
        n_classes=n_classes,
        num_layers=num_layers,
        num_trees=num_trees,
        depth=depth,
    )

    Xtr_t = torch.tensor(X_tr, dtype=torch.float32)
    ytr_t = torch.tensor(y_tr, dtype=torch.long)
    Xte_t = torch.tensor(X_te, dtype=torch.float32)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    n = Xtr_t.shape[0]
    rng = np.random.default_rng(seed)

    # Trigger lazy data-aware init with a full-batch forward in train mode.
    model.train()
    with torch.no_grad():
        _ = model(Xtr_t)

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            logits = model(Xtr_t[idx])
            loss = loss_fn(logits, ytr_t[idx])
            loss.backward()
            opt.step()
    wall = time.time() - t0

    model.eval()
    y_pred = model.predict(Xte_t).numpy()
    return {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "f1_macro": float(f1_score(y_te, y_pred, average="macro")),
        "n_trees_total": int(num_layers * num_trees),
        "depth": depth,
        "wall_s": round(wall, 2),
        "final_loss": float(loss.item()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Reproduce NODE results.")
    p.add_argument("--dataset", default="iris", choices=list(DATASETS))
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--num_trees", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--test_fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="../results")
    args = p.parse_args()

    X, y = load_dataset(args.dataset)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_fraction, random_state=args.seed, stratify=y
    )

    # NODE's paper preprocesses tabular features with a quantile transform to a
    # normal distribution; reproduce that here.
    n_q = min(1000, X_tr.shape[0])
    qt = QuantileTransformer(
        output_distribution="normal", n_quantiles=n_q, random_state=args.seed
    ).fit(X_tr)
    X_tr, X_te = qt.transform(X_tr), qt.transform(X_te)

    print(f"\n=== NODE on {args.dataset} ===")
    print(f"train={X_tr.shape[0]} test={X_te.shape[0]} features={X.shape[1]} "
          f"classes={int(np.max(y) + 1)}")
    print(f"layers={args.num_layers} trees/layer={args.num_trees} depth={args.depth}")

    node_res = train_node(
        X_tr, y_tr, X_te, y_te,
        num_layers=args.num_layers,
        num_trees=args.num_trees,
        depth=args.depth,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    dt = DecisionTreeClassifier(random_state=args.seed).fit(X_tr, y_tr)
    rf = RandomForestClassifier(n_estimators=100, random_state=args.seed).fit(X_tr, y_tr)
    baselines = {
        "sklearn_decision_tree": float(accuracy_score(y_te, dt.predict(X_te))),
        "sklearn_random_forest": float(accuracy_score(y_te, rf.predict(X_te))),
    }

    print("\n--- Results ---")
    print(f"NODE     accuracy: {node_res['accuracy']:.4f}  "
          f"f1_macro: {node_res['f1_macro']:.4f}  "
          f"trees: {node_res['n_trees_total']}  ({node_res['wall_s']}s)")
    print(f"CART DT  accuracy: {baselines['sklearn_decision_tree']:.4f}")
    print(f"RF       accuracy: {baselines['sklearn_random_forest']:.4f}")

    out_dir = Path(__file__).resolve().parent / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "dataset": args.dataset,
        "config": vars(args),
        "node": node_res,
        "baselines": baselines,
    }
    out_path = out_dir / f"node_{args.dataset}.json"
    out_path.write_text(json.dumps(record, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
