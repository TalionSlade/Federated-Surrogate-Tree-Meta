# Neural / Differentiable Decision Trees

Reproductions of the prominent **neural tree** architectures — hybrids that keep
the interpretability of decision trees while being end-to-end differentiable
(tree splits become soft activations, so cut-points and leaf classifiers train
by SGD).

This is a sibling track to the FedRF-Distill federated random-forest work in the
parent repo. The long-term aim is to evaluate whether differentiable trees are
viable local models / surrogates inside the federated distillation pipeline.

## Models

| Model | Paper | Domain | Split activation | Status |
|-------|-------|--------|------------------|--------|
| **DNDT** | Yang et al., 2018 (arXiv:1806.06988) | Tabular | Softmax soft-binning | ✅ implemented |
| **NODE** | Popov et al., 2020 (arXiv:1909.06312) | Tabular | Entmax (sparse) | ✅ implemented |
| **NBDT** | Wan et al., 2021 (arXiv:2004.00221) | Vision | Hierarchical softmax | ⏳ planned |
| **dNDF** | Kontschieder et al., 2015 (ICCV) | Vision | Sigmoid gates | ⏳ planned |

Reference repos:
- DNDT (TF): https://github.com/wOOL/DNDT
- NODE (PyTorch): https://github.com/Qwicen/node
- NBDT (PyTorch): https://github.com/alvinwan/neural-backed-decision-trees
- dNDF (PyTorch): https://github.com/jingxil/Neural-Decision-Forests

## DNDT — Deep Neural Decision Trees

A decision tree realised entirely as a neural net:
1. **Soft binning** routes each feature into `n_cut+1` bins via a temperature
   softmax over `W·x + b` (W constant, b = cumulative cut points).
2. **Kronecker product** of per-feature bin memberships gives a soft membership
   over all `∏(n_cut+1)` leaves.
3. A learned **leaf-score matrix** maps leaves to class logits.

Leaf count grows multiplicatively with feature count, so DNDT is meant for
low-dimensional tabular data (the paper uses small UCI sets). For higher-dim
data use `--feature_topk` to keep only the top-k features by mutual information.

### Run

```bash
# from neural_trees/dndt/
python train.py --dataset iris   --n_cut 1 --epochs 1000
python train.py --dataset wine   --n_cut 1 --epochs 2000
python train.py --dataset breast_cancer --n_cut 1 --feature_topk 6
```

Results (accuracy, F1, leaf count, wall time + sklearn DT/RF baselines) are
written to `neural_trees/results/dndt_<dataset>.json`.

### Files
- `dndt/dndt.py` — the `DNDT` model (soft binning, Kronecker routing, leaf scores)
- `dndt/train.py` — train/eval harness with CART + RF baselines
- `dndt/run_all.py` — sweep over iris / wine / breast_cancer

## NODE — Neural Oblivious Decision Ensembles

A strong deep tabular baseline. Building block is the **ODST** (Oblivious
Differentiable Sparse Tree) layer:
1. **Sparse feature selection** — each (tree, level) picks its splitting
   feature via `entmax` (alpha=1.5) over inputs; most weights go to exactly
   zero, so trees select a small discrete feature set while staying
   differentiable.
2. **Soft splits** — 2-way `entmax` over `±(value − threshold)·temp`.
3. **Oblivious trees** — same (feature, threshold) at every node of a level →
   `2**depth` leaves (a decision table); a learned response tensor maps leaves
   to outputs.
4. Layers stacked **DenseNet-style** (each layer sees the input + all previous
   layer outputs); final tree outputs averaged into class logits.

Unlike DNDT, NODE handles high-dimensional features natively — no subsetting.
Features are quantile-transformed to normal (as in the paper).

### Run

```bash
# from neural_trees/node/
python train.py --dataset iris          --epochs 80
python train.py --dataset wine          --epochs 100
python train.py --dataset breast_cancer --epochs 100
python run_all.py    # all three
```

Tune with `--num_layers`, `--num_trees`, `--depth`, `--lr`, `--batch_size`.
Results → `neural_trees/results/node_<dataset>.json`.

### Files
- `node/node.py` — `ODST`, `DenseBlock`, `NODEClassifier`
- `node/train.py` — train/eval harness (quantile transform + CART/RF baselines)
- `node/run_all.py` — sweep over iris / wine / breast_cancer

## Reproduction Results (single 80/20 split)

| Dataset | DNDT | NODE | CART | RF |
|---------|------|------|------|-----|
| iris | 0.9333 | **0.9667** | 0.9333 | 0.9000 |
| wine | 1.0000¹ | 0.9722 | 0.9444 | 1.0000 |
| breast_cancer | 0.9386¹ | 0.9561 | 0.9211² | 0.9474² |

¹ DNDT with top-6 features (leaf count = ∏ bins explodes otherwise).
² Baselines differ slightly between DNDT/NODE rows due to different feature
preprocessing (DNDT: standardize; NODE: quantile-normal). NODE uses all
features in every case.
