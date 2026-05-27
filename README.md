# FedRF-Distill

**Privacy-Preserving Cross-Framework Federated Distillation for Tree Ensembles**

FedRF-Distill is a research framework for federated learning over tree-based models (Random Forests, XGBoost, LightGBM, CatBoost). It introduces two contributions over prior tree-FL work:

1. **Formal Differential Privacy** — `(ε, δ)`-DP guarantees on soft labels and surrogate models, with a privacy accountant tracking budget across rounds.
2. **Cross-Framework Heterogeneity** — Clients may locally use sklearn, XGBoost, LightGBM, or CatBoost interchangeably; the distillation/aggregation layer is model-agnostic.

## Quick Start

```bash
pip install -e ".[all-models,datasets,dev]"
```

```python
from fedrf_distill.config.schema import ExperimentConfig
from fedrf_distill.orchestration.simulator import Simulator

cfg = ExperimentConfig.from_yaml("experiments/configs/base.yaml")
sim = Simulator(cfg)
result = sim.run()
print(result.summary())
```

## Design Principles

- **Protocol-driven**: every component (client, aggregator, DP mechanism, model adapter) is defined by a `Protocol` or ABC and plug-replaceable.
- **Strict typing**: full `mypy --strict` compliance.
- **Reproducible**: every result carries `git_sha`, `seed`, `config.yaml`, `requirements.lock`.
- **Composable privacy**: DP mechanisms compose cleanly; `PrivacyAccountant` tracks `(ε, δ)` budget across rounds.

## Documentation

See `docs/index.html` for the full architecture, algorithms, privacy proofs, and 15+ flow diagrams.

## Citation

```bibtex
@inproceedings{fedrfdistill2026,
  title = {Privacy-Preserving Cross-Framework Federated Distillation for Tree Ensembles},
  year  = {2026},
  note  = {Under review}
}
```
