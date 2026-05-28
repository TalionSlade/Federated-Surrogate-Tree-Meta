# FedRF-Distill — Evaluation Report

_Generated: 2026-05-27 21:01:25 UTC_

- **Configs**: 3
- **Datasets**: breast_cancer
- **DP mechanisms**: laplace, null


## Privacy-Utility Frontier

_Rows sorted by cumulative ε (no-DP baselines first)._

| Dataset | Config | DP | ε/round | Clients | Final Acc | Final F1 | Cum. ε | Total Bytes | Wall (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| breast_cancer | breast_cancer_nodp | null | — | 10 | 0.8947 | 0.8845 | 0.000 | 80,905 | 8.0 |
| breast_cancer | breast_cancer_dp_laplace | laplace | 0.500 | 10 | 0.4298 | 0.3959 | 25.000 | 98,665 | 7.8 |
| breast_cancer | breast_cancer_dp_laplace_eps5 | laplace | 5.000 | 10 | 0.8333 | 0.8234 | 250.000 | 88,425 | 7.8 |

## Per-Round Trajectories

### `breast_cancer_dp_laplace`

| t | Acc | F1 | AUC | Bytes | Cum. ε | Wall (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.3860 | 0.3334 | 0.4408 | 18,613 | 5.000 | 1.59 |
| 1 | 0.3333 | 0.2500 | 0.4581 | 20,533 | 10.000 | 1.56 |
| 2 | 0.3333 | 0.2500 | 0.4382 | 18,613 | 15.000 | 1.62 |
| 3 | 0.3333 | 0.2500 | 0.5445 | 20,693 | 20.000 | 1.52 |
| 4 | 0.4298 | 0.3959 | 0.3920 | 20,213 | 25.000 | 1.53 |

### `breast_cancer_dp_laplace_eps5`

| t | Acc | F1 | AUC | Bytes | Cum. ε | Wall (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.7456 | 0.7440 | 0.0933 | 17,973 | 50.000 | 1.53 |
| 1 | 0.5965 | 0.5965 | 0.2850 | 18,133 | 100.000 | 1.56 |
| 2 | 0.8333 | 0.8275 | 0.1520 | 17,973 | 150.000 | 1.61 |
| 3 | 0.7456 | 0.7423 | 0.1385 | 17,173 | 200.000 | 1.55 |
| 4 | 0.8333 | 0.8234 | 0.0935 | 17,173 | 250.000 | 1.58 |

### `breast_cancer_nodp`

| t | Acc | F1 | AUC | Bytes | Cum. ε | Wall (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.8947 | 0.8845 | 0.0770 | 16,213 | 0.000 | 1.58 |
| 1 | 0.8772 | 0.8652 | 0.0637 | 16,213 | 0.000 | 1.58 |
| 2 | 0.8947 | 0.8845 | 0.0770 | 16,053 | 0.000 | 1.65 |
| 3 | 0.8947 | 0.8845 | 0.0770 | 16,213 | 0.000 | 1.52 |
| 4 | 0.8947 | 0.8845 | 0.0770 | 16,213 | 0.000 | 1.63 |

## Membership Inference Attack (Yeom)

_AUC = 0.5 → empirical privacy preserved; ≥ 0.6 → measurable leakage._

| Config | DP | ε/round | Cum. ε | Attack AUC | Advantage | TPR@1%FPR | Mem loss | Non-mem loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| breast_cancer_dp_laplace | laplace | 0.500 | 25.000 | 0.5770 | 0.1714 | 0.1714 | 13.4115 | 14.1629 |
| breast_cancer_dp_laplace_eps5 | laplace | 5.000 | 250.000 | 0.5281 | 0.2506 | 0.2593 | 1.4248 | 2.1569 |
| breast_cancer_nodp | null | — | 0.000 | 0.5882 | 0.2835 | 0.2923 | 1.4269 | 2.2282 |
