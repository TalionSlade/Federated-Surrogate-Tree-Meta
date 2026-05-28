"""Pydantic v2 schema for FedRF-Distill experiments.

A single :class:`ExperimentConfig` captures every knob: dataset, partitioning,
privacy mechanism, model frameworks per client (heterogeneous), aggregation,
meta-learning, and round-loop hyper-params. This makes experiments fully
reproducible from a YAML file.

Loading
-------
>>> import yaml
>>> from fedrf_distill.config import ExperimentConfig
>>> with open("experiments/configs/adult-dp1.yaml") as f:
...     cfg = ExperimentConfig.model_validate(yaml.safe_load(f))
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fedrf_distill.distillation.surrogate import DistillationMode


# ── Atomic config blocks ────────────────────────────────────────────────────
class DataConfig(BaseModel):
    """Which dataset, with which preprocessing."""

    model_config = ConfigDict(frozen=False, extra="forbid")
    name: str
    """Registry key, e.g. 'adult', 'covertype'."""
    test_fraction: float = 0.2
    val_fraction: float = 0.0
    random_state: int = 0


class PartitionConfig(BaseModel):
    """How to slice train data across K clients."""

    model_config = ConfigDict(frozen=False, extra="forbid")
    strategy: Literal["iid", "dirichlet", "pathological", "quantity_skew"] = "dirichlet"
    n_clients: int = 10
    alpha: float = 0.5  # used by dirichlet
    k_classes_per_client: int = 2  # used by pathological
    skew_exponent: float = 1.5  # used by quantity_skew
    seed: int = 0


class ProxyConfig(BaseModel):
    model_config = ConfigDict(frozen=False, extra="forbid")
    source: Literal["synthetic_uniform", "synthetic_gaussian", "external"] = (
        "synthetic_uniform"
    )
    n_samples: int = 2000
    seed: int = 0


class ModelConfig(BaseModel):
    """Teacher / student / per-client framework configuration."""

    model_config = ConfigDict(frozen=False, extra="forbid")
    teacher_framework: str = "sklearn_rf"
    teacher_kwargs: dict[str, Any] = Field(default_factory=dict)
    student_framework: str = "sklearn_dt"
    student_kwargs: dict[str, Any] = Field(default_factory=dict)

    # When non-empty, override the teacher framework on a per-client basis.
    # ``frameworks_per_client[k]`` must match the key registered with the model
    # factory. Used to demonstrate cross-framework heterogeneity (Hook B).
    frameworks_per_client: list[str] = Field(default_factory=list)


class DistillationConfig(BaseModel):
    model_config = ConfigDict(frozen=False, extra="forbid")
    mode: DistillationMode = DistillationMode.ARGMAX_WEIGHTED
    temperature: float = 1.0
    floor: float = 0.0


class DPConfig(BaseModel):
    """Differential privacy configuration (Hook A)."""

    model_config = ConfigDict(frozen=False, extra="forbid")
    mechanism: Literal["null", "laplace", "gaussian"] = "laplace"
    epsilon_per_round: float = 1.0
    delta_per_round: float = 1e-5
    target_epsilon: float = 8.0
    target_delta: float = 1e-5
    accountant: Literal["basic", "rdp"] = "basic"
    seed: int = 0


class AggregatorConfig(BaseModel):
    model_config = ConfigDict(frozen=False, extra="forbid")
    strategy: Literal[
        "proxy_redistillation", "confidence_weighted", "majority_vote"
    ] = "proxy_redistillation"
    student_framework: str = "sklearn_dt"
    student_kwargs: dict[str, Any] = Field(default_factory=dict)


class MetaConfig(BaseModel):
    model_config = ConfigDict(frozen=False, extra="forbid")
    enabled: bool = True
    strategy: Literal["stacking", "none"] = "stacking"
    standardise_proba_cols: bool = False


# ── Top-level experiment config ─────────────────────────────────────────────
class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    model_config = ConfigDict(frozen=False, extra="forbid")
    name: str
    output_dir: str = "experiments/results"
    n_rounds: int = 10
    random_state: int = 42

    data: DataConfig
    partition: PartitionConfig = Field(default_factory=PartitionConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    distillation: DistillationConfig = Field(default_factory=DistillationConfig)
    dp: DPConfig = Field(default_factory=DPConfig)
    aggregator: AggregatorConfig = Field(default_factory=AggregatorConfig)
    meta: MetaConfig = Field(default_factory=MetaConfig)

    @field_validator("n_rounds")
    @classmethod
    def _positive_rounds(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("n_rounds must be > 0")
        return v
