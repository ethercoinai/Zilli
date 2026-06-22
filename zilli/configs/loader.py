from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

from zilli.models.config import ModelConfig, ModelProfile

_DEFAULT_CONFIG_PATHS = [
    Path.home() / ".zilli.yaml",
    Path(__file__).parent / "model_config.yaml",
]


class ClassifierRule(BaseModel):
    pattern: str
    route: str = "full_route"

    model_config = ConfigDict(extra="forbid")


class ClassifierConfig(BaseModel):
    rules: list[ClassifierRule] = Field(default_factory=list)
    long_request_threshold: int = 500

    model_config = ConfigDict(extra="forbid")


class RoutingConfig(BaseModel):
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)

    model_config = ConfigDict(extra="forbid")


class PIIConfig(BaseModel):
    custom_patterns: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class IsolationPolicy(BaseModel):
    access_level: str = "internal"
    allowed_roles: list[str] = Field(default_factory=lambda: ["planner", "executor", "reviewer"])
    max_input_length: int = 32768
    require_sanitization: bool = True
    audit_required: bool = True
    retention_days: int = 90

    model_config = ConfigDict(extra="forbid")


class SecurityConfig(BaseModel):
    pii: PIIConfig = Field(default_factory=PIIConfig)
    isolation_default_policy: IsolationPolicy = Field(
        default_factory=IsolationPolicy, alias="isolation_default_policy",
    )

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class WorkflowEntry(BaseModel):
    access_level: str = "confidential"
    require_audit: bool = True
    require_sanitization: bool = True
    retention_days: int = 90

    model_config = ConfigDict(extra="forbid")


class IndustryConfig(BaseModel):
    workflows: dict[str, WorkflowEntry] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class AuditConfig(BaseModel):
    log_dir: str = "./audit_logs"
    sanitize: bool = True

    model_config = ConfigDict(extra="forbid")


class TrainingConfig(BaseModel):
    algorithm: str = "CISPO"
    clip_range: float = 0.2
    kl_penalty: float = 0.01
    is_weight_cap: float = 5.0
    gamma: float = 0.99

    model_config = ConfigDict(extra="forbid")


class ModelsConfig(BaseModel):
    profile: Optional[ModelProfile] = None
    custom_backends: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class ZilliConfig(BaseModel):
    version: str = "0.1.0"
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    industry: IndustryConfig = Field(default_factory=IndustryConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_yaml(cls, path: Path) -> "ZilliConfig":
        raw = _load_yaml(path)
        return cls._parse(raw)

    @classmethod
    def _parse(cls, raw: dict) -> "ZilliConfig":
        parsed: dict[str, Any] = {}

        if "version" in raw:
            parsed["version"] = raw["version"]

        models_raw = raw.get("models", {})
        if models_raw:
            profile_raw = models_raw.get("profile", {})
            if "models" in profile_raw:
                model_configs = [ModelConfig(**m) for m in profile_raw["models"]]
                profile = ModelProfile(
                    models=model_configs,
                    monthly_budget_usd=profile_raw.get("monthly_budget_usd", 500.0),
                    fallback_strategy=profile_raw.get("fallback_strategy", "lower_tier"),
                )
                parsed["models"] = {"profile": profile}
            else:
                flat_models = models_raw.get("models", [])
                if flat_models:
                    model_configs = [ModelConfig(**m) for m in flat_models]
                    profile = ModelProfile(
                        models=model_configs,
                        monthly_budget_usd=models_raw.get("monthly_budget_usd", 500.0),
                        fallback_strategy=models_raw.get("fallback_strategy", "lower_tier"),
                    )
                    parsed["models"] = {"profile": profile}

        if "routing" in raw:
            parsed["routing"] = raw["routing"]
        if "security" in raw:
            parsed["security"] = raw["security"]
        if "industry" in raw:
            parsed["industry"] = raw["industry"]
        if "audit" in raw:
            parsed["audit"] = raw["audit"]
        if "training" in raw:
            parsed["training"] = raw["training"]

        return cls(**parsed)

    def to_model_profile(self) -> ModelProfile:
        if self.models and self.models.profile:
            return self.models.profile
        return ModelProfile()

    def to_training_dict(self) -> dict:
        return self.training.model_dump(exclude_none=True)


def _load_yaml(path: Path) -> dict:
    resolved = path.resolve()
    if ".." in str(resolved.relative_to(resolved.anchor)):
        raise ValueError(f"Path traversal detected in config path: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a top-level mapping, got {type(data).__name__}")
    return data


def load_config(path: Optional[Path] = None) -> ZilliConfig:
    if path is not None:
        return ZilliConfig.from_yaml(path)

    for candidate in _DEFAULT_CONFIG_PATHS:
        if candidate.exists():
            return ZilliConfig.from_yaml(candidate)

    return ZilliConfig()
