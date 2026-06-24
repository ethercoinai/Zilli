from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ModelRole(str, Enum):
    PLANNER = "planner"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"

    def __str__(self) -> str:
        return self.value


class DeploymentType(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"


class ModelConfig(BaseModel):
    name: str = Field(..., description="Logical name for this model endpoint")
    backend: str = Field("ollama", description="Backend type: ollama, vllm, llamacpp")
    model_id: str = Field(..., description="Model identifier (e.g. qwen3:27b)")
    role: ModelRole = Field(..., description="Role this model serves")
    deployment: DeploymentType = Field(
        DeploymentType.LOCAL,
        description="Where this model runs: local (on-prem) or cloud",
    )
    base_url: str = Field("http://127.0.0.1:11434", description="API base URL")
    max_tokens: int = 2048
    temperature: float = 0.1
    num_ctx: int = 4096
    cost_per_call: float = 0.0
    is_fallback: bool = False

    model_config = ConfigDict(extra="forbid")


class ModelProfile(BaseModel):
    models: list[ModelConfig] = Field(
        default_factory=lambda: [
            ModelConfig(
                name="planner",
                model_id="qwen3:27b",
                role=ModelRole.PLANNER,
                max_tokens=4096,
                temperature=0.2,
                num_ctx=8192,
            ),
            ModelConfig(
                name="executor",
                model_id="qwen3:7b",
                role=ModelRole.EXECUTOR,
                max_tokens=4096,
                temperature=0.1,
                num_ctx=4096,
            ),
            ModelConfig(
                name="reviewer",
                model_id="qwen3:27b",
                role=ModelRole.REVIEWER,
                max_tokens=2048,
                temperature=0.1,
                num_ctx=8192,
            ),
        ]
    )

    monthly_budget_usd: float = 500.0
    fallback_strategy: str = Field(
        "lower_tier",
        description="What to do when primary model fails: lower_tier | cache | error"
    )
