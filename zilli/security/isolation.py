import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger("zilli.security.isolation")


class AccessLevel(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

    def __str__(self) -> str:
        return self.value


@dataclass
class IsolationPolicy:
    tenant_id: str = "default"
    access_level: AccessLevel = AccessLevel.INTERNAL
    allowed_roles: list[str] = field(default_factory=lambda: ["planner", "executor", "reviewer"])
    max_input_length: int = 32768
    require_sanitization: bool = True
    audit_required: bool = True
    retention_days: int = 90


class DataIsolation:
    def __init__(self, default_policy: Optional[IsolationPolicy] = None):
        self.default_policy = default_policy or IsolationPolicy()
        self._tenants: dict[str, IsolationPolicy] = {}

    def register_tenant(self, tenant_id: str, policy: IsolationPolicy):
        self._tenants[tenant_id] = policy
        logger.info("Registered tenant %s with access level %s", tenant_id, policy.access_level)

    def get_policy(self, tenant_id: str) -> IsolationPolicy:
        return self._tenants.get(tenant_id, self.default_policy)

    def check_access(self, tenant_id: str, role: str) -> bool:
        policy = self.get_policy(tenant_id)
        if role not in policy.allowed_roles:
            logger.warning("Role %s not allowed for tenant %s", role, tenant_id)
            return False
        return True

    def remove_tenant(self, tenant_id: str):
        self._tenants.pop(tenant_id, None)
        logger.info("Removed tenant %s", tenant_id)

    def list_tenants(self) -> list[dict]:
        return [
            {
                "tenant_id": tid,
                "access_level": p.access_level.value,
                "allowed_roles": p.allowed_roles,
                "require_sanitization": p.require_sanitization,
            }
            for tid, p in self._tenants.items()
        ]
