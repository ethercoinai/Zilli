from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from zilli.privacy.classifier import CLASS_LEVEL, DataClass


class CloudProvider(str, Enum):
    NONE = "none"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    SAKANA = "sakana"
    CUSTOM = "custom"


@dataclass
class SanitizationRule:
    mask_char: str = "***"
    mask_emails: bool = True
    mask_phones: bool = True
    mask_ids: bool = True
    mask_credit_cards: bool = True
    mask_addresses: bool = True
    mask_api_keys: bool = True
    preserve_structure: bool = True
    max_field_length: int = 100_000


@dataclass
class DataGovernancePolicy:
    tenant_id: str = "default"
    max_allowed_class: DataClass = DataClass.CONFIDENTIAL
    allowed_cloud_providers: list[CloudProvider] = field(default_factory=lambda: [
        CloudProvider.OPENAI, CloudProvider.ANTHROPIC,
    ])
    sanitization: SanitizationRule = field(default_factory=SanitizationRule)
    require_consent: bool = False
    audit_all_calls: bool = True
    retention_days: int = 90
    auto_sanitize: bool = True
    notify_on_breach: bool = True
    compliance_mode: str = "standard"

    def allows_cloud_for(self, data_class: DataClass) -> bool:
        if self.max_allowed_class == DataClass.PUBLIC:
            return data_class == DataClass.PUBLIC
        if self.max_allowed_class == DataClass.INTERNAL:
            return data_class in (DataClass.PUBLIC, DataClass.INTERNAL)
        if self.max_allowed_class == DataClass.CONFIDENTIAL:
            return CLASS_LEVEL[data_class] <= CLASS_LEVEL[DataClass.CONFIDENTIAL]
        if self.max_allowed_class == DataClass.RESTRICTED:
            return CLASS_LEVEL[data_class] <= CLASS_LEVEL[DataClass.RESTRICTED]
        return False

    def max_class_can_use_cloud(self) -> DataClass:
        if self.max_allowed_class == DataClass.PUBLIC:
            return DataClass.PUBLIC
        if self.max_allowed_class == DataClass.INTERNAL:
            return DataClass.INTERNAL
        if self.max_allowed_class == DataClass.CONFIDENTIAL:
            return DataClass.CONFIDENTIAL
        if self.max_allowed_class == DataClass.RESTRICTED:
            return DataClass.RESTRICTED
        return DataClass.PUBLIC


class PolicyStore:
    def __init__(self, path: Optional[str] = None):
        self._policies: dict[str, DataGovernancePolicy] = {}
        self._path = Path(path) if path else None
        if self._path and self._path.exists():
            self._load()

    def get(self, tenant_id: str) -> DataGovernancePolicy:
        return self._policies.get(tenant_id, DataGovernancePolicy(tenant_id=tenant_id))

    def set(self, tenant_id: str, policy: DataGovernancePolicy):
        self._policies[tenant_id] = policy
        if self._path:
            self._save()

    def list_tenants(self) -> list[str]:
        return list(self._policies.keys())

    def _save(self):
        data = {
            tid: {
                "tenant_id": p.tenant_id,
                "max_allowed_class": p.max_allowed_class.value,
                "allowed_cloud_providers": [c.value for c in p.allowed_cloud_providers],
                "require_consent": p.require_consent,
                "audit_all_calls": p.audit_all_calls,
                "retention_days": p.retention_days,
                "auto_sanitize": p.auto_sanitize,
                "compliance_mode": p.compliance_mode,
            }
            for tid, p in self._policies.items()
        }
        self._path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def _load(self):
        data = json.loads(self._path.read_text())  # noqa: SYNC101
        for tid, d in data.items():
            self._policies[tid] = DataGovernancePolicy(
                tenant_id=d.get("tenant_id", tid),
                max_allowed_class=DataClass(d.get("max_allowed_class", "confidential")),
                allowed_cloud_providers=[
                    CloudProvider(c) for c in d.get("allowed_cloud_providers", [])
                ],
                require_consent=d.get("require_consent", True),
                audit_all_calls=d.get("audit_all_calls", True),
                retention_days=d.get("retention_days", 90),
                auto_sanitize=d.get("auto_sanitize", True),
                compliance_mode=d.get("compliance_mode", "standard"),
            )
