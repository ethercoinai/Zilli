from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger("zilli.privacy.consent")


class ConsentStatus(str, Enum):
    GRANTED = "granted"
    DENIED = "denied"
    EXPIRED = "expired"
    REVOKED = "revoked"
    PENDING = "pending"


class DataUse(str, Enum):
    LOCAL_INFERENCE = "local_inference"
    CLOUD_INFERENCE = "cloud_inference"
    TRAINING = "training"
    DISTILLATION = "distillation"
    AUDIT = "audit"
    ANONYMIZED_ANALYTICS = "anonymized_analytics"


@dataclass
class ConsentRecord:
    tenant_id: str
    user_id: str
    data_use: DataUse
    status: ConsentStatus
    granted_at: float
    expires_at: float
    revoked_at: Optional[float] = None
    purpose: str = ""


class ConsentManager:
    def __init__(self):
        self._records: list[ConsentRecord] = []

    def grant(self, tenant_id: str, user_id: str, data_use: DataUse,
              ttl_days: int = 365, purpose: str = "") -> ConsentRecord:
        now = time.time()
        record = ConsentRecord(
            tenant_id=tenant_id,
            user_id=user_id,
            data_use=data_use,
            status=ConsentStatus.GRANTED,
            granted_at=now,
            expires_at=now + ttl_days * 86400,
            purpose=purpose,
        )
        self._records.append(record)
        logger.info("Consent granted: %s/%s for %s", tenant_id, user_id, data_use.value)
        return record

    def revoke(self, tenant_id: str, user_id: str, data_use: DataUse) -> bool:
        for r in reversed(self._records):
            if (r.tenant_id == tenant_id and r.user_id == user_id
                    and r.data_use == data_use and r.status == ConsentStatus.GRANTED):
                r.status = ConsentStatus.REVOKED
                r.revoked_at = time.time()
                logger.info("Consent revoked: %s/%s for %s", tenant_id, user_id, data_use.value)
                return True
        return False

    def check(self, tenant_id: str, user_id: str, data_use: DataUse) -> bool:
        now = time.time()
        for r in reversed(self._records):
            if (r.tenant_id == tenant_id and r.user_id == user_id
                    and r.data_use == data_use):
                if r.status == ConsentStatus.GRANTED and r.expires_at > now:
                    return True
                if r.status == ConsentStatus.REVOKED:
                    return False
        return False

    def list_for_user(self, tenant_id: str, user_id: str) -> list[ConsentRecord]:
        return [
            r for r in self._records
            if r.tenant_id == tenant_id and r.user_id == user_id
        ]

    def list_active(self, tenant_id: str) -> list[ConsentRecord]:
        now = time.time()
        return [
            r for r in self._records
            if r.tenant_id == tenant_id
            and r.status == ConsentStatus.GRANTED
            and r.expires_at > now
        ]

    def expire_all(self, tenant_id: str, user_id: str):
        now = time.time()
        for r in self._records:
            if r.tenant_id == tenant_id and r.user_id == user_id:
                if r.status == ConsentStatus.GRANTED and r.expires_at > now:
                    r.expires_at = now
                    r.status = ConsentStatus.EXPIRED
                    logger.info("Consent expired: %s/%s for %s", tenant_id, user_id, r.data_use.value)
