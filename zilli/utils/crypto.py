from __future__ import annotations

import hashlib
import hmac


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def verify_api_key(key: str, key_hash: str) -> bool:
    return hmac.compare_digest(hash_api_key(key), key_hash)
