from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("zilli.cache")


@dataclass
class CacheConfig:
    enabled: bool = True
    memory_size: int = 256
    disk_dir: str = "./.zilli_cache"
    ttl_seconds: int = 3600
    max_entry_size: int = 100000
    disk_persistence: bool = False


@dataclass
class CacheEntry:
    prompt_hash: str
    response_text: str
    model_name: str
    tokens_in: int = 0
    tokens_out: int = 0
    created_at: float = 0.0
    hit_count: int = 0

    def is_expired(self, ttl: int) -> bool:
        return time.time() - self.created_at > ttl


@dataclass
class CacheStats:
    entries: int = 0
    hits: int = 0
    misses: int = 0
    memory_entries: int = 0
    disk_entries: int = 0


def _hash_key(prompt: str, model_name: str, temperature: float) -> str:
    raw = f"{model_name}||{temperature}||{prompt}"
    return hashlib.sha256(raw.encode()).hexdigest()


class CacheEngine:
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._memory: dict[str, CacheEntry] = {}
        self._hits = 0
        self._misses = 0
        self._disk_count = 0
        self._lock = threading.Lock()
        self._disk_dir = Path(self.config.disk_dir)
        if self.config.enabled:
            self._disk_dir.mkdir(parents=True, exist_ok=True)
            self._load_index()

    def _disk_path(self, key: str) -> Path:
        return self._disk_dir / f"{key}.json"

    def _load_index(self):
        index_path = self._disk_dir / "_index.json"
        if index_path.exists():
            try:
                keys = json.loads(index_path.read_text())
                for key in keys[-self.config.memory_size:]:
                    dp = self._disk_path(key)
                    if dp.exists():
                        try:
                            data = json.loads(dp.read_text())
                            entry = CacheEntry(**data)
                            self._memory[key] = entry
                        except Exception:
                            pass
            except Exception:
                pass

    def _save_index(self):
        if not self.config.enabled:
            return
        index_path = self._disk_dir / "_index.json"
        try:
            index_path.write_text(json.dumps(list(self._memory.keys()), indent=2))
        except OSError as e:
            logger.warning("Failed to save cache index: %s", e)

    def get(self, prompt: str, model_name: str, temperature: float = 0.0) -> Optional[CacheEntry]:
        if not self.config.enabled:
            return None

        key = _hash_key(prompt, model_name, temperature)

        with self._lock:
            entry = self._memory.get(key)
            if entry is not None:
                if entry.is_expired(self.config.ttl_seconds):
                    del self._memory[key]
                    self._misses += 1
                    return None
                entry.hit_count += 1
                self._touch(key)
                self._hits += 1
                return entry

            dp = self._disk_path(key)
            if dp.exists():
                try:
                    data = json.loads(dp.read_text())
                    entry = CacheEntry(**data)
                    if entry.is_expired(self.config.ttl_seconds):
                        dp.unlink(missing_ok=True)
                        self._misses += 1
                        return None
                    entry.hit_count += 1
                    self._memory[key] = entry
                    self._trim_memory()
                    self._hits += 1
                    return entry
                except Exception:
                    pass

            self._misses += 1
            return None

    def set(self, prompt: str, model_name: str, response_text: str,
            tokens_in: int = 0, tokens_out: int = 0,
            temperature: float = 0.0):
        if not self.config.enabled:
            return
        if len(response_text) > self.config.max_entry_size:
            return

        key = _hash_key(prompt, model_name, temperature)
        entry = CacheEntry(
            prompt_hash=key,
            response_text=response_text,
            model_name=model_name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            created_at=time.time(),
            hit_count=0,
        )

        with self._lock:
            self._memory[key] = entry
            self._trim_memory()

        dp = self._disk_path(key)
        try:
            dp.write_text(json.dumps({
                "prompt_hash": key,
                "response_text": response_text,
                "model_name": model_name,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "created_at": entry.created_at,
                "hit_count": 0,
            }, ensure_ascii=False))
        except OSError as e:
            logger.warning("Failed to write cache entry: %s", e)

        self._save_index()
        self._disk_count += 1

    def _touch(self, key: str):
        entry = self._memory.get(key)
        if entry is not None:
            entry.hit_count += 1

    def _trim_memory(self):
        while len(self._memory) > self.config.memory_size:
            oldest = min(self._memory.keys(), key=lambda k: self._memory[k].hit_count)
            del self._memory[oldest]

    def invalidate(self, prompt: str, model_name: str, temperature: float = 0.0):
        key = _hash_key(prompt, model_name, temperature)
        with self._lock:
            self._memory.pop(key, None)
        dp = self._disk_path(key)
        dp.unlink(missing_ok=True)

    def clear(self):
        self._memory.clear()
        for f in self._disk_dir.glob("*.json"):
            f.unlink(missing_ok=True)
        self._hits = 0
        self._misses = 0

    def stats(self) -> CacheStats:
        return CacheStats(
            entries=len(self._memory),
            hits=self._hits,
            misses=self._misses,
            memory_entries=len(self._memory),
            disk_entries=max(0, self._disk_count),
        )


def noop_cache() -> CacheEngine:
    return CacheEngine(config=CacheConfig(enabled=False))
