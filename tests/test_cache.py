import tempfile

from zilli.cache.engine import CacheConfig, CacheEngine, noop_cache


class TestCacheConfig:
    def test_defaults(self):
        cfg = CacheConfig()
        assert cfg.enabled is True
        assert cfg.memory_size == 256
        assert cfg.ttl_seconds == 3600

    def test_custom(self):
        cfg = CacheConfig(enabled=False, memory_size=10, ttl_seconds=60)
        assert cfg.enabled is False
        assert cfg.memory_size == 10


class TestCacheEngine:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cfg = CacheConfig(
            enabled=True,
            memory_size=16,
            disk_dir=self.tmpdir,
            ttl_seconds=3600,
        )
        self.cache = CacheEngine(self.cfg)

    def test_miss_on_empty(self):
        entry = self.cache.get("hello", "planner")
        assert entry is None

    def test_set_and_get(self):
        self.cache.set("hello", "planner", "world", tokens_in=5, tokens_out=10)
        entry = self.cache.get("hello", "planner")
        assert entry is not None
        assert entry.response_text == "world"
        assert entry.tokens_in == 5
        assert entry.tokens_out == 10

    def test_miss_different_model(self):
        self.cache.set("hello", "planner", "world")
        entry = self.cache.get("hello", "executor")
        assert entry is None

    def test_miss_different_temperature(self):
        self.cache.set("hello", "planner", "world", temperature=0.0)
        entry = self.cache.get("hello", "planner", temperature=0.5)
        assert entry is None

    def test_hit_count_increments(self):
        self.cache.set("test", "m", "data")
        e1 = self.cache.get("test", "m")
        assert e1 is not None
        e2 = self.cache.get("test", "m")
        assert e2 is not None
        assert e2.hit_count >= 1

    def test_stats(self):
        stats = self.cache.stats()
        assert stats.misses >= 0
        assert stats.hits >= 0

    def test_stats_after_hit(self):
        self.cache.set("x", "m", "y")
        self.cache.get("x", "m")
        stats = self.cache.stats()
        assert stats.hits >= 1

    def test_clear(self):
        self.cache.set("a", "m", "b")
        self.cache.clear()
        entry = self.cache.get("a", "m")
        assert entry is None

    def test_invalidate(self):
        self.cache.set("k", "m", "v")
        self.cache.invalidate("k", "m")
        entry = self.cache.get("k", "m")
        assert entry is None

    def test_disk_persistence(self):
        self.cache.set("persist", "m", "data")
        cache2 = CacheEngine(self.cfg)
        entry = cache2.get("persist", "m")
        assert entry is not None
        assert entry.response_text == "data"

    def test_memory_trim(self):
        small_cfg = CacheConfig(
            enabled=True, memory_size=2, disk_dir=self.tmpdir, ttl_seconds=3600,
        )
        c = CacheEngine(small_cfg)
        c.set("a", "m", "1")
        c.set("b", "m", "2")
        c.set("c", "m", "3")
        assert len(c._memory) <= 2

    def test_disable(self):
        off = CacheEngine(CacheConfig(enabled=False))
        off.set("x", "m", "y")
        entry = off.get("x", "m")
        assert entry is None

    def test_large_entry_skipped(self):
        cfg = CacheConfig(enabled=True, max_entry_size=10, disk_dir=self.tmpdir)
        c = CacheEngine(cfg)
        c.set("big", "m", "x" * 20)
        entry = c.get("big", "m")
        assert entry is None


class TestNoopCache:
    def test_noop_returns_disabled(self):
        c = noop_cache()
        assert c.config.enabled is False

    def test_noop_get_returns_none(self):
        c = noop_cache()
        assert c.get("x", "m") is None

    def test_noop_set_does_nothing(self):
        c = noop_cache()
        c.set("x", "m", "y")
        assert c.get("x", "m") is None


class TestCacheDisabled:
    def test_disabled_accepts_config(self):
        from zilli.cache import CacheConfig, CacheEngine
        cfg = CacheConfig(enabled=False)
        c = CacheEngine(cfg)
        assert c.config.enabled is False

    def test_imports(self):
        from zilli.cache import CacheEntry, CacheStats
        assert CacheEntry is not None
        assert CacheStats is not None
