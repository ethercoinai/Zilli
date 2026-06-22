import tempfile
from pathlib import Path

import yaml

from zilli.configs.loader import ZilliConfig, load_config


def _write_cfg(data: dict) -> Path:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(data, tmp)
    tmp.close()
    return Path(tmp.name)


class TestZilliConfig:
    def test_default_config(self):
        cfg = ZilliConfig()
        assert cfg.version == "0.1.0"
        assert cfg.models.profile is None
        assert cfg.routing.classifier.long_request_threshold == 500
        assert cfg.security.pii.custom_patterns == {}
        assert cfg.audit.log_dir == "./audit_logs"
        assert cfg.training.algorithm == "CISPO"

    def test_from_yaml_minimal(self):
        path = _write_cfg({"version": "0.2.0"})
        cfg = ZilliConfig.from_yaml(path)
        assert cfg.version == "0.2.0"
        path.unlink()

    def test_from_yaml_with_models(self):
        path = _write_cfg({
            "version": "0.2.0",
            "models": {
                "monthly_budget_usd": 1000.0,
                "fallback_strategy": "lower_tier",
                "models": [
                    {"name": "p1", "model_id": "m1", "role": "planner"},
                    {"name": "e1", "model_id": "m2", "role": "executor"},
                ],
            },
        })
        cfg = ZilliConfig.from_yaml(path)
        assert cfg.to_model_profile().monthly_budget_usd == 1000.0
        profile = cfg.to_model_profile()
        assert len(profile.models) == 2
        path.unlink()

    def test_from_yaml_with_routing(self):
        path = _write_cfg({
            "routing": {
                "classifier": {
                    "long_request_threshold": 300,
                    "rules": [
                        {"pattern": "(?i)(test)", "route": "full_route"},
                    ],
                },
            },
        })
        cfg = ZilliConfig.from_yaml(path)
        assert cfg.routing.classifier.long_request_threshold == 300
        assert len(cfg.routing.classifier.rules) == 1
        assert cfg.routing.classifier.rules[0].pattern == "(?i)(test)"
        path.unlink()

    def test_from_yaml_with_audit(self):
        path = _write_cfg({
            "audit": {
                "log_dir": "/tmp/zilli_audit",
                "sanitize": False,
            },
        })
        cfg = ZilliConfig.from_yaml(path)
        assert cfg.audit.log_dir == "/tmp/zilli_audit"
        assert cfg.audit.sanitize is False
        path.unlink()

    def test_from_yaml_with_training(self):
        path = _write_cfg({
            "training": {
                "algorithm": "GRPO",
                "clip_range": 0.3,
            },
        })
        cfg = ZilliConfig.from_yaml(path)
        assert cfg.training.algorithm == "GRPO"
        assert cfg.training.clip_range == 0.3
        path.unlink()

    def test_to_model_profile_empty(self):
        cfg = ZilliConfig()
        profile = cfg.to_model_profile()
        assert profile.monthly_budget_usd == 500.0
        assert len(profile.models) == 3

    def test_to_training_dict(self):
        cfg = ZilliConfig(training={"algorithm": "CISPO", "clip_range": 0.2})
        d = cfg.to_training_dict()
        assert d["algorithm"] == "CISPO"
        assert d["clip_range"] == 0.2

    def test_load_config_no_file(self):
        cfg = load_config()
        assert isinstance(cfg, ZilliConfig)

    def test_load_config_custom_path(self):
        path = _write_cfg({"version": "0.2.0"})
        cfg = load_config(path)
        assert cfg.version == "0.2.0"
        path.unlink()

    def test_load_config_file_not_found(self):
        import pytest
        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/zilli.yaml"))

    def test_load_config_invalid_yaml(self):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        tmp.write("not: yaml: [broken\n")
        tmp.close()
        path = Path(tmp.name)
        import pytest
        with pytest.raises(Exception):
            load_config(path)
        path.unlink()

    def test_config_models_profile_propagation(self):
        path = _write_cfg({
            "models": {
                "profile": {
                    "monthly_budget_usd": 200.0,
                    "models": [
                        {"name": "x", "model_id": "y", "role": "planner"},
                    ],
                },
            },
        })
        cfg = ZilliConfig.from_yaml(path)
        profile = cfg.to_model_profile()
        assert profile.monthly_budget_usd == 200.0
        assert len(profile.models) == 1
        path.unlink()

    def test_security_config(self):
        path = _write_cfg({
            "security": {
                "isolation_default_policy": {
                    "access_level": "restricted",
                    "allowed_roles": ["planner"],
                },
            },
        })
        cfg = ZilliConfig.from_yaml(path)
        assert cfg.security.isolation_default_policy.access_level == "restricted"
        assert cfg.security.isolation_default_policy.allowed_roles == ["planner"]
        path.unlink()

    def test_industry_config(self):
        path = _write_cfg({
            "industry": {
                "workflows": {
                    "legal": {"access_level": "restricted", "retention_days": 180},
                },
            },
        })
        cfg = ZilliConfig.from_yaml(path)
        assert cfg.industry.workflows["legal"].access_level == "restricted"
        assert cfg.industry.workflows["legal"].retention_days == 180
        path.unlink()


class TestConfigIntegration:
    def test_model_registry_accepts_config(self):
        path = _write_cfg({
            "models": {
                "models": [
                    {"name": "p", "model_id": "m1", "role": "planner"},
                    {"name": "e", "model_id": "m2", "role": "executor"},
                ],
            },
        })
        cfg = ZilliConfig.from_yaml(path)
        from zilli.models.registry import ModelRegistry
        registry = ModelRegistry(config=cfg)
        assert registry.profile.monthly_budget_usd == 500.0
        models = registry.list_models()
        assert len(models) == 2
        path.unlink()

    def test_scheduler_accepts_config(self):
        path = _write_cfg({
            "models": {
                "monthly_budget_usd": 300.0,
                "models": [
                    {"name": "p", "model_id": "m1", "role": "planner", "cost_per_call": 0.02},
                ],
            },
        })
        cfg = ZilliConfig.from_yaml(path)
        from zilli.adaptive.sota_scheduler import DynamicSOTAScheduler
        s = DynamicSOTAScheduler(config=cfg)
        assert s.monthly_budget == 300.0
        assert s.cost_per_call.get("p") == 0.02
        path.unlink()

    def test_cost_controller_accepts_config(self):
        import tempfile as tf

        path = _write_cfg({
            "models": {
                "monthly_budget_usd": 400.0,
                "models": [
                    {"name": "p", "model_id": "m1", "role": "planner"},
                ],
            },
        })
        cfg = ZilliConfig.from_yaml(path)
        from zilli.envs.cost_controller import CostController
        tmp = tf.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        cc = CostController(budget_file=tmp.name, config=cfg)
        assert cc.scheduler.monthly_budget == 400.0
        Path(tmp.name).unlink(missing_ok=True)
        path.unlink()

    def test_route_classifier_accepts_config(self):
        path = _write_cfg({
            "routing": {
                "classifier": {
                    "long_request_threshold": 100,
                    "rules": [
                        {"pattern": "test", "route": "full_route"},
                    ],
                },
            },
        })
        cfg = ZilliConfig.from_yaml(path)
        from zilli.routing.classifier import RouteClassifier, RouteType
        classifier = RouteClassifier(config=cfg)
        assert classifier.long_request_threshold == 100
        # The config-based rules should include "test" -> full_route
        d = classifier.classify("this is a test")
        assert d.route == RouteType.FULL_ROUTE
        path.unlink()
