from pathlib import Path
from unittest.mock import patch

from zilli.cli import _run_distill


class TestRunDistill:
    def test_distill_default(self, tmp_path):
        with patch("builtins.print") as mock_print:
            _run_distill(num_samples=10, log_dir=str(tmp_path))
            calls = [str(c) for c in mock_print.call_args_list]
            assert any("Distillation complete" in c for c in calls)

    def test_distill_with_checkpoint(self, tmp_path):
        ckpt = str(tmp_path / "ckpt.json")
        with patch("builtins.print"):
            _run_distill(num_samples=10, checkpoint_path=ckpt, log_dir=str(tmp_path))
        assert Path(ckpt).exists()

    def test_distill_resume_from_checkpoint(self, tmp_path):
        ckpt = str(tmp_path / "resume.json")
        with patch("builtins.print"):
            _run_distill(num_samples=10, checkpoint_path=ckpt, log_dir=str(tmp_path))
        assert Path(ckpt).exists()
        with patch("builtins.print") as mock_print:
            _run_distill(num_samples=5, checkpoint_path=ckpt, log_dir=str(tmp_path))
            calls = [str(c) for c in mock_print.call_args_list]
            assert any("Resumed from checkpoint" in c for c in calls)

    def test_distill_with_config(self, tmp_path):
        cfg = str(tmp_path / "config.yaml")
        import yaml
        with open(cfg, "w") as f:
            yaml.dump({"distillation": {"lambda_bc": 0.5, "lambda_rl": 0.3}}, f)
        with patch("builtins.print") as mock_print:
            _run_distill(config_path=cfg, num_samples=5, log_dir=str(tmp_path))
            calls = [str(c) for c in mock_print.call_args_list]
            assert any("Distillation complete" in c for c in calls)

    def test_distill_config_not_found(self, tmp_path):
        with patch("builtins.print") as mock_print:
            _run_distill(config_path="/nonexistent.yaml", num_samples=5, log_dir=str(tmp_path))
            calls = [str(c) for c in mock_print.call_args_list]
            assert any("Config not found" in c for c in calls)

    def test_distill_with_ab_test(self, tmp_path):
        ab_cfg = str(tmp_path / "ab_test.yaml")
        import yaml
        ab_config = {
            "lineage": {
                "name": "cli_test",
                "auto_baseline": True,
                "rounds": [
                    {
                        "name": "round_1",
                        "variants": [
                            {"name": "A", "lambda_bc": 1.0},
                            {"name": "B", "lambda_bc": 0.5},
                        ],
                    },
                ],
            },
        }
        with open(ab_cfg, "w") as f:
            yaml.dump(ab_config, f)
        with patch("builtins.print") as mock_print:
            _run_distill(num_samples=10, ab_test_path=ab_cfg, log_dir=str(tmp_path))
            calls = [str(c) for c in mock_print.call_args_list]
            assert any("Lineage:" in c or "round_1" in c for c in calls)

    def test_distill_ab_config_not_found(self, tmp_path):
        with patch("builtins.print") as mock_print:
            _run_distill(num_samples=5, ab_test_path="/nonexistent.yaml", log_dir=str(tmp_path))
            calls = [str(c) for c in mock_print.call_args_list]
            assert any("AB test config not found" in c for c in calls)

    def test_distill_no_rounds_in_ab(self, tmp_path):
        ab_cfg = str(tmp_path / "empty_ab.yaml")
        import yaml
        with open(ab_cfg, "w") as f:
            yaml.dump({"lineage": {"name": "empty"}}, f)
        with patch("builtins.print") as mock_print:
            _run_distill(num_samples=5, ab_test_path=ab_cfg, log_dir=str(tmp_path))
            calls = [str(c) for c in mock_print.call_args_list]
            assert any("No rounds defined" in c for c in calls)

    def test_main_help(self):
        from zilli.cli import main
        with patch("argparse.ArgumentParser.print_help") as mock_help:
            with patch("sys.argv", ["zilli"]):
                main()
            mock_help.assert_called_once()
