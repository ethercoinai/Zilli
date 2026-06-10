from zilli.distillation.dsl import (
    ExperimentLineage,
    ExperimentParams,
    RoundDef,
    lineage_report,
    run_multi_round,
)
from zilli.training.distillation import DistillationSample


def _sample(exec_reward=0.5, plan_reward=0.8):
    return DistillationSample(
        executor_action={"tool": "w"},
        planner_action={"tool": "w"},
        executor_log_prob=-1.0,
        planner_log_prob=-1.5,
        executor_reward=exec_reward,
        planner_reward=plan_reward,
        executor_embedding=[0.1, 0.2],
        planner_embedding=[0.3, 0.4],
    )


class TestRoundDef:
    def test_create_round(self):
        rd = RoundDef(name="round_1")
        assert rd.name == "round_1"
        assert len(rd.variants) == 0

    def test_add_variant(self):
        rd = RoundDef(name="test")
        p = ExperimentParams(name="A")
        rd.add(p)
        assert len(rd.variants) == 1
        assert rd.variants[0].name == "A"

    def test_chainable_add(self):
        rd = RoundDef(name="chain")
        rd.add(ExperimentParams(name="A")).add(ExperimentParams(name="B"))
        assert len(rd.variants) == 2


class TestExperimentLineage:
    def test_create_lineage(self):
        lineage = ExperimentLineage(name="test")
        assert lineage.name == "test"
        assert len(lineage.rounds) == 0

    def test_add_round(self):
        lineage = ExperimentLineage(name="test")
        lineage.add_round("r1", [ExperimentParams(name="A")])
        lineage.add_round("r2", [ExperimentParams(name="B")])
        assert len(lineage.rounds) == 2
        assert lineage.rounds[0].name == "r1"

    def test_chainable_add_round(self):
        lineage = ExperimentLineage(name="chain")
        (lineage
         .add_round("r1", [ExperimentParams(name="A")])
         .add_round("r2", [ExperimentParams(name="B")]))
        assert len(lineage.rounds) == 2

    def test_auto_baseline_default(self):
        lineage = ExperimentLineage(name="test")
        assert lineage.auto_baseline is True

    def test_disable_auto_baseline(self):
        lineage = ExperimentLineage(name="test", auto_baseline=False)
        assert lineage.auto_baseline is False

    def test_summary_before_run(self):
        lineage = ExperimentLineage(name="test")
        lineage.add_round("r1", [ExperimentParams(name="A")])
        s = lineage.summary()
        assert s["lineage"] == "test"
        assert s["rounds"] == 1
        assert len(s["history"]) == 0

    def test_summary_after_run(self, tmp_path):
        lineage = ExperimentLineage(name="test")
        lineage.add_round("r1", [ExperimentParams(name="A", lambda_bc=1.0),
                                 ExperimentParams(name="B", lambda_bc=0.5)])
        samples = [_sample() for _ in range(20)]
        run_multi_round(lineage, samples, log_dir=str(tmp_path))
        s = lineage.summary()
        assert len(s["history"]) == 1
        assert s["best_overall"] is not None


class TestRunMultiRound:
    def test_single_round(self, tmp_path):
        lineage = ExperimentLineage(name="single")
        lineage.add_round("r1", [
            ExperimentParams(name="A", lambda_bc=1.0),
            ExperimentParams(name="B", lambda_bc=0.5),
        ])
        samples = [_sample() for _ in range(20)]
        result = run_multi_round(lineage, samples, log_dir=str(tmp_path))
        assert len(result.results) == 1
        assert result.results[0].round_name == "r1"
        assert len(result.results[0].iteration.results) == 2
        assert result.best_params is not None

    def test_multi_round_auto_baseline(self, tmp_path):
        lineage = ExperimentLineage(name="multi")
        lineage.add_round("r1", [
            ExperimentParams(name="A", lambda_bc=1.0),
            ExperimentParams(name="B", lambda_bc=0.5),
        ])
        lineage.add_round("r2", [
            ExperimentParams(name="C", lambda_rl=0.8),
        ])
        samples = [_sample() for _ in range(20)]
        result = run_multi_round(lineage, samples, log_dir=str(tmp_path))
        assert len(result.results) == 2
        r2_variants = result.results[1].iteration.results
        assert len(r2_variants) >= 2

    def test_multi_round_no_auto_baseline(self, tmp_path):
        lineage = ExperimentLineage(name="multi_noauto", auto_baseline=False)
        lineage.add_round("r1", [ExperimentParams(name="A")])
        lineage.add_round("r2", [ExperimentParams(name="B")])
        samples = [_sample() for _ in range(20)]
        result = run_multi_round(lineage, samples, log_dir=str(tmp_path))
        assert len(result.results) == 2
        assert len(result.results[1].iteration.results) == 1

    def test_three_rounds(self, tmp_path):
        lineage = ExperimentLineage(name="three")
        lineage.add_round("r1", [ExperimentParams(name="A")])
        lineage.add_round("r2", [ExperimentParams(name="B")])
        lineage.add_round("r3", [ExperimentParams(name="C")])
        samples = [_sample() for _ in range(20)]
        result = run_multi_round(lineage, samples, log_dir=str(tmp_path))
        assert len(result.results) == 3

    def test_best_params_from_multi_round(self, tmp_path):
        lineage = ExperimentLineage(name="besttest")
        lineage.add_round("r1", [
            ExperimentParams(name="A", lambda_bc=1.0),
            ExperimentParams(name="B", lambda_bc=0.5),
        ])
        lineage.add_round("r2", [
            ExperimentParams(name="C", lambda_rl=0.2),
            ExperimentParams(name="D", lambda_rl=0.8),
        ])
        samples = [_sample() for _ in range(20)]
        result = run_multi_round(lineage, samples, log_dir=str(tmp_path))
        assert result.best_params is not None
        assert result.best_params.name in ("A", "B", "C", "D")

    def test_run_with_empty_samples(self, tmp_path):
        lineage = ExperimentLineage(name="empty")
        lineage.add_round("r1", [ExperimentParams(name="A")])
        result = run_multi_round(lineage, [], log_dir=str(tmp_path))
        assert len(result.results) == 1
        assert result.results[0].iteration.results[0].total_samples == 0


class TestLineageReport:
    def test_report_generated(self, tmp_path):
        lineage = ExperimentLineage(name="report")
        lineage.add_round("r1", [ExperimentParams(name="A")])
        samples = [_sample() for _ in range(20)]
        run_multi_round(lineage, samples, log_dir=str(tmp_path))
        report = lineage_report(lineage)
        assert "Lineage: report" in report
        assert "Round [0]" in report
        assert "r1" in report
