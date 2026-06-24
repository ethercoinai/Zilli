from zilli.training.data import (
    make_dummy_distillation_samples,
    make_dummy_failure,
    make_dummy_golden,
)


class TestMakeDummyGolden:
    def test_default_count(self):
        trajs = make_dummy_golden()
        assert len(trajs) == 10

    def test_custom_count(self):
        trajs = make_dummy_golden(count=5)
        assert len(trajs) == 5

    def test_each_has_trajectory(self):
        trajs = make_dummy_golden(count=3)
        for t in trajs:
            assert "trajectory" in t
            assert "reward" in t
            assert len(t["trajectory"]) == 2


class TestMakeDummyFailure:
    def test_default_count(self):
        trajs = make_dummy_failure()
        assert len(trajs) == 5

    def test_each_has_single_step(self):
        trajs = make_dummy_failure(count=3)
        for t in trajs:
            assert len(t["trajectory"]) == 1
            assert t["trajectory"][0]["observation"]["success"] is False


class TestMakeDummyDistillation:
    def test_default_count(self):
        samples = make_dummy_distillation_samples()
        assert len(samples) == 100

    def test_correct_type(self):
        from zilli.training.distillation import DistillationSample
        samples = make_dummy_distillation_samples(count=3)
        for s in samples:
            assert isinstance(s, DistillationSample)
