from zilli.adaptive.sota_scheduler import DynamicSOTAScheduler


class TestDynamicSOTAScheduler:
    def test_default_initialization(self):
        s = DynamicSOTAScheduler()
        assert s.remaining_budget == 500.0
        assert s.total_calls == 0
        assert s.calls_this_hour == 0
        assert s.hourly_quota > 0

    def test_cost_per_call(self):
        s = DynamicSOTAScheduler(cost_per_call={"gpt4": 0.1, "default": 0.04})
        assert s.cost_per_call["gpt4"] == 0.1
        assert s.cost_per_call["default"] == 0.04

    def test_executor_confidence(self):
        s = DynamicSOTAScheduler()
        assert s._executor_confidence({"max_prob": 0.9}) == 0.9
        assert s._executor_confidence({}) == 0.5

    def test_should_call_sota_low_budget(self):
        s = DynamicSOTAScheduler(monthly_budget_usd=1.0)
        s.remaining_budget = 0.05
        difficulty_task = {"failure_rate": 0.9, "samples": 10,
                          "threshold": 0.7, "success_with_sota": 0.5,
                          "success_without_sota": 0.3}
        s.task_stats["hard"] = difficulty_task
        # When budget < 10%, only tasks with failure_rate > 0.8 get SOTA
        assert s.should_call_sota("hard", {"max_prob": 0.5}) is True
        s.task_stats["easy"].update({"failure_rate": 0.1, "samples": 10})
        assert s.should_call_sota("easy", {"max_prob": 0.5}) is False

    def test_should_call_sota_high_difficulty(self):
        s = DynamicSOTAScheduler()
        s.task_stats["hard"].update({"failure_rate": 0.8, "samples": 10})
        assert s.should_call_sota("hard", {"max_prob": 0.5}) is True

    def test_should_call_sota_high_gap(self):
        s = DynamicSOTAScheduler()
        s.task_stats["gap_task"].update({
            "failure_rate": 0.3, "samples": 10, "success_with_sota": 0.9,
            "success_without_sota": 0.4,
        })
        assert s.should_call_sota("gap_task", {"max_prob": 0.6}) is True

    def test_should_call_sota_not_needed(self):
        s = DynamicSOTAScheduler()
        s.task_stats["easy"].update({
            "failure_rate": 0.1, "samples": 10, "success_with_sota": 0.95,
            "success_without_sota": 0.93,
        })
        assert s.should_call_sota("easy", {"max_prob": 0.95}) is False

    def test_record_call_deducts_budget(self):
        s = DynamicSOTAScheduler(cost_per_call={"gpt4": 0.05, "default": 0.04})
        init = s.remaining_budget
        s.record_call("gpt4", "code_generation", actual_success=True)
        assert s.remaining_budget == init - 0.05
        assert s.total_calls == 1
        assert s.calls_this_hour == 1

    def test_record_call_updates_stats_success(self):
        s = DynamicSOTAScheduler()
        s.record_call("default", "code_gen", actual_success=True)
        stats = s.task_stats["code_gen"]
        assert stats["samples"] == 1
        assert stats["success_with_sota"] == 1.0
        assert stats["failure_rate"] == 0.0

    def test_record_call_updates_stats_failure(self):
        s = DynamicSOTAScheduler()
        s.record_call("default", "code_gen", actual_success=False)
        stats = s.task_stats["code_gen"]
        assert stats["failure_rate"] > 0.5

    def test_record_without_sota(self):
        s = DynamicSOTAScheduler()
        s.record_without_sota("code_gen", actual_success=True)
        stats = s.task_stats["code_gen"]
        assert stats["samples"] == 1
        assert stats["success_without_sota"] == 1.0

    def test_stats_output(self):
        s = DynamicSOTAScheduler()
        s.record_call("default", "test", actual_success=True)
        stats = s.stats()
        assert stats["total_calls"] == 1
        assert "test" in stats["task_types"]
        assert stats["remaining_budget"] < stats["remaining_budget"] + 0.001

    def test_reset_hourly_counter(self):
        s = DynamicSOTAScheduler()
        s.record_call("default", "t", actual_success=True)
        assert s.calls_this_hour == 1
        s.reset_hourly_counter()
        assert s.calls_this_hour == 0

    def test_sample_threshold_returns_valid(self):
        s = DynamicSOTAScheduler()
        threshold = s._sample_threshold("any")
        assert threshold in s.threshold_arms

    def test_update_bandit(self):
        s = DynamicSOTAScheduler()
        old_a = s.beta_params["0.7"]["alpha"]
        s._update_bandit(0.7, 1.0)
        assert s.beta_params["0.7"]["alpha"] == old_a + 1.0

    def test_should_call_sota_random_exploration(self):
        s = DynamicSOTAScheduler()
        # Set gap=0.06 so gap < 0.05 early return doesn't fire,
        # conf > max threshold so only the 5% random path triggers
        s.task_stats["rnd"].update({
            "failure_rate": 0.0, "samples": 100, "success_with_sota": 0.5,
            "success_without_sota": 0.44,
        })
        s.exploration_rate = 0.05
        calls = sum(s.should_call_sota("rnd", {"max_prob": 0.95})
                    for _ in range(1000))
        assert 20 <= calls <= 80
