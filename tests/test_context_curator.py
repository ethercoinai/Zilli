from __future__ import annotations

from zilli.loops.context_curator import ContextCurator, Trajectory


def test_add_and_get_bullet():
    curator = ContextCurator()
    bid = curator.add_bullet("Always validate input", category="security", confidence=0.9)
    assert len(bid) == 12
    bullets = curator.get(category="security")
    assert len(bullets) == 1
    assert bullets[0].description == "Always validate input"


def test_get_all_bullets():
    curator = ContextCurator()
    curator.add_bullet("tip1", category="general")
    curator.add_bullet("tip2", category="security")
    assert len(curator.get()) == 2


def test_format_context():
    curator = ContextCurator()
    curator.add_bullet("Test bullet", category="general", confidence=0.8)
    formatted = curator.format_context()
    assert "Test bullet" in formatted
    assert "Curated Context Playbook" in formatted


def test_reflect_from_trajectories():
    curator = ContextCurator()
    trajectories = [
        Trajectory(task_id="t1", actions=[], outcome="failure",
                   verifier_evidence="Missing error handling"),
        Trajectory(task_id="t2", actions=[], outcome="failure",
                   verifier_evidence="Missing error handling"),
    ]
    new_ids = curator.reflect(trajectories)
    assert len(new_ids) >= 1
    bullets = curator.get(category="pitfall")
    assert len(bullets) >= 1


def test_reflect_success():
    curator = ContextCurator()
    trajectories = [
        Trajectory(task_id="t1", actions=[], outcome="success",
                   verifier_evidence="All tests passed"),
    ]
    new_ids = curator.reflect(trajectories)
    assert len(new_ids) >= 1
    bullets = curator.get(category="success_pattern")
    assert len(bullets) >= 1


def test_prune():
    curator = ContextCurator()
    for i in range(300):
        curator.add_bullet(f"bullet-{i}", confidence=0.1)
    assert curator.bullet_count <= 200


def test_clear():
    curator = ContextCurator()
    curator.add_bullet("test")
    curator.clear()
    assert curator.bullet_count == 0
