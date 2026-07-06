import asyncio
import pytest
from zilli.dag.engine import (
    DAGNode, DAGEdge, TaskDAG, DAGExecutor, DAGValidationResult,
    NodeStatus, TaskType, ExecutionResult,
)


class TestDAGNode:
    def test_create_node(self):
        node = DAGNode(task_id="A", description="Test task")
        assert node.task_id == "A"
        assert node.status == NodeStatus.PENDING
        assert not node.is_terminal

    def test_terminal_status(self):
        for status in [NodeStatus.COMPLETED, NodeStatus.FAILED, NodeStatus.SKIPPED]:
            node = DAGNode(task_id="A", description="Test", status=status)
            assert node.is_terminal

    def test_non_terminal_status(self):
        for status in [NodeStatus.PENDING, NodeStatus.READY, NodeStatus.RUNNING]:
            node = DAGNode(task_id="A", description="Test", status=status)
            assert not node.is_terminal


class TestDAGEdge:
    def test_valid_edge(self):
        edge = DAGEdge(source="A", target="B")
        assert edge.source == "A"
        assert edge.target == "B"

    def test_self_loop_raises(self):
        with pytest.raises(ValueError, match="Self-loop"):
            DAGEdge(source="A", target="A")


class TestTaskDAG:
    def test_add_node(self):
        dag = TaskDAG()
        dag.add_node(DAGNode(task_id="A", description="Task A"))
        assert "A" in dag.nodes

    def test_duplicate_node_raises(self):
        dag = TaskDAG()
        dag.add_node(DAGNode(task_id="A", description="Task A"))
        with pytest.raises(ValueError, match="Duplicate"):
            dag.add_node(DAGNode(task_id="A", description="Task A again"))

    def test_add_edge(self):
        dag = TaskDAG()
        dag.add_node(DAGNode(task_id="A", description="A"))
        dag.add_node(DAGNode(task_id="B", description="B"))
        dag.add_edge(DAGEdge(source="A", target="B"))
        assert len(dag.edges) == 1

    def test_edge_missing_node_raises(self):
        dag = TaskDAG()
        dag.add_node(DAGNode(task_id="A", description="A"))
        with pytest.raises(ValueError, match="not found"):
            dag.add_edge(DAGEdge(source="A", target="B"))

    def test_remove_node(self):
        dag = TaskDAG()
        dag.add_node(DAGNode(task_id="A", description="A"))
        dag.add_node(DAGNode(task_id="B", description="B"))
        dag.add_edge(DAGEdge(source="A", target="B"))
        dag.remove_node("A")
        assert "A" not in dag.nodes
        assert len(dag.edges) == 0

    def test_validate_no_cycle(self):
        dag = TaskDAG()
        dag.add_node(DAGNode(task_id="A", description="A"))
        dag.add_node(DAGNode(task_id="B", description="B"))
        dag.add_node(DAGNode(task_id="C", description="C"))
        dag.add_edge(DAGEdge(source="A", target="B"))
        dag.add_edge(DAGEdge(source="B", target="C"))
        result = dag.validate()
        assert result.valid
        assert result.topo_order == ["A", "B", "C"]

    def test_validate_cycle_detected(self):
        dag = TaskDAG()
        dag.add_node(DAGNode(task_id="A", description="A"))
        dag.add_node(DAGNode(task_id="B", description="B"))
        dag.add_edge(DAGEdge(source="A", target="B"))
        dag.add_edge(DAGEdge(source="B", target="A"))
        result = dag.validate()
        assert not result.valid
        assert len(result.errors) > 0

    def test_critical_path(self):
        dag = TaskDAG()
        dag.add_node(DAGNode(task_id="A", description="A", estimated_latency_ms=100))
        dag.add_node(DAGNode(task_id="B", description="B", estimated_latency_ms=200))
        dag.add_node(DAGNode(task_id="C", description="C", estimated_latency_ms=50))
        dag.add_edge(DAGEdge(source="A", target="B"))
        dag.add_edge(DAGEdge(source="A", target="C"))
        result = dag.validate()
        assert "A" in result.critical_path
        assert "B" in result.critical_path

    def test_parallel_groups(self):
        dag = TaskDAG()
        dag.add_node(DAGNode(task_id="A", description="A"))
        dag.add_node(DAGNode(task_id="B", description="B"))
        dag.add_node(DAGNode(task_id="C", description="C"))
        dag.add_edge(DAGEdge(source="A", target="C"))
        dag.add_edge(DAGEdge(source="B", target="C"))
        result = dag.validate()
        assert len(result.parallel_groups) >= 2
        assert ["A", "B"] in result.parallel_groups or ["B", "A"] in result.parallel_groups

    def test_get_ready_nodes(self):
        dag = TaskDAG()
        dag.add_node(DAGNode(task_id="A", description="A"))
        dag.add_node(DAGNode(task_id="B", description="B"))
        dag.add_edge(DAGEdge(source="A", target="B"))
        ready = dag.get_ready_nodes()
        assert len(ready) == 1
        assert ready[0].task_id == "A"

    def test_to_dict_and_from_dict(self):
        dag = TaskDAG()
        dag.add_node(DAGNode(task_id="A", description="Task A", task_type=TaskType.CODEGEN))
        dag.add_node(DAGNode(task_id="B", description="Task B"))
        dag.add_edge(DAGEdge(source="A", target="B"))
        data = dag.to_dict()
        assert len(data["nodes"]) == 2
        restored = TaskDAG.from_dict(data)
        assert "A" in restored.nodes
        assert "B" in restored.nodes


class TestDAGExecutor:
    def test_execute_simple_dag(self):
        dag = TaskDAG()
        dag.add_node(DAGNode(task_id="A", description="A"))
        dag.add_node(DAGNode(task_id="B", description="B"))
        dag.add_edge(DAGEdge(source="A", target="B"))

        async def task_fn(node):
            return f"done_{node.task_id}"

        executor = DAGExecutor(max_concurrency=2)
        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(executor.execute(dag, task_fn))
        loop.close()
        assert len(results) == 2
        assert all(r.success for r in results)

    def test_execute_with_failure(self):
        dag = TaskDAG()
        dag.add_node(DAGNode(task_id="A", description="A"))

        async def failing_fn(node):
            raise RuntimeError("boom")

        executor = DAGExecutor()
        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(executor.execute(dag, failing_fn))
        loop.close()
        assert len(results) == 1
        assert not results[0].success
        assert "boom" in results[0].error

    def test_execute_with_retry(self):
        dag = TaskDAG()
        dag.add_node(DAGNode(
            task_id="A", description="A",
            retry_policy={"max_retries": 2, "backoff_s": 0.01},
        ))
        call_count = 0

        async def flaky_fn(node):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("fail")
            return "ok"

        executor = DAGExecutor()
        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(executor.execute(dag, flaky_fn))
        loop.close()
        assert results[0].success
        assert call_count == 2


__all__ = ["TestDAGNode", "TestDAGEdge", "TestTaskDAG", "TestDAGExecutor"]
