from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger("zilli.dag.engine")


class NodeStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskType(str, Enum):
    CODEGEN = "codegen"
    TRAIN = "train"
    EVALUATE = "evaluate"
    DEPLOY = "deploy"
    TEST = "test"
    CUSTOM = "custom"


@dataclass
class DAGNode:
    task_id: str
    description: str
    task_type: TaskType = TaskType.CUSTOM
    domain_tags: list[str] = field(default_factory=list)
    input_spec: dict[str, Any] = field(default_factory=dict)
    output_spec: dict[str, Any] = field(default_factory=dict)
    estimated_cost: float = 0.0
    estimated_latency_ms: float = 0.0
    privacy_level: int = 0
    compliance_tags: list[str] = field(default_factory=list)
    retry_policy: dict[str, Any] = field(default_factory=lambda: {"max_retries": 2, "backoff_s": 1.0})
    status: NodeStatus = NodeStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    complexity: str = "medium"
    hint: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.status in (NodeStatus.COMPLETED, NodeStatus.FAILED, NodeStatus.SKIPPED)


@dataclass
class DAGEdge:
    source: str
    target: str
    data_dependency: bool = True

    def __post_init__(self):
        if self.source == self.target:
            raise ValueError(f"Self-loop detected on node {self.source}")


@dataclass
class DAGValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    topo_order: list[str] = field(default_factory=list)
    critical_path: list[str] = field(default_factory=list)
    parallel_groups: list[list[str]] = field(default_factory=list)


class TaskDAG:
    def __init__(self):
        self.nodes: dict[str, DAGNode] = {}
        self.edges: list[DAGEdge] = []
        self._adj: dict[str, list[str]] = defaultdict(list)
        self._reverse_adj: dict[str, list[str]] = defaultdict(list)

    def add_node(self, node: DAGNode) -> None:
        if node.task_id in self.nodes:
            raise ValueError(f"Duplicate node id: {node.task_id}")
        self.nodes[node.task_id] = node

    def add_edge(self, edge: DAGEdge) -> None:
        if edge.source not in self.nodes:
            raise ValueError(f"Source node not found: {edge.source}")
        if edge.target not in self.nodes:
            raise ValueError(f"Target node not found: {edge.target}")
        self.edges.append(edge)
        self._adj[edge.source].append(edge.target)
        self._reverse_adj[edge.target].append(edge.source)

    def remove_node(self, task_id: str) -> None:
        self.nodes.pop(task_id, None)
        self.edges = [e for e in self.edges if e.source != task_id and e.target != task_id]
        self._adj.pop(task_id, None)
        self._reverse_adj.pop(task_id, None)
        for adj_list in self._adj.values():
            while task_id in adj_list:
                adj_list.remove(task_id)
        for adj_list in self._reverse_adj.values():
            while task_id in adj_list:
                adj_list.remove(task_id)

    def validate(self) -> DAGValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        in_degree = defaultdict(int)
        for edge in self.edges:
            in_degree[edge.target] += 1

        queue = deque()
        for nid in self.nodes:
            if in_degree[nid] == 0:
                queue.append(nid)

        topo_order: list[str] = []
        while queue:
            nid = queue.popleft()
            topo_order.append(nid)
            for child in self._adj[nid]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(topo_order) != len(self.nodes):
            missing = set(self.nodes) - set(topo_order)
            errors.append(f"Cycle detected involving nodes: {missing}")

        critical_path = self._compute_critical_path()
        parallel_groups = self._compute_parallel_groups()

        return DAGValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            topo_order=topo_order,
            critical_path=critical_path,
            parallel_groups=parallel_groups,
        )

    def _topo_sort(self) -> list[str]:
        in_degree = defaultdict(int)
        for edge in self.edges:
            in_degree[edge.target] += 1
        queue = deque()
        for nid in self.nodes:
            if in_degree[nid] == 0:
                queue.append(nid)
        order = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for child in self._adj[nid]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        return order

    def _compute_critical_path(self) -> list[str]:
        dist: dict[str, float] = {nid: 0.0 for nid in self.nodes}
        prev: dict[str, Optional[str]] = {nid: None for nid in self.nodes}
        order = self._topo_sort()

        for nid in order:
            node = self.nodes[nid]
            for child in self._adj[nid]:
                new_dist = dist[nid] + node.estimated_latency_ms
                if new_dist > dist[child]:
                    dist[child] = new_dist
                    prev[child] = nid

        if not dist:
            return []
        end_node = max(dist, key=dist.get)
        path = []
        current: Optional[str] = end_node
        while current is not None:
            path.append(current)
            current = prev[current]
        path.reverse()
        return path

    def _compute_parallel_groups(self) -> list[list[str]]:
        in_degree = defaultdict(int)
        for edge in self.edges:
            in_degree[edge.target] += 1

        groups: list[list[str]] = []
        remaining = set(self.nodes.keys())

        while remaining:
            ready = [nid for nid in remaining if in_degree[nid] == 0]
            if not ready:
                break
            groups.append(sorted(ready))
            for nid in ready:
                remaining.discard(nid)
                for child in self._adj[nid]:
                    in_degree[child] -= 1

        return groups

    def get_ready_nodes(self) -> list[DAGNode]:
        ready = []
        for nid, node in self.nodes.items():
            if node.status != NodeStatus.PENDING:
                continue
            any_parent_failed = any(
                self.nodes[p].status == NodeStatus.FAILED
                for p in self._reverse_adj[nid]
            )
            if any_parent_failed:
                node.status = NodeStatus.SKIPPED
                continue
            parents_done = all(
                self.nodes[p].status == NodeStatus.COMPLETED
                for p in self._reverse_adj[nid]
            )
            if parents_done:
                ready.append(node)
        return ready

    def to_dict(self) -> dict:
        return {
            "nodes": [
                {
                    "id": n.task_id,
                    "desc": n.description,
                    "type": n.task_type.value,
                    "domain": n.domain_tags,
                    "complexity": n.complexity,
                    "cost": n.estimated_cost,
                    "latency_ms": n.estimated_latency_ms,
                    "privacy_level": n.privacy_level,
                    "depends_on": self._reverse_adj.get(n.task_id, []),
                }
                for n in self.nodes.values()
            ],
            "edges": [{"source": e.source, "target": e.target} for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict) -> TaskDAG:
        dag = cls()
        for nd in data.get("nodes", []):
            dag.add_node(DAGNode(
                task_id=nd["id"],
                description=nd.get("desc", ""),
                task_type=TaskType(nd.get("type", "custom")),
                domain_tags=nd.get("domain", []),
                complexity=nd.get("complexity", "medium"),
                estimated_cost=nd.get("cost", 0.0),
                estimated_latency_ms=nd.get("latency_ms", 0.0),
                privacy_level=nd.get("privacy_level", 0),
            ))
        for ed in data.get("edges", []):
            dag.add_edge(DAGEdge(source=ed["source"], target=ed["target"]))
        for nd in data.get("nodes", []):
            for dep_id in nd.get("depends_on", []):
                if not any(e.source == dep_id and e.target == nd["id"] for e in dag.edges):
                    dag.add_edge(DAGEdge(source=dep_id, target=nd["id"]))
        return dag


TaskFunc = Callable[[DAGNode], Coroutine[Any, Any, Any]]


@dataclass
class ExecutionResult:
    node_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0


class DAGExecutor:
    def __init__(self, max_concurrency: int = 4):
        self.max_concurrency = max_concurrency
        self.results: list[ExecutionResult] = []

    async def execute(
        self,
        dag: TaskDAG,
        task_fn: TaskFunc,
        on_complete: Optional[Callable[[ExecutionResult], Coroutine[Any, Any, None]]] = None,
    ) -> list[ExecutionResult]:
        sem = asyncio.Semaphore(self.max_concurrency)
        self.results = []

        async def _run_node(node: DAGNode) -> ExecutionResult:
            async with sem:
                node.status = NodeStatus.RUNNING
                import time
                start = time.monotonic()
                retries = node.retry_policy.get("max_retries", 0)
                backoff = node.retry_policy.get("backoff_s", 1.0)
                last_error = None

                for attempt in range(retries + 1):
                    try:
                        result = await task_fn(node)
                        elapsed = (time.monotonic() - start) * 1000
                        node.status = NodeStatus.COMPLETED
                        node.result = result
                        er = ExecutionResult(node_id=node.task_id, success=True, result=result, duration_ms=elapsed)
                        self.results.append(er)
                        if on_complete:
                            await on_complete(er)
                        return er
                    except Exception as e:
                        last_error = str(e)
                        if attempt < retries:
                            await asyncio.sleep(backoff * (2 ** attempt))

                elapsed = (time.monotonic() - start) * 1000
                node.status = NodeStatus.FAILED
                node.error = last_error
                er = ExecutionResult(node_id=node.task_id, success=False, error=last_error, duration_ms=elapsed)
                self.results.append(er)
                if on_complete:
                    await on_complete(er)
                return er

        completed: set[str] = set()
        in_progress: dict[str, asyncio.Task] = {}

        while len(completed) < len(dag.nodes):
            ready = dag.get_ready_nodes()
            for node in ready:
                if node.task_id not in completed and node.task_id not in in_progress:
                    in_progress[node.task_id] = asyncio.create_task(_run_node(node))

            if in_progress:
                done, _ = await asyncio.wait(
                    in_progress.values(), return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    for nid, t in list(in_progress.items()):
                        if t is task:
                            completed.add(nid)
                            del in_progress[nid]
                            if dag.nodes[nid].status == NodeStatus.FAILED:
                                self._cascade_skip(dag, nid, completed)
                            break
            elif len(completed) < len(dag.nodes):
                remaining = [nid for nid in dag.nodes if nid not in completed]
                ready = dag.get_ready_nodes()
                if not ready and not in_progress:
                    stalled = [dag.nodes[nid].status for nid in remaining]
                    logger.warning(f"DAG stalled: remaining={remaining}, statuses={stalled}")
                break

        return self.results

    def _cascade_skip(self, dag: TaskDAG, failed_id: str, completed: set[str]) -> None:
        from collections import deque
        queue = deque(dag._adj.get(failed_id, []))
        visited = {failed_id}
        while queue:
            child = queue.popleft()
            if child in visited:
                continue
            visited.add(child)
            if child not in completed:
                dag.nodes[child].status = NodeStatus.SKIPPED
                completed.add(child)
                for grandchild in dag._adj.get(child, []):
                    if grandchild not in visited:
                        queue.append(grandchild)


__all__ = [
    "NodeStatus", "TaskType", "DAGNode", "DAGEdge",
    "DAGValidationResult", "TaskDAG", "DAGExecutor", "ExecutionResult",
]
