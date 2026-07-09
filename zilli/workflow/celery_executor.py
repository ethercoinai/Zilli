from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from zilli.dag.engine import DAGExecutor, DAGNode, NodeStatus, TaskDAG

logger = logging.getLogger("zilli.workflow.celery_executor")


@dataclass
class DAGRunRecord:
    run_id: str
    dag_json: str
    status: str = "pending"
    results: list[dict] = field(default_factory=list)
    created_at: float = 0.0
    completed_at: float = 0.0


class CeleryDAGExecutor:
    """Wraps DAGExecutor with Celery-backed persistent task execution.

    Each DAG node is submitted as a Celery task so execution survives
    worker restarts and provides distributed parallelism.
    Falls back to in-process DAGExecutor when Celery is unavailable.
    """

    def __init__(self, max_concurrency: int = 4):
        self.max_concurrency = max_concurrency
        self._celery_available = False
        self._celery_app = None
        self._records: dict[str, DAGRunRecord] = {}
        self._inproc = DAGExecutor(max_concurrency=max_concurrency)

        try:
            from zilli.workflow.celery_app import celery_app as _ca
            self._celery_app = _ca
            self._celery_available = True
        except Exception:
            logger.info("Celery not available — using in-process DAGExecutor")

    async def execute(
        self,
        dag: TaskDAG,
        task_fn,
        on_complete=None,
        run_id: str | None = None,
    ) -> list[Any]:
        if not self._celery_available:
            logger.info("Using in-process DAGExecutor (Celery not available)")
            return await self._inproc.execute(dag, task_fn, on_complete)

        rid = run_id or f"dag_{int(time.time())}"
        dag_json = json.dumps(dag.to_dict())

        record = DAGRunRecord(
            run_id=rid,
            dag_json=dag_json,
            created_at=time.time(),
        )
        self._records[rid] = record
        record.status = "running"

        results = []
        for node in self._iterate_ready(dag):
            result = await self._submit_celery_task(node, task_fn)
            results.append(result)

        record.results = [r.__dict__ if hasattr(r, '__dict__') else r for r in results]
        record.status = "completed"
        record.completed_at = time.time()
        return results

    async def _submit_celery_task(self, node: DAGNode, task_fn) -> Any:
        from celery import current_app
        task = current_app.send_task(
            "zilli.workflow.tasks.execute_dag_node",
            args=[node.task_id, node.description],
            kwargs={"task_type": node.task_type.value},
        )
        return await asyncio.to_thread(task.get, timeout=600)

    def _iterate_ready(self, dag: TaskDAG):
        from collections import deque
        ready = dag.get_ready_nodes()
        visited = {n.task_id for n in ready}
        queue = deque(ready)
        while queue:
            node = queue.popleft()
            yield node
            for child_id in dag._adj.get(node.task_id, []):
                if child_id not in visited:
                    child = dag.nodes.get(child_id)
                    if child and all(
                        dag.nodes[p].status == NodeStatus.COMPLETED
                        for p in dag._reverse_adj.get(child_id, [])
                    ):
                        visited.add(child_id)
                        queue.append(child)

    def get_run(self, run_id: str) -> DAGRunRecord | None:
        return self._records.get(run_id)

    def list_runs(self, limit: int = 10) -> list[DAGRunRecord]:
        return sorted(self._records.values(), key=lambda r: r.created_at, reverse=True)[:limit]
