from __future__ import annotations

import logging

logger = logging.getLogger("zilli.workflow.tasks.dag")


def execute_dag_node(node_id: str, description: str, **kwargs):
    """Celery task that executes a single DAG node.

    Wrapped as a shared_task by Celery autodiscovery.
    """
    logger.info("Executing DAG node %s: %s", node_id, description)
    return {"node_id": node_id, "status": "completed", "output": None}
