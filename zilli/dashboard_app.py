"""Zilli Streamlit management dashboard.

Usage:
    streamlit run zilli/dashboard_app.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Zilli Dashboard", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent
AUDIT_DIR = BASE_DIR / "audit_logs"
COST_FILE = Path.home() / ".zilli_budget.json"
STATE_FILE = BASE_DIR / "state" / "STATE.md"


def load_audit_logs(limit: int = 100) -> list[dict]:
    logs: list[dict] = []
    if not AUDIT_DIR.exists():
        return logs
    for f in sorted(AUDIT_DIR.glob("*.jsonl"), reverse=True)[:5]:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
                if len(logs) >= limit:
                    return logs
    return logs


def load_cost_stats() -> dict:
    if not COST_FILE.exists():
        return {"remaining_budget": 500.0, "total_calls": 0}
    try:
        return json.loads(COST_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def load_state() -> str:
    if STATE_FILE.exists():
        return STATE_FILE.read_text()
    return "No state file found"


st.title("Zilli Management Dashboard")

col1, col2, col3, col4 = st.columns(4)
cost = load_cost_stats()
col1.metric("Budget Remaining", f"${cost.get('remaining_budget', 0):.2f}")
col2.metric("Total API Calls", cost.get("total_calls", 0))
col3.metric("Audit Logs Found", len(load_audit_logs()))
col4.metric("State File", "OK" if STATE_FILE.exists() else "Missing")

tab1, tab2, tab3, tab4 = st.tabs(["Audit Logs", "Cost Control", "System State", "DAG Runs"])

with tab1:
    st.subheader("Recent Audit Events")
    logs = load_audit_logs()
    if logs:
        st.dataframe(
            [{k: str(v)[:80] for k, v in entry.items()} for entry in logs],
            use_container_width=True,
        )
        log_text = "\n".join(json.dumps(e, ensure_ascii=False) for e in logs)
        st.download_button(
            "Export as JSONL",
            data=log_text,
            file_name=f"audit_export_{int(time.time())}.jsonl",
            mime="application/x-ndjson",
        )
    else:
        st.info("No audit logs found")

with tab2:
    st.subheader("Cost & Budget")
    c = load_cost_stats()
    remaining = c.get("remaining_budget", 500.0)
    total = c.get("total_calls", 0)
    st.progress(max(0.0, min(1.0, remaining / 500.0)),
               text=f"${remaining:.2f} remaining")
    st.write(f"**Total calls**: {total}")
    st.write(f"**Last updated**: {c.get('updated_at', 'N/A')}")

with tab3:
    st.subheader("Current State")
    st.text(load_state())

with tab4:
    st.subheader("DAG Execution Records")
    st.info("Connect to Redis to view live DAG run history")
    if st.button("Clear All Records"):
        st.warning("Not implemented — Redis connection required")

st.caption("Zilli v0.3.0 — Management Dashboard")
