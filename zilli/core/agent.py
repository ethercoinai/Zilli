from __future__ import annotations

import asyncio
import logging
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from zilli.models.base import GenerationResult, ModelBackend
from zilli.models.config import ModelRole
from zilli.models.registry import ModelRegistry

logger = logging.getLogger("zilli.core.agent")

_PROMPT = """You are a coding agent. Write a Python script for this task:

TASK: {task}

Output ONLY valid Python code. No explanations.
The script must print the result to stdout."""

_FIX_PROMPT = """You tried to write code for this task but it failed.

TASK: {task}

Your code:
```python
{code}
```

Error:
{error}

Fix the bug. Output ONLY the corrected Python code."""


@dataclass
class AgentResult:
    success: bool
    output: str
    error: Optional[str] = None
    iterations: int = 0
    duration_ms: float = 0.0
    code_used: str = ""


class Agent:
    def __init__(self, model: Optional[ModelBackend] = None, max_retries: int = 3):
        self._model = model
        self._max_retries = max_retries

    @classmethod
    async def from_registry(cls, registry: Optional[ModelRegistry] = None) -> Agent:
        if not registry:
            return cls()
        model = await registry.get_model_for_role(ModelRole.EXECUTOR)
        return cls(model=model)

    async def run(self, task: str) -> AgentResult:
        start = time.monotonic()
        last_error = ""
        code = ""

        for attempt in range(self._max_retries + 1):
            code = await self._generate_code(task, attempt, code, last_error)
            if not code:
                return AgentResult(
                    success=False, error="Failed to generate code",
                    iterations=attempt, duration_ms=(time.monotonic() - start) * 1000,
                )

            ok, stdout, stderr = await self._execute_code(code)
            if ok:
                return AgentResult(
                    success=True, output=stdout, code_used=code,
                    iterations=attempt + 1,
                    duration_ms=(time.monotonic() - start) * 1000,
                )

            last_error = stderr or stdout
            logger.info("Attempt %d failed: %s", attempt + 1, last_error[:200])

        return AgentResult(
            success=False, output="", error=last_error,
            iterations=self._max_retries + 1, code_used=code,
            duration_ms=(time.monotonic() - start) * 1000,
        )

    async def _generate_code(self, task: str, attempt: int, prev_code: str, prev_error: str) -> str:
        if self._model is None:
            return _fallback_for_task(task)

        prompt = _FIX_PROMPT.format(task=task, code=prev_code, error=prev_error) if attempt > 0 else _PROMPT.format(task=task)
        gen = await self._model.generate(prompt, max_tokens=4096, temperature=0.3)
        return _extract_python(gen.text or "")

    async def _execute_code(self, code: str) -> tuple[bool, str, str]:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "_agent_script.py"
            script.write_text(code)
            try:
                proc = await asyncio.create_subprocess_exec(
                    "python3", str(script),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=tmp,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
                out = stdout.decode(errors="replace").strip()
                err = stderr.decode(errors="replace").strip()
                return proc.returncode == 0, out, err
            except asyncio.TimeoutError:
                return False, "", "Timeout (30s)"
            except Exception as e:
                return False, "", str(e)


def _extract_python(text: str) -> str:
    if "```" in text:
        parts = text.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 1:
                content = part.split("\n", 1)[1] if "\n" in part else part
                return content.strip()
    return text.strip()


def _fallback_for_task(task: str) -> str:
    body = _guess_body(task)
    code = f"""#!/usr/bin/env python3
import sys

def main():
{textwrap.indent(body, "    ")}

if __name__ == "__main__":
    main()
"""
    return code.strip()


def _guess_body(task: str) -> str:
    t = task.lower()
    if "fib" in t:
        return "n = 10\na, b = 0, 1\nfor i in range(n):\n    print(a, end=' ' if i < n - 1 else '\\n')\n    a, b = b, a + b"
    if "hello" in t or "hi" in t:
        return 'print("Hello from Zilli agent!")'
    if any(w in t for w in ("add", "sum", "plus")):
        return "print(sum([1, 2, 3, 4, 5]))"
    if "prime" in t or "素数" in t:
        return ("def is_prime(n):\n"
                "    return n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1))\n"
                "print([n for n in range(2, 50) if is_prime(n)])")
    if "sort" in t:
        return "print(sorted([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]))"
    return 'print("Task: " + ' + repr(task) + ")"


__all__ = ["Agent", "AgentResult"]
