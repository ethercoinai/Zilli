from __future__ import annotations

import asyncio
import logging
import shlex
import subprocess
import time
from typing import Any, Callable, Optional

from zilli.loops.base import VerificationResult, Verifier
from zilli.models.base import GenerationResult, ModelBackend

logger = logging.getLogger("zilli.loops.verification")


class TestSuiteVerifier(Verifier):
    def __init__(self, command: str, timeout: float = 120.0, cwd: Optional[str] = None):
        self._command = command
        self._timeout = timeout
        self._cwd = cwd

    async def verify(self, input_data: Any, output: Any) -> VerificationResult:
        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *shlex.split(self._command),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self._cwd,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self._timeout,
                )
                elapsed = (time.monotonic() - start) * 1000
                if proc.returncode == 0:
                    return VerificationResult(
                        passed=True,
                        evidence=f"Exit code 0 in {elapsed:.0f}ms",
                        details=stdout.decode(errors="replace")[:2000],
                    )
                return VerificationResult(
                    passed=False,
                    evidence=f"Exit code {proc.returncode}",
                    details=stderr.decode(errors="replace")[:2000],
                )
            except asyncio.TimeoutError:
                proc.kill()
                return VerificationResult(
                    passed=False,
                    evidence=f"Timed out after {self._timeout}s",
                )
        except Exception as e:
            return VerificationResult(
                passed=False,
                evidence=f"Execution error: {e}",
            )


class PredicateVerifier(Verifier):
    def __init__(self, predicate: Callable[[Any, Any], bool], name: str = "predicate"):
        self._pred = predicate
        self._name = name

    async def verify(self, input_data: Any, output: Any) -> VerificationResult:
        try:
            passed = self._pred(input_data, output)
            return VerificationResult(
                passed=passed,
                evidence=f"{self._name}: {'pass' if passed else 'fail'}",
            )
        except Exception as e:
            return VerificationResult(
                passed=False,
                evidence=f"{self._name} raised: {e}",
            )


class ExternalModelVerifier(Verifier):
    def __init__(self, model: ModelBackend, judge_prompt: str = ""):
        self._model = model
        self._judge_prompt = judge_prompt or (
            "You are a strict verifier. Determine if the following output "
            "satisfies the requirements. Reply with PASS or FAIL and a brief reason."
        )

    async def verify(self, input_data: Any, output: Any) -> VerificationResult:
        prompt = f"{self._judge_prompt}\n\nRequirement: {input_data}\n\nOutput: {output}"
        result: GenerationResult = await self._model.generate(prompt, max_tokens=256)
        text = (result.text or "").strip().upper()
        passed = text.startswith("PASS")
        return VerificationResult(
            passed=passed,
            evidence=text[:500],
            details=result.text or "",
            confidence=0.0 if result.error else 0.8,
        )


class CompositeVerifier(Verifier):
    def __init__(self, verifiers: list[Verifier], require_all: bool = True):
        self._verifiers = verifiers
        self._require_all = require_all

    async def verify(self, input_data: Any, output: Any) -> VerificationResult:
        results = await asyncio.gather(*[v.verify(input_data, output) for v in self._verifiers])
        passed = all(r.passed for r in results) if self._require_all else any(r.passed for r in results)
        evidence = "; ".join(r.evidence for r in results)
        details = "\n".join(f"[{'PASS' if r.passed else 'FAIL'}] {r.evidence}" for r in results)
        return VerificationResult(
            passed=passed,
            evidence=evidence,
            details=details,
            confidence=min(r.confidence for r in results) if results else 0.0,
        )
