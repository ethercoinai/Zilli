from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from zilli.loops.base import LoopResult
from zilli.loops.runner import LoopRunner
from zilli.loops.verification import TestSuiteVerifier
from zilli.models.base import GenerationResult, ModelBackend
from zilli.swe.context import CodeContext, ExploreResult
from zilli.swe.patch import SWEPatch, read_file_content
from zilli.swe.sandbox import Sandbox, SandboxConfig

logger = logging.getLogger("zilli.swe.agent")


@dataclass
class SWEConfig:
    model_name: str = ""
    max_iterations: int = 3
    test_command: str = "python -m pytest tests/ -x -q"
    explore_timeout: float = 30.0
    test_timeout: float = 120.0
    sandbox_enabled: bool = False
    sandbox_image: str = "python:3.12-slim"
    target_files: list[str] = field(default_factory=list)
    verbose: bool = False


@dataclass
class SWEResult:
    success: bool
    patch: SWEPatch
    context: CodeContext
    loop_result: Optional[LoopResult] = None
    total_duration_ms: float = 0.0
    iterations: int = 0
    error: Optional[str] = None


_DIAGNOSE_PROMPT = """You are a code reviewer analyzing a test failure.

Issue: {issue}

Latest test output:
{test_output}

Relevant source code:
{snippets}

Files explored: {files}

Analyze the root cause. Be specific: which file, which function, what is wrong.
Propose a minimal fix. Output:
ROOT_CAUSE: <one line>
FIX: <description of the change>
AFFECTED_FILES: <comma-separated paths>"""

_GENERATE_FIX_PROMPT = """You are a software engineer fixing a bug. You have analyzed the root cause and need to write the fix.

Issue: {issue}
Root cause: {analysis}

File to fix: {target_file}
Current content:
```python
{content}
```

Write the COMPLETE fixed version of this file. Output ONLY the fixed file content, no explanation."""


class SWEAgent:
    def __init__(
        self,
        config: SWEConfig,
        model_backend: Optional[ModelBackend] = None,
        sandbox: Optional[Sandbox] = None,
    ):
        self._config = config
        self._model = model_backend
        self._sandbox = sandbox or Sandbox(
            SandboxConfig(image=config.sandbox_image) if config.sandbox_enabled else None
        )
        self._context: Optional[CodeContext] = None

    async def run(self, issue: str, repo_path: str) -> SWEResult:
        start = time.monotonic()
        repo = Path(repo_path).resolve()
        if not repo.exists():
            return SWEResult(
                success=False, patch=SWEPatch(),
                context=CodeContext(issue=issue, repo_path=repo),
                error=f"Repository not found: {repo_path}",
            )

        self._context = CodeContext(issue=issue, repo_path=repo)

        if self._config.sandbox_enabled:
            await self._sandbox.ensure_container(repo)

        verifier = TestSuiteVerifier(
            command=self._config.test_command,
            timeout=self._config.test_timeout,
            cwd=str(repo),
        )

        runner = LoopRunner(
            process_fn=self._fix_attempt,
            verifier=verifier,
            trigger=_AlwaysTrigger(),
            max_retries=self._config.max_iterations,
            correction_fn=self._narrow_scope,
            name="swe",
        )

        loop_result = await runner.run(issue)
        total_ms = (time.monotonic() - start) * 1000

        patch = SWEPatch()
        if self._context and self._context.current_patch:
            patch.description = self._context.fix_proposal or "Auto-generated fix"

        if self._config.sandbox_enabled:
            await self._sandbox.cleanup()

        return SWEResult(
            success=loop_result.success,
            patch=patch,
            context=self._context or CodeContext(issue=issue, repo_path=repo),
            loop_result=loop_result,
            total_duration_ms=total_ms,
            iterations=loop_result.total_retries + 1,
        )

    async def _fix_attempt(self, issue: str) -> dict[str, Any]:
        ctx = self._context
        if ctx is None:
            return {"error": "No context"}

        ctx.iteration += 1
        explore = await self._explore(ctx)
        ctx.narrow(
            files=explore.files,
            search=explore.search_query,
            error=explore.error_context,
        )

        diagnosis = await self._diagnose(explore, ctx)
        ctx.error_analysis = diagnosis
        logger.info("Iteration %d diagnosis: %s", ctx.iteration, diagnosis[:100])

        patch = await self._propose_fix(explore, diagnosis, ctx)
        if patch.files:
            await patch.apply(ctx.repo_path)
            ctx.current_patch = patch.to_diff()
            ctx.fix_proposal = patch.description
            if self._config.verbose:
                logger.info("Applied patch:\n%s", ctx.current_patch)

        return {
            "iteration": ctx.iteration,
            "diagnosis": diagnosis,
            "patch": ctx.current_patch,
            "explored": list(ctx.explored_files),
        }

    async def _explore(self, ctx: CodeContext) -> ExploreResult:
        search_targets = self._config.target_files if self._config.target_files else ["src", "."]
        files: list[str] = []
        snippets: list[str] = []
        error_text = ctx.test_output

        for pattern in search_targets:
            found = await self._grep(ctx.repo_path, pattern, ctx.issue)
            files.extend(f for f in found if f not in ctx.explored_files)

        if not files:
            files = await self._glob_files(ctx.repo_path, ["**/*.py"])

        for f in files[:10]:
            content = await self._read_file(ctx.repo_path / f)
            if content:
                lines = content.splitlines()
                snippet = "\n".join(lines[:50])
                snippets.append(f"--- {f} ---\n{snippet}")

        return ExploreResult(
            files=files[:20],
            error_context=error_text,
            relevant_snippets=snippets,
            search_query=" | ".join(search_targets),
            summary=f"Found {len(files)} files, examined {len(snippets)}",
        )

    async def _diagnose(self, explore: ExploreResult, ctx: CodeContext) -> str:
        if self._model is None:
            return "No model configured — using fallback heuristic analysis"

        prompt = _DIAGNOSE_PROMPT.format(
            issue=ctx.issue[:500],
            test_output=ctx.test_output[:1000],
            snippets="\n".join(explore.relevant_snippets[:5])[:2000],
            files=", ".join(explore.files[:10]),
        )
        result: GenerationResult = await self._model.generate(prompt, max_tokens=1024)
        return result.text or "No analysis generated"

    async def _propose_fix(self, explore: ExploreResult, diagnosis: str, ctx: CodeContext) -> SWEPatch:
        patch = SWEPatch(description=diagnosis)
        target_files = await self._extract_targets(diagnosis, explore.files)

        for tf in target_files:
            path = ctx.repo_path / tf
            old = read_file_content(path)
            if not old:
                continue

            if self._model is None:
                continue

            prompt = _GENERATE_FIX_PROMPT.format(
                issue=ctx.issue[:300],
                analysis=diagnosis[:500],
                target_file=tf,
                content=old,
            )
            result: GenerationResult = await self._model.generate(prompt, max_tokens=4096)
            new_content = self._extract_code(result.text or "")

            if new_content and new_content != old:
                patch.add_file(tf, old, new_content)

        return patch

    def _narrow_scope(self, issue: str, output: dict[str, Any], evidence: str) -> str:
        ctx = self._context
        if ctx is None:
            return issue
        ctx.test_output = evidence
        if self._config.verbose:
            logger.info("Narrowing scope after failure: %s", evidence[:100])
        return issue

    async def _grep(self, repo: Path, pattern: str, context: str) -> list[str]:
        keywords = [w for w in context.split() if len(w) > 3][:5]
        results: set[str] = set()
        for kw in keywords:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "grep", "-rn", "-l", kw, "--include=*.py",
                    str(repo),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
                results.update(
                    str(Path(line).relative_to(repo))
                    for line in stdout.decode(errors="replace").splitlines() if line
                )
            except Exception:
                continue
        return sorted(results)[:20]

    async def _glob_files(self, repo: Path, patterns: list[str]) -> list[str]:
        import glob as glob_mod
        results: set[str] = set()
        for pat in patterns:
            matched = glob_mod.glob(str(repo / pat), recursive=True)
            results.update(
                str(Path(m).relative_to(repo)) for m in matched if Path(m).is_file()
            )
        return sorted(results)[:30]

    async def _read_file(self, path: Path) -> str:
        try:
            return await asyncio.to_thread(path.read_text, encoding="utf-8")
        except Exception:
            return ""

    async def _extract_targets(self, diagnosis: str, explored: list[str]) -> list[str]:
        for line in diagnosis.splitlines():
            if line.upper().startswith("AFFECTED_FILES:"):
                parts = line.split(":", 1)[1].strip()
                candidates = [p.strip() for p in parts.split(",") if p.strip()]
                valid = [c for c in candidates if c in explored or c.endswith(".py")]
                if valid:
                    return valid
        return explored[:3]

    def _extract_code(self, text: str) -> str:
        if "```" in text:
            parts = text.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1 and ("python" in part or not part.startswith("\n")):
                    code = part.split("\n", 1)[1] if "\n" in part else part
                    return code.strip()
        return text.strip()


class _AlwaysTrigger:
    async def wait(self) -> bool:
        return False

    async def reset(self) -> None:
        pass


def _make_reproduce_verifier(repo_path: str, test_command: str, timeout: float = 120.0) -> TestSuiteVerifier:
    return TestSuiteVerifier(command=test_command, timeout=timeout, cwd=repo_path)
