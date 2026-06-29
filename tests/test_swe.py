from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from zilli.loops.verification import TestSuiteVerifier
from zilli.models.base import GenerationResult
from zilli.swe.agent import SWEAgent, SWEConfig, SWEResult
from zilli.swe.context import CodeContext, ExploreResult
from zilli.swe.patch import SWEPatch, read_file_content


class TestCodeContext:
    def test_summarize_basic(self):
        ctx = CodeContext(issue="Fix the bug", repo_path=Path("/tmp/repo"))
        summary = ctx.summarize()
        assert "Fix the bug" in summary
        assert "/tmp/repo" in summary

    def test_summarize_with_explored_files(self):
        ctx = CodeContext(
            issue="Fix parser",
            repo_path=Path("/tmp/repo"),
            explored_files={"src/parser.py", "src/lexer.py"},
        )
        summary = ctx.summarize()
        assert "src/parser.py" in summary
        assert "src/lexer.py" in summary

    def test_narrow(self):
        ctx = CodeContext(issue="Fix", repo_path=Path("/tmp/repo"))
        ctx.narrow(files=["src/bug.py"], search="error", error="NameError")
        assert "src/bug.py" in ctx.explored_files
        assert ctx.error_analysis == "NameError"
        assert ctx.narrowed is True

    def test_narrow_appends_to_existing(self):
        ctx = CodeContext(issue="Fix", repo_path=Path("/tmp/repo"), explored_files={"a.py"})
        ctx.narrow(files=["b.py"], search="", error="")
        assert "a.py" in ctx.explored_files
        assert "b.py" in ctx.explored_files


class TestExploreResult:
    def test_dataclass(self):
        r = ExploreResult(files=["a.py", "b.py"], error_context="Error")
        assert r.files == ["a.py", "b.py"]
        assert r.error_context == "Error"
        assert r.relevant_snippets == []


class TestSWEPatch:
    def test_empty_patch(self):
        p = SWEPatch()
        assert p.to_diff() == ""
        assert p.total_changes == 0

    def test_to_diff_with_single_file(self):
        p = SWEPatch()
        p.add_file("src/main.py", "print('old')\n", "print('new')\n")
        diff = p.to_diff()
        assert "src/main.py" in diff
        assert "-print('old')" in diff
        assert "+print('new')" in diff

    def test_total_changes_counts(self):
        p = SWEPatch()
        p.add_file("a.py", "line1\nline2\n", "line1\nchanged\n")
        assert p.total_changes == 1

    @pytest.mark.asyncio
    async def test_apply_and_revert(self, tmp_path):
        p = SWEPatch()
        p.add_file("sub/test.py", "old content\n", "new content\n")
        applied = await p.apply(tmp_path)
        assert len(applied) == 1
        assert (tmp_path / "sub" / "test.py").exists()
        reverted = await p.revert(tmp_path)
        assert len(reverted) == 1

    def test_read_file_content(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        assert read_file_content(f) == "hello"
        assert read_file_content(tmp_path / "nonexistent.txt") == ""


class TestSWEConfig:
    def test_defaults(self):
        cfg = SWEConfig()
        assert cfg.max_iterations == 3
        assert cfg.test_command == "python -m pytest tests/ -x -q"
        assert cfg.sandbox_enabled is False
        assert cfg.verbose is False

    def test_custom_config(self):
        cfg = SWEConfig(
            model_name="gpt-4",
            max_iterations=5,
            test_command="pytest -x",
            sandbox_enabled=True,
        )
        assert cfg.model_name == "gpt-4"
        assert cfg.max_iterations == 5
        assert cfg.test_command == "pytest -x"
        assert cfg.sandbox_enabled is True


class TestSWEAgent:
    @pytest.mark.asyncio
    async def test_run_missing_repo(self):
        result = await SWEAgent(SWEConfig()).run("fix bug", "/nonexistent/path")
        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_run_with_mock_model(self, tmp_path):
        mock_model = AsyncMock()
        mock_model.generate.return_value = GenerationResult(
            text="ROOT_CAUSE: Test failure\nFIX: Fix the test\nAFFECTED_FILES: test_file.py",
            model_name="mock",
        )

        cfg = SWEConfig(max_iterations=1)
        agent = SWEAgent(cfg, model_backend=mock_model)
        result = await agent.run("fix issue", str(tmp_path))
        assert result.context is not None
        assert result.context.iteration >= 0

    @pytest.mark.asyncio
    async def test_explore(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("def foo():\n    pass\n")
        agent = SWEAgent(SWEConfig())
        ctx = CodeContext(issue="fix foo", repo_path=tmp_path)
        result = await agent._explore(ctx)
        assert len(result.files) > 0
        assert "src/main.py" in result.files

    @pytest.mark.asyncio
    async def test_explore_empty_repo(self, tmp_path):
        agent = SWEAgent(SWEConfig())
        ctx = CodeContext(issue="fix", repo_path=tmp_path)
        result = await agent._explore(ctx)
        assert result.files == []

    @pytest.mark.asyncio
    async def test_diagnose_with_mock(self):
        mock_model = AsyncMock()
        mock_model.generate.return_value = GenerationResult(
            text="ROOT_CAUSE: Import error\nFIX: Add missing import",
            model_name="mock",
        )
        agent = SWEAgent(SWEConfig(), model_backend=mock_model)
        ctx = CodeContext(issue="import error", repo_path=Path("/tmp"))
        explore = ExploreResult(files=["main.py"], error_context="Error")
        diagnosis = await agent._diagnose(explore, ctx)
        assert "Import" in diagnosis
        mock_model.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_diagnose_no_model(self):
        agent = SWEAgent(SWEConfig())
        ctx = CodeContext(issue="fix", repo_path=Path("/tmp"))
        explore = ExploreResult(files=[], error_context="")
        diagnosis = await agent._diagnose(explore, ctx)
        assert "fallback" in diagnosis or "No model" in diagnosis

    def test_extract_code_with_markdown(self):
        agent = SWEAgent(SWEConfig())
        text = "```python\nprint('hello')\n```"
        assert agent._extract_code(text) == "print('hello')"

    def test_extract_code_plain(self):
        agent = SWEAgent(SWEConfig())
        assert agent._extract_code("print('hello')") == "print('hello')"

    def test_extract_code_empty(self):
        agent = SWEAgent(SWEConfig())
        assert agent._extract_code("") == ""


class TestTestSuiteVerifier:
    @pytest.mark.asyncio
    async def test_verify_timeout(self):
        v = TestSuiteVerifier("sleep 10", timeout=0.1)
        result = await v.verify(None, None)
        assert result.passed is False
        assert "Timed out" in result.evidence

    @pytest.mark.asyncio
    async def test_verify_success(self):
        v = TestSuiteVerifier("true", timeout=5.0)
        result = await v.verify(None, None)
        assert result.passed is True
        assert "Exit code 0" in result.evidence

    @pytest.mark.asyncio
    async def test_verify_failure(self):
        v = TestSuiteVerifier("false", timeout=5.0)
        result = await v.verify(None, None)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_verify_bad_command(self):
        v = TestSuiteVerifier("nonexistent_command_xyz", timeout=5.0)
        result = await v.verify(None, None)
        assert result.passed is False


class TestSWEResult:
    def test_defaults(self):
        ctx = CodeContext(issue="fix", repo_path=Path("/tmp"))
        result = SWEResult(success=False, patch=SWEPatch(), context=ctx)
        assert result.success is False
        assert result.iterations == 0
        assert result.loop_result is None

    def test_success_with_data(self):
        ctx = CodeContext(issue="fix", repo_path=Path("/tmp"))
        patch = SWEPatch(description="test fix")
        result = SWEResult(
            success=True, patch=patch, context=ctx,
            total_duration_ms=100.0, iterations=2,
        )
        assert result.success is True
        assert result.patch.description == "test fix"
        assert result.total_duration_ms == 100.0
        assert result.iterations == 2

    def test_error_result(self):
        ctx = CodeContext(issue="fix", repo_path=Path("/tmp"))
        result = SWEResult(success=False, patch=SWEPatch(), context=ctx, error="something broke")
        assert result.error == "something broke"
