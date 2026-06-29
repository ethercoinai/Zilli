from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CodeContext:
    issue: str
    repo_path: Path
    explored_files: set[str] = field(default_factory=set)
    search_queries: list[str] = field(default_factory=list)
    test_output: str = ""
    error_analysis: str = ""
    fix_proposal: str = ""
    current_patch: str = ""
    iteration: int = 0
    narrowed: bool = False

    def summarize(self) -> str:
        lines = [
            f"Issue: {self.issue[:200]}",
            f"Repository: {self.repo_path}",
            f"Iteration: {self.iteration}",
            f"Files explored: {len(self.explored_files)}",
        ]
        if self.explored_files:
            lines.append("Explored files:")
            for f in list(self.explored_files)[:15]:
                lines.append(f"  {f}")
        if self.test_output:
            lines.append(f"Last test output:\n{self.test_output[:500]}")
        if self.error_analysis:
            lines.append(f"Error analysis:\n{self.error_analysis[:300]}")
        if self.fix_proposal:
            lines.append(f"Fix proposal:\n{self.fix_proposal[:300]}")
        return "\n".join(lines)

    def narrow(self, files: list[str], search: str, error: str) -> None:
        self.explored_files.update(files)
        if search:
            self.search_queries.append(search)
        if error:
            self.error_analysis = error
        self.narrowed = True


@dataclass
class ExploreResult:
    files: list[str]
    error_context: str
    relevant_snippets: list[str] = field(default_factory=list)
    search_query: str = ""
    summary: str = ""
