from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("zilli.swe.patch")


@dataclass
class PatchFile:
    path: str
    old_content: str
    new_content: str


@dataclass
class SWEPatch:
    files: list[PatchFile] = field(default_factory=list)
    description: str = ""

    def to_diff(self) -> str:
        lines: list[str] = []
        for pf in self.files:
            diff = difflib.unified_diff(
                pf.old_content.splitlines(keepends=True),
                pf.new_content.splitlines(keepends=True),
                fromfile=f"a/{pf.path}",
                tofile=f"b/{pf.path}",
            )
            lines.extend(diff)
        return "".join(lines)

    async def apply(self, repo_root: Path) -> list[Path]:
        written: list[Path] = []
        for pf in self.files:
            target = repo_root / pf.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(pf.new_content, encoding="utf-8")
            written.append(target)
            logger.info("Applied patch to %s", target)
        return written

    async def revert(self, repo_root: Path) -> list[Path]:
        reverted: list[Path] = []
        for pf in self.files:
            target = repo_root / pf.path
            if target.exists():
                target.write_text(pf.old_content, encoding="utf-8")
                reverted.append(target)
                logger.info("Reverted %s", target)
        return reverted

    def add_file(self, path: str, old: str, new: str) -> None:
        self.files.append(PatchFile(path=path, old_content=old, new_content=new))

    @property
    def total_changes(self) -> int:
        return sum(
            sum(1 for a, b in zip(pf.old_content.splitlines(), pf.new_content.splitlines()) if a != b)
            + abs(len(pf.new_content.splitlines()) - len(pf.old_content.splitlines()))
            for pf in self.files
        )


def read_file_content(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return ""


async def git_diff(repo_root: Path, staged: bool = False) -> str:
    import subprocess

    cmd = ["git", "diff"] if not staged else ["git", "diff", "--cached"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=repo_root, timeout=30,
        )
        return result.stdout
    except Exception as e:
        return f"Error getting diff: {e}"
