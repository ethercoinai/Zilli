from __future__ import annotations

import asyncio
import logging
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("zilli.swe.sandbox")

try:
    HAS_DOCKER = subprocess.run(
        ["docker", "--version"], capture_output=True, text=True, timeout=10,
    ).returncode == 0
except Exception:
    HAS_DOCKER = False


@dataclass
class SandboxConfig:
    image: str = "python:3.12-slim"
    workdir: str = "/repo"
    memory_limit: str = "4g"
    cpu_limit: str = "2"
    network_disabled: bool = False


@dataclass
class CommandResult:
    stdout: str
    stderr: str
    returncode: int
    duration_ms: float = 0.0


class Sandbox:
    def __init__(self, config: Optional[SandboxConfig] = None):
        self._config = config or SandboxConfig()
        self._container_id: Optional[str] = None
        self._tmp_dir: Optional[Path] = None

    async def ensure_container(self, repo_path: Path) -> str:
        if self._container_id:
            return self._container_id
        if not HAS_DOCKER:
            logger.warning("Docker not available, running locally")
            return ""
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="zilli_swe_"))
        container_id = await self._run_host(
            ["docker", "run", "-d",
             "--rm",
             "--memory", self._config.memory_limit,
             "--cpus", self._config.cpu_limit,
             "-w", self._config.workdir,
             "-v", f"{repo_path.resolve()}:{self._config.workdir}",
             self._config.image,
             "tail", "-f", "/dev/null"],
        )
        self._container_id = container_id.strip()
        logger.info("Sandbox container %s ready", self._container_id[:12])
        return self._container_id

    async def run_command(self, cmd: str, cwd: Optional[str] = None, timeout: float = 120.0) -> CommandResult:
        import time
        start = time.monotonic()
        if self._container_id:
            docker_cmd = ["docker", "exec"]
            if cwd:
                docker_cmd.extend(["-w", cwd])
            docker_cmd.extend([self._container_id, "sh", "-c", cmd])
            proc = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        else:
            proc = await asyncio.create_subprocess_exec(
                *shlex.split(cmd) if not any(c in cmd for c in "|;&") else ["sh", "-c", cmd],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
            )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            duration = (time.monotonic() - start) * 1000
            return CommandResult(
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
                returncode=proc.returncode or 0,
                duration_ms=duration,
            )
        except asyncio.TimeoutError:
            proc.kill()
            duration = (time.monotonic() - start) * 1000
            return CommandResult(
                stdout="", stderr=f"Timed out after {timeout}s",
                returncode=-1, duration_ms=duration,
            )

    async def copy_to(self, src: str, dst: str) -> None:
        if not self._container_id:
            return
        await self._run_host(["docker", "cp", src, f"{self._container_id}:{dst}"])

    async def copy_from(self, src: str, dst: str) -> None:
        if not self._container_id:
            return
        await self._run_host(["docker", "cp", f"{self._container_id}:{src}", dst])

    async def cleanup(self) -> None:
        if self._container_id:
            try:
                await self._run_host(["docker", "rm", "-f", self._container_id])
            except Exception as e:
                logger.warning("Container cleanup error: %s", e)
            self._container_id = None
        if self._tmp_dir and self._tmp_dir.exists():
            import shutil
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None

    async def _run_host(self, cmd: list[str]) -> str:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        return stdout.decode(errors="replace").strip()
