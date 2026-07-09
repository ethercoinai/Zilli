#!/usr/bin/env python3
"""SWE-agent driven batch audit — uses Zilli exploration pipeline per project."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class AuditFinding:
    severity: str
    category: str
    rule: str
    message: str
    file: str = ""
    line: int = 0
    snippet: str = ""


@dataclass
class ProjectAudit:
    name: str
    path: Path
    language: str
    file_count: int
    findings: list[AuditFinding] = field(default_factory=list)
    duration_ms: float = 0.0

    def add(self, sev: str, cat: str, rule: str, msg: str, f: str = "", line: int = 0, snippet: str = ""):
        self.findings.append(AuditFinding(sev, cat, rule, msg, f, line, snippet))

    @property
    def summary(self) -> dict:
        c = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for fi in self.findings:
            if fi.severity in c:
                c[fi.severity] += 1
        return c


def _detect_language(path: Path) -> str:
    exts = {}
    for f in path.rglob("*"):
        if f.is_file() and ".git" not in str(f) and "node_modules" not in str(f):
            exts[f.suffix] = exts.get(f.suffix, 0) + 1
    if exts.get(".py", 0) > 3:
        return "python"
    if exts.get(".ts", 0) > 3 or exts.get(".tsx", 0) > 3:
        return "typescript"
    if exts.get(".js", 0) > 3:
        return "javascript"
    if exts.get(".rs", 0) > 1:
        return "rust"
    if exts.get(".dart", 0) > 1:
        return "dart"
    if exts.get(".html", 0) > 3:
        return "html"
    return "other"


def _file_count(path: Path, lang: str) -> int:
    ext_map = {"python": ".py", "typescript": ".ts .tsx", "javascript": ".js .jsx",
               "rust": ".rs", "dart": ".dart", "html": ".html .css .js"}
    exts = ext_map.get(lang, ".py").split()
    count = 0
    for e in exts:
        count += sum(1 for _ in path.rglob(f"*{e}")
                     if ".git" not in str(_) and "node_modules" not in str(_))
    return count


# ── Audit patterns ───────────────────────────────────────────────────
# These mirror SWE agent's exploration phase — targeted grep for real issues

CHECKS: dict[str, list[tuple[str, str, str, str]]] = {
    # (severity, rule, glob, grep_pattern)
    "security": [
        ("high", "hardcoded-api-key", "*.py *.ts *.js *.rs",
         r'(api[_-]?key|secret|password|token)\s*[=:]\s*["\'][A-Za-z0-9_\-\.]{16,}'),
        ("high", "hardcoded-bearer", "*.py *.ts *.js",
         r'["\']Authorization["\']\s*:?\s*["\']Bearer\s+[A-Za-z0-9_\-\.]{10,}'),
        ("critical", "shell-injection", "*.py",
         r'(os\.system\(|subprocess\..*shell=True|asyncio\.create_subprocess_shell\()'),
        ("critical", "eval-exec", "*.py",
         r'(?<![_a-zA-Z])eval\(|(?<![_a-zA-Z])exec\(|__import__\('),
        ("high", "raw-sql", "*.py",
         r'execute\(["\'].*SELECT|raw\(["\'].*SELECT'),
    ],
    "bug": [
        ("high", "mutable-default", "*.py",
         r'def \w+\([^)]*=\s*(\[\]|\{\}|set\(\))'),
        ("medium", "broad-except", "*.py",
         r'except\s+Exception\s*:'),
        ("medium", "bare-except", "*.py",
         r'except\s*:'),
        ("medium", "async-blocking-io", "*.py",
         r'(\.read_text\(\)|\.write_text\(\)|\.read_bytes\(\)|\.write_bytes\(\))'),
    ],
    "complexity": [
        ("low", "large-function", "*.py",
         r'def \w+\(.*\):'),
    ],
}


async def _grep(repo: Path, pattern: str, include: str) -> list[tuple[str, int, str]]:
    results = []
    try:
        r = await asyncio.create_subprocess_exec(
            "grep", "-rn", "--include=" + include, "-E", pattern, str(repo),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(r.communicate(), timeout=30)
        for line in stdout.decode(errors="replace").splitlines():
            parts = line.split(":", 2)
            if len(parts) >= 2:
                line_no = int(parts[1]) if parts[1].isdigit() else 0
                snippet = parts[2][:120] if len(parts) > 2 else ""
                fpath = parts[0]
                try:
                    fpath = str(Path(fpath).relative_to(repo))
                except ValueError:
                    pass
                if "# noqa:" in snippet:
                    continue
                results.append((fpath, line_no, snippet))
    except (OSError, asyncio.TimeoutError):
        pass
    return results


async def _check_npm(repo: Path) -> list[AuditFinding]:
    findings = []
    pkg = repo / "package.json"
    if not pkg.exists():
        return findings
    try:
        r = await asyncio.create_subprocess_exec(
            "npm", "audit", "--json",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=repo,
        )
        stdout, _ = await asyncio.wait_for(r.communicate(), timeout=30)
        data = json.loads(stdout)
        for pkg_name, info in data.get("vulnerabilities", {}).items():
            sev = info.get("severity", "low")
            if sev in ("critical", "high"):
                findings.append(AuditFinding(
                    "high" if sev == "critical" else "medium",
                    "security", "npm-vuln",
                    f"{pkg_name}: {info.get('title', info.get('via', [{}])[0] if isinstance(info.get('via'), list) else 'unknown')}",
                ))
    except (OSError, json.JSONDecodeError, asyncio.TimeoutError):
        pass
    return findings


async def audit_project(path: Path) -> ProjectAudit:
    start = time.monotonic()
    lang = _detect_language(path)
    count = _file_count(path, lang)
    name = path.name if path.name not in ("backend", "frontend", "web", "ai", "mobile") else path.parent.name + "/" + path.name

    audit = ProjectAudit(name=name, path=path, language=lang, file_count=count)

    # Security checks
    for severity, rule, include, pattern in CHECKS["security"]:
        if include == "*.py *.ts *.js *.rs":
            inc = "*.py" if lang == "python" else ("*.ts" if lang == "typescript" else ("*.js" if lang == "javascript" else ("*.rs" if lang == "rust" else "*.py")))
        else:
            inc = include
        # Always check Python patterns for Python projects, etc.
        results = await _grep(path, pattern, inc)
        is_markdown = any(f.endswith(".md") for f, _, _ in results[:3])
        for fpath, line, snippet in results:
            # Skip known false positives: markdown docs, lockfiles, minified
            if fpath.endswith(".md") and is_markdown:
                continue
            if "yarn.lock" in fpath or "package-lock.json" in fpath:
                continue
            # Skip builtins.__import__ patch pattern (standard test mock)
            if rule == "eval-exec" and 'builtins.__import__' in snippet:
                continue
            msg = {"hardcoded-api-key": "Potential hardcoded credential",
                   "hardcoded-bearer": "Hardcoded bearer token",
                   "shell-injection": "Shell injection risk (use subprocess_exec)",
                   "eval-exec": "Dynamic code execution (eval/exec)",
                   "raw-sql": "Raw SQL query (use parameterized)"}.get(rule, rule)
            audit.add(severity, "security", rule, msg, fpath, line, snippet)

    # Bug checks (Python only)
    if lang == "python":
        for severity, rule, include, pattern in CHECKS["bug"]:
            results = await _grep(path, pattern, include)
            for fpath, line, snippet in results:
                # async-blocking-io false positive filter: skip if not inside async def
                if rule == "async-blocking-io":
                    full = path / fpath
                    in_async = False
                    if full.exists():
                        _lines = full.read_text().splitlines()
                        start = max(0, line - 2)
                        i = start
                        while i >= max(0, line - 20):
                            s = _lines[i].strip()
                            if s.startswith("async def "):
                                in_async = True
                                break
                            if s.startswith("def ") or s.startswith("class "):
                                break  # sync function or class — stop searching
                            i -= 1
                    if not in_async:
                        continue  # false positive: sync I/O in sync code
                msg = {"mutable-default": "Mutable default argument (use None)",
                       "broad-except": "Broad except Exception",
                       "bare-except": "Bare except (catches BaseException)",
                       "async-blocking-io": "Sync file I/O in async code"}.get(rule, rule)
                audit.add(severity, "bug", rule, msg, fpath, line, snippet)

    # npm audit
    if lang in ("typescript", "javascript"):
        audit.findings.extend(await _check_npm(path))

    audit.duration_ms = (time.monotonic() - start) * 1000
    return audit


# ── Projects ─────────────────────────────────────────────────────────

PROJECTS = [
    "/home/jackliao/gstack",
    "/home/jackliao/gbrain",
    "/home/jackliao/quant-loop",
    "/home/jackliao/x-automation",
    "/home/jackliao/文档/ETHER以太/Zilli/Zilli",
    "/home/jackliao/文档/ETHER以太/acas",
    "/home/jackliao/文档/ETHER以太/starmeters",
    "/home/jackliao/文档/MSTR/mstr_quant_arb",
    "/home/jackliao/文档/TSPOW/test-chain",
    "/home/jackliao/sinnet/backend",
    "/home/jackliao/sinnet/ai",
    "/home/jackliao/文档/ETHER以太/Zilli/zilli-rs",
    "/home/jackliao/Desktop/EthercoinCom",
    "/home/jackliao/Desktop/youtube-translate-extension",
    "/home/jackliao/ethercoin",
    "/home/jackliao/Desktop/ethercoinorg",
    "/home/jackliao/文档/ETHER以太/EthercoinCOM/ethercoin-management/backend",
    "/home/jackliao/文档/ETHER以太/EthercoinCOM/ethercoin-management/frontend",
    "/home/jackliao/文档/Openfans/web",
    "/home/jackliao/文档/BitcoinBall",
    "/home/jackliao/文档/TSPOW/simulator",
]


async def main():
    results: list[ProjectAudit] = []
    print(f"\n{'='*70}")
    print("  Zilli SWE Agent — 全项目批量审计")
    print("  审计模式: 真实代码问题扫描 (排除样式/TODO 噪音)")
    print(f"  项目数: {len(PROJECTS)}")
    print(f"{'='*70}\n")

    for p in PROJECTS:
        rp = Path(p)
        if not rp.exists():
            print(f"  ⚠  SKIP: {p}")
            continue
        name = rp.name if rp.name not in ("backend", "frontend", "web", "ai", "mobile") else rp.parent.name + "/" + rp.name
        print(f"  🔍  {name:30s} ... ", end="", flush=True)
        audit = await audit_project(rp)
        results.append(audit)
        s = audit.summary
        parts = []
        if s["critical"]:
            parts.append(f"{s['critical']}C")
        if s["high"]:
            parts.append(f"{s['high']}H")
        if s["medium"]:
            parts.append(f"{s['medium']}M")
        if s["low"]:
            parts.append(f"{s['low']}L")
        status = "  ".join(parts) if parts else "✅ clean"
        print(f"{audit.duration_ms:7.0f}ms  {status}")

    # ── Summary table ──
    print(f"\n{'='*70}")
    print("  审计汇总 (按严重度排序)")
    print(f"{'='*70}")
    header = f"  {'项目':30s} {'语言':10s} {'文件':>5s}  C   H   M   L"
    print(header)
    print(f"  {'-'*30} {'-'*10} {'-'*5}  {'-'*3} {'-'*3} {'-'*3} {'-'*3}")
    for r in sorted(results, key=lambda x: (-x.summary["critical"], -x.summary["high"], -x.summary["medium"])):
        s = r.summary
        print(f"  {r.name:30s} {r.language:10s} {r.file_count:5d}  {s['critical']:3d} {s['high']:3d} {s['medium']:3d} {s['low']:3d}")

    total = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for r in results:
        for k in total:
            total[k] += r.summary[k]
    print(f"  {'─'*30} {'─'*10} {'─'*5}  {'─'*3} {'─'*3} {'─'*3} {'─'*3}")
    print(f"  {'总计':30s} {'':10s} {'':5s}  {total['critical']:3d} {total['high']:3d} {total['medium']:3d} {total['low']:3d}")

    # ── Report ──
    rp = Path("swe_audit_report.md").resolve()
    with open(rp, "w") as f:
        f.write("# Zilli SWE Agent — 全项目代码审计报告\n\n")
        f.write(f"审计时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"项目数: {len(results)}\n\n")

        f.write("## 汇总\n\n")
        f.write("| 项目 | 语言 | 文件 | Critical | High | Medium | Low |\n")
        f.write("|------|------|-----:|--------:|-----:|-------:|----:|\n")
        for r in sorted(results, key=lambda x: (-x.summary["critical"], -x.summary["high"])):
            s = r.summary
            f.write(f"| {r.name} | {r.language} | {r.file_count} | "
                    f"{s['critical']} | {s['high']} | {s['medium']} | {s['low']} |\n")

        f.write("\n## 发现详情\n\n")
        for r in sorted(results, key=lambda x: (-x.summary["critical"], -x.summary["high"])):
            if not r.findings:
                continue
            f.write(f"### {r.name}\n\n")
            f.write(f"- 路径: `{r.path}`\n")
            f.write(f"- 语言: {r.language} | 文件: {r.file_count} | 耗时: {r.duration_ms:.0f}ms\n\n")

            sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            for fi in sorted(r.findings, key=lambda x: sev_order.get(x.severity, 99)):
                icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}
                fp = fi.file
                loc = f"`{fp}`:{fi.line}" if fi.line else f"`{fp}`"
                snippet = f" `{fi.snippet}`" if fi.snippet else ""
                f.write(f"- {icon.get(fi.severity, '⚪')} **[{fi.severity}]** {fi.category}/{fi.rule}: "
                        f"{fi.message} — {loc}{snippet}\n")
            f.write("\n")

    print(f"\n📄 {rp}")

    # ── Action plan (worst offenders) ──
    print(f"\n{'='*70}")
    print("  需立即修复的项目")
    print(f"{'='*70}")
    bad = [r for r in results if r.summary["critical"] > 0 or r.summary["high"] > 0]
    for r in sorted(bad, key=lambda x: -x.summary["high"]):
        print(f"\n  🚨 {r.name} ({r.language}, {r.file_count} files)")
        for fi in sorted(r.findings, key=lambda x: sev_order.get(x.severity, 99)):
            if fi.severity in ("critical", "high"):
                icon = {"critical": "🔴", "high": "🟠"}[fi.severity]
                print(f"    {icon} [{fi.severity}] {fi.rule}: {fi.message}")
                print(f"       {fi.file}:{fi.line}")

    if not bad:
        print("\n  ✅ 所有项目无 Critical/High 发现")


if __name__ == "__main__":
    asyncio.run(main())
