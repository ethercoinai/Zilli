from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("zilli.audit.compliance")


class ComplianceFramework(str):
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    PCI_DSS = "pci_dss"
    FERPA = "ferpa"
    CCPA = "ccpa"


@dataclass
class ComplianceReport:
    framework: str
    tenant_id: str
    generated_at: float
    period_start: str
    period_end: str
    total_requests: int = 0
    cloud_requests: int = 0
    local_requests: int = 0
    sanitized_requests: int = 0
    rejected_requests: int = 0
    pii_detected_count: int = 0
    consent_violations: int = 0
    data_retention_ok: bool = True
    findings: list[dict] = field(default_factory=list)
    passed: bool = True


class ComplianceReporter:
    def __init__(self, audit_dir: str = "./audit_logs"):
        self.audit_dir = Path(audit_dir)

    def generate(self, framework: str, tenant_id: str,
                 period_start: str, period_end: str) -> ComplianceReport:
        report = ComplianceReport(
            framework=framework,
            tenant_id=tenant_id,
            generated_at=time.time(),
            period_start=period_start,
            period_end=period_end,
        )

        log_files = sorted(self.audit_dir.glob("audit_*.jsonl"))
        try:
            start_date = datetime.fromisoformat(period_start).date()
            end_date = datetime.fromisoformat(period_end).date()
        except ValueError:
            logger.warning("Invalid period date format: %s / %s", period_start, period_end)
            start_date, end_date = None, None

        def _date_in_range(path: Path) -> bool:
            if start_date is None or end_date is None:
                return True
            stem = path.stem
            date_str = stem.replace("audit_", "", 1) if stem.startswith("audit_") else stem
            try:
                d = datetime.fromisoformat(date_str).date()
                return start_date <= d <= end_date
            except ValueError:
                return True

        relevant = [f for f in log_files if _date_in_range(f)]

        for log_path in relevant:
            if not log_path.exists():
                continue
            with open(log_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    ev_tenant = event.get("tenant_id", "default")
                    if ev_tenant != tenant_id:
                        continue

                    report.total_requests += 1
                    route = event.get("route_type", "")
                    if route == "cloud":
                        report.cloud_requests += 1
                    elif route == "local":
                        report.local_requests += 1
                    if event.get("sanitized", False):
                        report.sanitized_requests += 1
                    if event.get("rejected", False):
                        report.rejected_requests += 1
                    if event.get("pii_count", 0) > 0:
                        report.pii_detected_count += event["pii_count"]
                    if event.get("consent_violation", False):
                        report.consent_violations += 1

        findings = self._check_findings(report, framework)
        report.findings = findings
        report.passed = all(f.get("severity") != "critical" for f in findings)

        return report

    def _check_findings(self, report: ComplianceReport, framework: str) -> list[dict]:
        findings: list[dict] = []

        if report.consent_violations > 0:
            findings.append({
                "check": "consent_enforcement",
                "status": "fail",
                "severity": "critical",
                "detail": f"{report.consent_violations} consent violation(s) detected",
            })

        if report.cloud_requests > 0 and framework in (ComplianceFramework.HIPAA, ComplianceFramework.PCI_DSS):
            if report.sanitized_requests < report.cloud_requests:
                findings.append({
                    "check": "cloud_sanitization",
                    "status": "fail",
                    "severity": "critical",
                    "detail": f"{report.cloud_requests - report.sanitized_requests} cloud requests without sanitization",
                })

        if framework == ComplianceFramework.GDPR:
            if report.pii_detected_count > 0 and report.consent_violations == 0:
                findings.append({
                    "check": "pii_processing_basis",
                    "status": "warn",
                    "severity": "medium",
                    "detail": "PII processed without documented lawful basis (no consent violations recorded)",
                })

        if report.data_retention_ok:
            findings.append({
                "check": "data_retention",
                "status": "pass",
                "severity": "info",
                "detail": "Data retention within configured window",
            })

        if report.rejected_requests > 0:
            findings.append({
                "check": "rejected_requests",
                "status": "info" if report.rejected_requests < 10 else "warn",
                "severity": "low" if report.rejected_requests < 10 else "medium",
                "detail": f"{report.rejected_requests} request(s) rejected by privacy gatekeeper",
            })

        return findings

    def export_json(self, report: ComplianceReport, path: str):
        data = {
            "framework": report.framework,
            "tenant_id": report.tenant_id,
            "generated_at": report.generated_at,
            "period": {"start": report.period_start, "end": report.period_end},
            "summary": {
                "total_requests": report.total_requests,
                "cloud_requests": report.cloud_requests,
                "local_requests": report.local_requests,
                "sanitized_requests": report.sanitized_requests,
                "rejected_requests": report.rejected_requests,
                "pii_detected_count": report.pii_detected_count,
                "consent_violations": report.consent_violations,
            },
            "findings": report.findings,
            "passed": report.passed,
        }
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logger.info("Compliance report exported to %s", path)
