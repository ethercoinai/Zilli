from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

from zilli.audit import AuditEvent, AuditLevel, AuditLogger

if TYPE_CHECKING:
    from zilli.configs import ZilliConfig
from zilli.models.registry import ModelRegistry
from zilli.routing import LocalHybridRouter, RouteClassifier
from zilli.routing.router import RouteResult
from zilli.security.isolation import AccessLevel, DataIsolation, IsolationPolicy
from zilli.security.pii import Sanitizer

logger = logging.getLogger("zilli.industry")


class IndustryType(str, Enum):
    LEGAL = "legal"
    MEDICAL = "medical"
    FINANCIAL = "financial"
    EDUCATION = "education"

    def __str__(self) -> str:
        return self.value


@dataclass
class IndustryWorkflow:
    industry: IndustryType
    compliance_rules: list[str] = field(default_factory=list)
    access_level: AccessLevel = AccessLevel.CONFIDENTIAL
    require_audit: bool = True
    require_sanitization: bool = True
    retention_days: int = 90

    def sanitize_input(self, text: str, sanitizer: Optional[Sanitizer] = None) -> str:
        s = sanitizer or Sanitizer()
        result = s.sanitize(text)
        if result.findings:
            logger.info(
                "Sanitized %d PII items from %s input",
                len(result.findings), self.industry.value,
            )
        return result.sanitized

    def audit_call(self, audit_logger: AuditLogger, request: str, result: RouteResult, tenant: str = "default"):
        if not self.require_audit:
            return
        audit_logger.route_decision(
            route_type=result.route_type.value,
            request=request,
            reason=result.decision.reason,
            tenant_id=tenant,
        )
        if result.error:
            audit_logger.log(AuditEvent(
                event_type="industry_error",
                level=AuditLevel.ERROR,
                message=f"{self.industry.value} workflow error: {result.error}",
                tenant_id=tenant,
            ))


LEGAL = IndustryWorkflow(
    industry=IndustryType.LEGAL,
    access_level=AccessLevel.RESTRICTED,
    compliance_rules=[
        "attorney-client privilege must be preserved",
        "no client confidential information outside local server",
        "all model calls must be audited",
        "output must cite applicable legal standards",
    ],
)

MEDICAL = IndustryWorkflow(
    industry=IndustryType.MEDICAL,
    access_level=AccessLevel.RESTRICTED,
    compliance_rules=[
        "HIPAA compliance required",
        "all PHI must be detected and handled per policy",
        "no patient data outside local server",
        "diagnostic suggestions must include disclaimer",
    ],
)

FINANCIAL = IndustryWorkflow(
    industry=IndustryType.FINANCIAL,
    access_level=AccessLevel.CONFIDENTIAL,
    compliance_rules=[
        "SOX compliance required",
        "all financial data must be audited",
        "no PII in audit logs",
        "risk assessments must clearly label confidence levels",
    ],
)

EDUCATION = IndustryWorkflow(
    industry=IndustryType.EDUCATION,
    access_level=AccessLevel.CONFIDENTIAL,
    compliance_rules=[
        "FERPA compliance required",
        "student PII must be removed from model inputs",
        "grades and assessments must be anonymized",
    ],
)


class WorkflowRegistry:
    _WORKFLOWS: dict[IndustryType, IndustryWorkflow] = {
        IndustryType.LEGAL: LEGAL,
        IndustryType.MEDICAL: MEDICAL,
        IndustryType.FINANCIAL: FINANCIAL,
        IndustryType.EDUCATION: EDUCATION,
    }

    def __init__(
        self,
        model_registry: Optional[ModelRegistry] = None,
        audit_logger: Optional[AuditLogger] = None,
        data_isolation: Optional[DataIsolation] = None,
        config: Optional["ZilliConfig"] = None,
    ):
        self.model_registry = model_registry or ModelRegistry(config=config)
        self.config = config

        audit_cfg = getattr(config, "audit", None) if config else None
        log_dir = audit_cfg.log_dir if audit_cfg else "./audit_logs"
        sanitize = audit_cfg.sanitize if audit_cfg else True
        self.audit_logger = audit_logger or AuditLogger(log_dir=log_dir, sanitize=sanitize)

        self.data_isolation = data_isolation or DataIsolation()

    def get_workflow(self, industry: IndustryType) -> IndustryWorkflow:
        return self._WORKFLOWS.get(industry)

    def list_industries(self) -> list[dict]:
        return [
            {
                "id": ind.value,
                "compliance_rules": wf.compliance_rules,
                "access_level": wf.access_level.value,
                "require_audit": wf.require_audit,
                "retention_days": wf.retention_days,
            }
            for ind, wf in self._WORKFLOWS.items()
        ]

    async def run(
        self,
        request: str,
        industry: IndustryType,
        tenant_id: str = "default",
        force_full_route: bool = False,
        sanitize: bool = True,
    ) -> RouteResult:
        workflow = self.get_workflow(industry)
        if workflow is None:
            raise ValueError(f"Unknown industry: {industry}")

        policy = IsolationPolicy(
            tenant_id=tenant_id,
            access_level=workflow.access_level,
            require_sanitization=workflow.require_sanitization,
            audit_required=workflow.require_audit,
            retention_days=workflow.retention_days,
        )
        self.data_isolation.register_tenant(tenant_id, policy)

        processed_request = request
        if sanitize and workflow.require_sanitization:
            processed_request = workflow.sanitize_input(request)

        classifier = RouteClassifier(model_registry=self.model_registry)
        router = LocalHybridRouter(
            registry=self.model_registry,
            classifier=classifier,
        )

        result = await router.run(
            request=processed_request,
            industry=industry.value,
            force_full_route=force_full_route,
        )

        if workflow.require_audit:
            workflow.audit_call(self.audit_logger, request, result, tenant_id)

        return result
