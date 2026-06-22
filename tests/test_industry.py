from zilli.industry.workflows import IndustryType, WorkflowRegistry


class TestIndustryType:
    def test_enum_values(self):
        assert IndustryType.LEGAL.value == "legal"
        assert IndustryType.MEDICAL.value == "medical"
        assert IndustryType.FINANCIAL.value == "financial"
        assert IndustryType.EDUCATION.value == "education"

    def test_enum_str(self):
        assert str(IndustryType.MEDICAL) == "medical"


class TestIndustryWorkflow:
    def test_legal_workflow(self):
        from zilli.industry.workflows import LEGAL
        assert LEGAL.industry == IndustryType.LEGAL
        assert len(LEGAL.compliance_rules) > 0
        assert any("attorney-client" in r for r in LEGAL.compliance_rules)

    def test_medical_workflow(self):
        from zilli.industry.workflows import MEDICAL
        assert MEDICAL.industry == IndustryType.MEDICAL
        assert any("HIPAA" in r for r in MEDICAL.compliance_rules)

    def test_financial_workflow(self):
        from zilli.industry.workflows import FINANCIAL
        assert FINANCIAL.industry == IndustryType.FINANCIAL
        assert any("SOX" in r for r in FINANCIAL.compliance_rules)

    def test_education_workflow(self):
        from zilli.industry.workflows import EDUCATION
        assert EDUCATION.industry == IndustryType.EDUCATION
        assert any("FERPA" in r for r in EDUCATION.compliance_rules)


class TestWorkflowRegistry:
    def test_list_industries(self):
        registry = WorkflowRegistry()
        industries = registry.list_industries()
        assert len(industries) == 4
        ids = {ind["id"] for ind in industries}
        assert ids == {"legal", "medical", "financial", "education"}

    def test_get_workflow(self):
        registry = WorkflowRegistry()
        legal = registry.get_workflow(IndustryType.LEGAL)
        assert legal is not None
        assert legal.industry == IndustryType.LEGAL

    def test_get_unknown(self):
        registry = WorkflowRegistry()
        result = registry.get_workflow("unknown")  # type: ignore
        assert result is None

    def test_sanitize_input_removes_pii(self):
        from zilli.industry.workflows import MEDICAL
        result = MEDICAL.sanitize_input("Patient email: test@hospital.com")
        assert "test@hospital.com" not in result
        assert "***" in result

    def test_sanitize_input_clean(self):
        from zilli.industry.workflows import LEGAL
        result = LEGAL.sanitize_input("Standard legal question")
        assert result == "Standard legal question"
