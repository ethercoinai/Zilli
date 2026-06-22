from zilli.security.isolation import AccessLevel, DataIsolation, IsolationPolicy
from zilli.security.pii import PIICategory, PIIDetector, Sanitizer


class TestPIIDetector:
    def setup_method(self):
        self.detector = PIIDetector()

    def test_detect_email(self):
        findings = self.detector.detect("Contact me at test@example.com")
        assert len(findings) == 1
        assert findings[0].category == PIICategory.EMAIL
        assert findings[0].text == "test@example.com"

    def test_detect_phone(self):
        findings = self.detector.detect("Call 13800138000 for details")
        assert len(findings) == 1
        assert findings[0].category == PIICategory.PHONE

    def test_detect_ssn(self):
        findings = self.detector.detect("SSN: 123-45-6789")
        assert len(findings) == 1
        assert findings[0].category == PIICategory.SSN

    def test_detect_chinese_id(self):
        findings = self.detector.detect("ID: 110101199001011234")
        categories = {f.category for f in findings}
        assert PIICategory.CHINESE_ID in categories

    def test_no_pii(self):
        findings = self.detector.detect("Hello, this is a simple message.")
        assert len(findings) == 0

    def test_has_pii(self):
        assert self.detector.has_pii("email@test.com") is True
        assert self.detector.has_pii("plain text") is False

    def test_multiple_findings(self):
        text = "Email: a@b.com, Phone: 13800138000"
        findings = self.detector.detect(text)
        assert len(findings) == 2
        assert findings[0].start < findings[1].start

    def test_ip_address(self):
        findings = self.detector.detect("Server: 192.168.1.1")
        assert len(findings) == 1
        assert findings[0].category == PIICategory.IP_ADDRESS

    def test_credit_card(self):
        findings = self.detector.detect("Card: 4111111111111111")
        assert len(findings) == 1
        assert findings[0].category == PIICategory.CREDIT_CARD


class TestSanitizer:
    def setup_method(self):
        self.sanitizer = Sanitizer()

    def test_sanitize_email(self):
        result = self.sanitizer.sanitize("Email: user@test.com")
        assert "***" in result.sanitized
        assert "user@test.com" not in result.sanitized
        assert len(result.findings) == 1

    def test_sanitize_multiple(self):
        result = self.sanitizer.sanitize("a@b.com and 13800138000")
        assert "***" in result.sanitized
        assert len(result.findings) == 2

    def test_no_pii(self):
        result = self.sanitizer.sanitize("Hello world")
        assert result.sanitized == "Hello world"
        assert len(result.findings) == 0

    def test_sanitize_for_log(self):
        sanitized = self.sanitizer.sanitize_for_log("Email: a@b.com")
        assert "a@b.com" not in sanitized


class TestDataIsolation:
    def setup_method(self):
        self.isolation = DataIsolation()

    def test_default_policy(self):
        policy = self.isolation.get_policy("unknown")
        assert policy.tenant_id == "default"
        assert policy.access_level == AccessLevel.INTERNAL

    def test_register_tenant(self):
        policy = IsolationPolicy(
            tenant_id="acme_corp",
            access_level=AccessLevel.CONFIDENTIAL,
        )
        self.isolation.register_tenant("acme_corp", policy)
        loaded = self.isolation.get_policy("acme_corp")
        assert loaded.access_level == AccessLevel.CONFIDENTIAL
        assert loaded.require_sanitization is True

    def test_access_check(self):
        policy = IsolationPolicy(
            tenant_id="law_firm",
            access_level=AccessLevel.RESTRICTED,
            allowed_roles=["planner", "reviewer"],
        )
        self.isolation.register_tenant("law_firm", policy)
        assert self.isolation.check_access("law_firm", "planner") is True
        assert self.isolation.check_access("law_firm", "executor") is False

    def test_remove_tenant(self):
        self.isolation.register_tenant("test", IsolationPolicy(tenant_id="test"))
        self.isolation.remove_tenant("test")
        assert self.isolation.get_policy("test").tenant_id == "default"

    def test_list_tenants(self):
        self.isolation.register_tenant("a", IsolationPolicy(tenant_id="a"))
        self.isolation.register_tenant("b", IsolationPolicy(tenant_id="b"))
        tenants = self.isolation.list_tenants()
        assert len(tenants) == 2
