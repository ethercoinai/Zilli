from zilli.privacy.classifier import DataClass, DataClassifier
from zilli.privacy.consent import ConsentManager, DataUse
from zilli.privacy.engine import PrivacyEngine, SanitizationMode
from zilli.privacy.policy import DataGovernancePolicy, PolicyStore
from zilli.privacy.reid import ReIDAssessor, ReIDRisk


class TestDataClassifier:
    def setup_method(self):
        self.classifier = DataClassifier()

    def test_public_text(self):
        result = self.classifier.classify("Hello, this is a test")
        assert result.data_class == DataClass.PUBLIC
        assert result.requires_sanitization is False
        assert result.can_use_cloud is True

    def test_email_makes_confidential(self):
        result = self.classifier.classify("Contact me at test@example.com")
        assert result.data_class == DataClass.CONFIDENTIAL
        assert result.requires_sanitization is True

    def test_credit_card_makes_regulated(self):
        result = self.classifier.classify("Card: 4111111111111111")
        assert result.data_class == DataClass.REGULATED
        assert result.can_use_cloud is False

    def test_regulated_keyword(self):
        result = self.classifier.classify("This contains PHI data")
        assert result.data_class == DataClass.REGULATED

    def test_classify_batch(self):
        results = self.classifier.classify_batch(["hello", "email@test.com"])
        assert results[0].data_class == DataClass.PUBLIC
        assert results[1].data_class == DataClass.CONFIDENTIAL


class TestPolicyStore:
    def setup_method(self):
        self.store = PolicyStore()

    def test_default_policy(self):
        policy = self.store.get("unknown")
        assert policy.tenant_id == "unknown"
        assert policy.max_allowed_class == DataClass.CONFIDENTIAL

    def test_set_and_get(self):
        policy = DataGovernancePolicy(
            tenant_id="acme",
            max_allowed_class=DataClass.INTERNAL,
        )
        self.store.set("acme", policy)
        loaded = self.store.get("acme")
        assert loaded.max_allowed_class == DataClass.INTERNAL

    def test_allows_cloud(self):
        policy = DataGovernancePolicy(max_allowed_class=DataClass.CONFIDENTIAL)
        assert policy.allows_cloud_for(DataClass.PUBLIC) is True
        assert policy.allows_cloud_for(DataClass.INTERNAL) is True
        assert policy.allows_cloud_for(DataClass.CONFIDENTIAL) is True
        assert policy.allows_cloud_for(DataClass.RESTRICTED) is False


class TestReIDAssessor:
    def setup_method(self):
        self.assessor = ReIDAssessor()

    def test_clean_text_no_risk(self):
        result = self.assessor.assess("This is a clean text")
        assert result.risk == ReIDRisk.NONE
        assert result.direct_identifiers == 0

    def test_direct_identifiers_critical(self):
        result = self.assessor.assess("Email: test@example.com")
        assert result.risk == ReIDRisk.CRITICAL
        assert result.direct_identifiers > 0

    def test_safe_for_cloud(self):
        result = self.assessor.assess("clean text")
        assert self.assessor.is_safe_for_cloud(result) is True

    def test_quasi_identifier_detected(self):
        result = self.assessor.assess("The patient is male and born on 01/15/1990")
        assert len(result.quasi_identifiers) > 0
        assert result.risk in (ReIDRisk.LOW, ReIDRisk.MEDIUM)


class TestConsentManager:
    def setup_method(self):
        self.manager = ConsentManager()

    def test_grant_and_check(self):
        self.manager.grant("tenant1", "user1", DataUse.LOCAL_INFERENCE)
        assert self.manager.check("tenant1", "user1", DataUse.LOCAL_INFERENCE) is True
        assert self.manager.check("tenant1", "user1", DataUse.CLOUD_INFERENCE) is False

    def test_revoke(self):
        self.manager.grant("tenant1", "user1", DataUse.CLOUD_INFERENCE)
        assert self.manager.check("tenant1", "user1", DataUse.CLOUD_INFERENCE) is True
        self.manager.revoke("tenant1", "user1", DataUse.CLOUD_INFERENCE)
        assert self.manager.check("tenant1", "user1", DataUse.CLOUD_INFERENCE) is False

    def test_list_active(self):
        self.manager.grant("t1", "u1", DataUse.LOCAL_INFERENCE)
        self.manager.grant("t1", "u1", DataUse.AUDIT)
        active = self.manager.list_active("t1")
        assert len(active) == 2


class TestPrivacyEngine:
    def setup_method(self):
        self.engine = PrivacyEngine()

    def test_public_text_passthrough(self):
        result = self.engine.evaluate("Hello world")
        assert result.passed is True
        assert result.sanitized_text == "Hello world"
        assert result.can_proceed_local is True

    def test_sensitive_text_auto_sanitize(self):
        result = self.engine.evaluate("Email: test@example.com")
        assert result.passed is True
        assert "test@example.com" not in result.sanitized_text

    def test_restricted_rejected_for_cloud(self):
        result = self.engine.evaluate(
            "This is classified as trade secret data",
            mode=SanitizationMode.NONE,
            require_cloud=True,
        )
        assert result.can_proceed_cloud is False

    def test_strict_mode_sanitizes_all(self):
        result = self.engine.evaluate(
            "My email is a@b.com and phone is 13800138000",
            mode=SanitizationMode.STRICT,
        )
        assert "***" in result.sanitized_text
        assert "a@b.com" not in result.sanitized_text
        assert "13800138000" not in result.sanitized_text

    def test_consent_check(self):
        self.engine.consent.grant("t1", "u1", DataUse.LOCAL_INFERENCE)
        result = self.engine.evaluate("Hello", tenant_id="t1", user_id="u1")
        assert result.passed is True

    def test_missing_consent_for_cloud(self):
        result = self.engine.evaluate(
            "Hello", tenant_id="t1", user_id="u1",
            require_cloud=True,
        )
        assert result.can_proceed_cloud is False


class TestDeploymentTypeImport:
    def test_deployment_type_imported(self):
        from zilli.models.config import DeploymentType
        assert DeploymentType.LOCAL.value == "local"
        assert DeploymentType.CLOUD.value == "cloud"
