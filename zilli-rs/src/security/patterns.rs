pub const EMAIL: &str = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}";
pub const PHONE: &str = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b";
pub const SSN: &str = r"\b\d{3}-\d{2}-\d{4}\b";
pub const CREDIT_CARD: &str = r"\b(?:\d{4}[-\s]?){3}\d{4}\b";
pub const CHINESE_ID: &str = r"\b[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b";
pub const API_KEY: &str = r#"(?i)(?:api[_-]?key|token|secret)\s*[:=]\s*['"]?[A-Za-z0-9_\-]{16,}['"]?"#;
pub const ADDRESS: &str = r"\b\d{1,5}\s+[A-Za-z0-9,]+(?:\s+[A-Za-z0-9,]+)*(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b";
pub const IP: &str = r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b";
pub const MEDICAL_RECORD: &str = r"(?i)\b(?:patient|diagnosis|procedure|treatment)\s*(?:id|number|code)?\s*:\s*\w+\b";

pub fn mask_pii(value: &str) -> String {
    if value.len() <= 4 {
        "****".to_string()
    } else {
        let prefix = &value[..2];
        let masked = "*".repeat(value.len() - 4);
        let suffix = &value[value.len() - 2..];
        format!("{}{}{}", prefix, masked, suffix)
    }
}
