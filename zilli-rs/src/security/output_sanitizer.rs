use regex::Regex;

#[derive(Debug)]
pub struct OutputSanitizer {
    harmful_patterns: Vec<(String, Regex)>,
}

impl OutputSanitizer {
    pub fn new() -> Self {
        Self {
            harmful_patterns: vec![
                ("harmful_code".into(), Regex::new(r"(?i)(?:rm\s+-rf\s+(?:/|\*|--no-preserve-root\s+/)|format\s+C:|shutdown\s+-h\s+now|del\s+/[fs]\s+/)").unwrap()),
                ("personal_info_leak".into(), Regex::new(r"(?i)(?:social\s+security|passport\s+number|driver'?s?\s+license)").unwrap()),
                ("bias".into(), Regex::new(r"(?i)\b(?:\ball\s+)?[Xx]\s+(?:are|is)\s+(?:lazy|stupid|inferior|evil)\b").unwrap()),
                ("dangerous_instruction".into(), Regex::new(r"(?i)how\s+to\s+(?:make|build|create)\s+(?:bomb|weapon|drug|poison)").unwrap()),
            ],
        }
    }

    pub fn check(&self, text: &str) -> OutputVerdict {
        let mut warnings = Vec::new();

        for (category, pattern) in &self.harmful_patterns {
            if pattern.is_match(text) {
                warnings.push(category.clone());
            }
        }

        let passed = warnings.is_empty();
        let sanitized = if passed {
            None
        } else {
            Some(self.sanitize(text))
        };
        OutputVerdict { passed, warnings, sanitized }
    }

    pub fn sanitize(&self, text: &str) -> String {
        let mut result = text.to_string();
        for (_category, pattern) in &self.harmful_patterns {
            result = pattern.replace_all(&result, "[REDACTED]").to_string();
        }
        result
    }
}

impl Default for OutputSanitizer {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct OutputVerdict {
    pub passed: bool,
    pub warnings: Vec<String>,
    pub sanitized: Option<String>,
}
