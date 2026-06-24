use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use strum::{Display, EnumString};

#[derive(Debug, Clone, Serialize, Deserialize, Display, EnumString, PartialEq, Eq)]
pub enum RouteType {
    FullRoute,
    FastLane,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteDecision {
    pub route: RouteType,
    pub reason: String,
}

pub struct RouteClassifier {
    fast_lane_patterns: Vec<String>,
    industry_routes: HashMap<String, RouteType>,
}

impl RouteClassifier {
    pub fn new() -> Self {
        let mut industry_routes = HashMap::new();
        industry_routes.insert("education".into(), RouteType::FastLane);
        industry_routes.insert("legal".into(), RouteType::FullRoute);
        industry_routes.insert("medical".into(), RouteType::FullRoute);
        industry_routes.insert("financial".into(), RouteType::FullRoute);

        Self {
            fast_lane_patterns: vec![
                "list".into(),
                "get".into(),
                "simple".into(),
                "format".into(),
                "translate".into(),
            ],
            industry_routes,
        }
    }

    pub fn classify(&self, request: &str, industry: &str) -> RouteDecision {
        if !industry.is_empty() {
            if let Some(route) = self.industry_routes.get(industry) {
                return RouteDecision {
                    route: route.clone(),
                    reason: format!("Industry '{}' maps to {:?}", industry, route),
                };
            }
        }

        let request_lower = request.to_lowercase();
        for pattern in &self.fast_lane_patterns {
            if request_lower.contains(pattern) {
                return RouteDecision {
                    route: RouteType::FastLane,
                    reason: format!("Fast lane pattern matched: '{}'", pattern),
                };
            }
        }

        if request.len() > 500 {
            return RouteDecision {
                route: RouteType::FullRoute,
                reason: "Long request requires full route".into(),
            };
        }

        RouteDecision {
            route: RouteType::FastLane,
            reason: "Default: fast lane for short request".into(),
        }
    }
}

impl Default for RouteClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_lane_default() {
        let c = RouteClassifier::new();
        let d = c.classify("hello world", "");
        assert_eq!(d.route, RouteType::FastLane);
    }

    #[test]
    fn test_full_route_long_request() {
        let c = RouteClassifier::new();
        let long = "a".repeat(501);
        let d = c.classify(&long, "");
        assert_eq!(d.route, RouteType::FullRoute);
    }

    #[test]
    fn test_industry_legal_full_route() {
        let c = RouteClassifier::new();
        let d = c.classify("anything", "legal");
        assert_eq!(d.route, RouteType::FullRoute);
    }

    #[test]
    fn test_industry_education_fast_lane() {
        let c = RouteClassifier::new();
        let d = c.classify("anything", "education");
        assert_eq!(d.route, RouteType::FastLane);
    }
}
