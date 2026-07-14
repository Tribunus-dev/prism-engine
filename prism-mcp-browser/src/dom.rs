use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct DomRevision(pub u64);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DomNodeId {
    pub tab: String,
    pub revision: DomRevision,
    pub ordinal: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomNode {
    pub id: DomNodeId,
    pub tag: String,
    pub role: Option<String>,
    pub name: String,
    pub text: String,
    pub selector: String,
    pub visible: bool,
    pub enabled: bool,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomDocument {
    pub revision: DomRevision,
    pub url: String,
    pub title: String,
    pub text: String,
    pub nodes: Vec<DomNode>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct DomQuery {
    pub css: Option<String>,
    pub role: Option<String>,
    pub name: Option<String>,
    pub text: Option<String>,
    pub visible: Option<bool>,
    pub enabled: Option<bool>,
}

impl DomQuery {
    pub fn matches(&self, node: &DomNode) -> bool {
        self.role
            .as_ref()
            .is_none_or(|v| node.role.as_deref() == Some(v))
            && self.name.as_ref().is_none_or(|v| node.name.contains(v))
            && self.text.as_ref().is_none_or(|v| node.text.contains(v))
            && self.visible.is_none_or(|v| node.visible == v)
            && self.enabled.is_none_or(|v| node.enabled == v)
    }
}

pub fn node_from_value(value: &Value, tab: String, revision: DomRevision, ordinal: u32) -> DomNode {
    DomNode {
        id: DomNodeId {
            tab,
            revision,
            ordinal,
        },
        tag: value["tag"].as_str().unwrap_or_default().to_owned(),
        role: value["role"].as_str().map(str::to_owned),
        name: value["name"].as_str().unwrap_or_default().to_owned(),
        text: value["text"].as_str().unwrap_or_default().to_owned(),
        selector: value["selector"].as_str().unwrap_or_default().to_owned(),
        visible: value["visible"].as_bool().unwrap_or(false),
        enabled: value["enabled"].as_bool().unwrap_or(false),
        x: value["x"].as_f64().unwrap_or_default(),
        y: value["y"].as_f64().unwrap_or_default(),
        width: value["width"].as_f64().unwrap_or_default(),
        height: value["height"].as_f64().unwrap_or_default(),
    }
}
