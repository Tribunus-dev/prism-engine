use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchHit {
    pub id: String,
    pub score: f32,
}
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct SearchIndex {
    entries: Vec<(String, String)>,
}
impl SearchIndex {
    pub fn insert(&mut self, id: impl Into<String>, text: impl Into<String>) {
        self.entries.push((id.into(), text.into()))
    }
    pub fn search(&self, q: &str) -> Vec<SearchHit> {
        let q = q.to_lowercase();
        self.entries
            .iter()
            .filter(|(_, t)| t.to_lowercase().contains(&q))
            .map(|(id, _)| SearchHit {
                id: id.clone(),
                score: 1.0,
            })
            .collect()
    }
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}
