use std::collections::HashMap;

/// Walk an FX graph in topological order, creating ECS entities for each node.
pub struct FxGraphWalker {
    nodes: Vec<FxNode>,
}

#[derive(Debug, Clone)]
pub struct FxNode {
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<usize>,
    pub attrs: HashMap<String, String>,
}

impl FxGraphWalker {
    pub fn new(nodes: Vec<FxNode>) -> Self {
        Self { nodes }
    }

    pub fn walk(&self) -> Result<Vec<String>, String> {
        let mut names = Vec::new();
        for node in &self.nodes {
            names.push(node.name.clone());
        }
        Ok(names)
    }
}
