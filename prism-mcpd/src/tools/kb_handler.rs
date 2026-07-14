use prism_mcp_core::{DaemonState, McpHandler, RequestContext, ToolRequest, ToolResult};

pub struct SearchKbHandler;

impl SearchKbHandler {
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self)
    }
}

impl McpHandler for SearchKbHandler {
    fn name(&self) -> &'static str {
        "search_kb"
    }

    fn description(&self) -> &'static str {
        "Full-text search across documents and sections. Returns ranked results."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": { "type": "string" },
                "limit": { "type": "integer", "default": 10 }
            },
            "required": ["query"]
        })
    }

    fn call(
        &self,
        request: ToolRequest<'_>,
        _context: &RequestContext,
        state: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        let query = request.args["query"].as_str().unwrap_or("");
        if query.is_empty() {
            return Err(anyhow::anyhow!("query is required"));
        }
        let limit = request.args["limit"].as_i64().unwrap_or(10) as usize;

        let results = state.knowledge_store.search(query, limit)?;

        if results.is_empty() {
            return Ok(ToolResult::text(format!("No results found for: {}", query)));
        }

        let mut output = format!("Found {} result(s) for '{}':\n", results.len(), query);
        for r in &results {
            output.push_str(&format!(
                "\n[{}] {} > {}\n{}\n---\n",
                r.doc_type, r.doc_title, r.heading, r.snippet
            ));
        }
        Ok(ToolResult::text(output))
    }
}

#[allow(dead_code)]
pub struct SearchResultRow {
    section_id: String,
    document_id: String,
    heading: String,
    word_count: i64,
    doc_title: String,
    doc_type: String,
    snippet: String,
    rank: f64,
}

// ── Additional KB tools ──────────────────────────────────────────────────

pub struct GetDocumentHandler;

impl McpHandler for GetDocumentHandler {
    fn name(&self) -> &'static str {
        "get_document"
    }
    fn description(&self) -> &'static str {
        "Retrieve a full document by ID."
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object", "properties": { "id": { "type": "string" } }, "required": ["id"] })
    }
    fn call(
        &self,
        request: ToolRequest<'_>,
        _ctx: &RequestContext,
        state: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        let id = request.args["id"].as_str().unwrap_or("");
        if id.is_empty() {
            return Err(anyhow::anyhow!("id is required"));
        }

        let document = state
            .knowledge_store
            .get_document(id)?
            .ok_or_else(|| anyhow::anyhow!("Document not found: {id}"))?;
        Ok(ToolResult::text(format!("# {} ({})\n\n**Type:** {} | **Status:** {} | **Version:** {} | **Created:** {}\n\n---\n{}", document.title, document.id, document.doc_type, document.status, document.version, document.created_at, document.content)))
    }
}

pub struct ListDocumentsHandler;

impl McpHandler for ListDocumentsHandler {
    fn name(&self) -> &'static str {
        "list_documents"
    }
    fn description(&self) -> &'static str {
        "List all documents with optional type filter."
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "doc_type": { "type": "string" },
                "limit": { "type": "integer", "default": 50 }
            }
        })
    }
    fn call(
        &self,
        request: ToolRequest<'_>,
        _ctx: &RequestContext,
        state: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        let doc_type =
            request.args["doc_type"]
                .as_str()
                .and_then(|s| if s.is_empty() { None } else { Some(s) });
        let limit = request.args["limit"].as_i64().unwrap_or(50) as i64;

        let rows = state
            .knowledge_store
            .list_documents(doc_type, limit as usize)?;
        let mut output = String::from("Documents:\n");
        for row in rows {
            output.push_str(&format!(
                "  [{}] {} ({}) v{} {} — {}\n",
                row.doc_type, row.title, row.id, row.version, row.status, row.updated_at
            ));
        }
        Ok(ToolResult::text(output))
    }
}
