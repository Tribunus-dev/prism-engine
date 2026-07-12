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

        let results = state.db.with_reader(|conn| {
            // FTS5 query — wrap each term in quotes for AND matching
            let fts_query = query
                .split_whitespace()
                .map(|w| format!("\"{}\"", w))
                .collect::<Vec<_>>()
                .join(" AND ");

            let sql = "SELECT s.id, s.document_id, s.heading, s.word_count,
                              d.title, d.doc_type,
                              snippet(sections_fts, 1, '<mark>', '</mark>', '...', 32),
                              rank
                       FROM sections_fts
                       JOIN sections s ON sections_fts.rowid = s.rowid
                       JOIN documents d ON s.document_id = d.id
                       WHERE sections_fts MATCH ?1
                       ORDER BY rank
                       LIMIT ?2";

            let mut stmt = conn.prepare(sql)?;
            let results = stmt.query_map(rusqlite::params![fts_query, limit as i64], |row| {
                Ok(SearchResultRow {
                    section_id: row.get::<_, String>(0)?,
                    document_id: row.get::<_, String>(1)?,
                    heading: row.get::<_, String>(2)?,
                    word_count: row.get::<_, i64>(3)?,
                    doc_title: row.get::<_, String>(4)?,
                    doc_type: row.get::<_, String>(5)?,
                    snippet: row.get::<_, String>(6)?,
                    rank: row.get::<_, f64>(7)?,
                })
            })?;

            let mut rows = Vec::new();
            for r in results {
                rows.push(r?);
            }
            Ok(rows)
        })?;

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

        state.db.with_reader(|conn| {
            let mut stmt = conn.prepare(
                "SELECT id, title, doc_type, content_md, version, status, created_at FROM documents WHERE id = ?1"
            )?;
            let mut rows = stmt.query(rusqlite::params![id])?;
            match rows.next()? {
                Some(row) => {
                    let title: String = row.get(1)?;
                    let doc_type: String = row.get(2)?;
                    let content: String = row.get(3)?;
                    let version: i64 = row.get(4)?;
                    let status: String = row.get(5)?;
                    let created: String = row.get(6)?;
                    Ok(ToolResult::text(format!(
                        "# {} ({})\n\n**Type:** {} | **Status:** {} | **Version:** {} | **Created:** {}\n\n---\n{}",
                        title, id, doc_type, status, version, created, content
                    )))
                }
                None => Err(anyhow::anyhow!("Document not found: {}", id)),
            }
        })
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

        state.db.with_reader(|conn| {
            let (sql, params): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = if let Some(dt) = doc_type {
                ("SELECT id, title, doc_type, version, status, updated_at FROM documents WHERE doc_type = ?1 ORDER BY updated_at DESC LIMIT ?2".into(),
                 vec![Box::new(dt.to_string()), Box::new(limit)])
            } else {
                ("SELECT id, title, doc_type, version, status, updated_at FROM documents ORDER BY updated_at DESC LIMIT ?1".into(),
                 vec![Box::new(limit)])
            };

            let mut stmt = conn.prepare(&sql)?;
            let rows = stmt.query_map(rusqlite::params_from_iter(params.iter()), |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                ))
            })?;

            let mut output = String::from("Documents:\n");
            for r in rows {
                let (id, title, dt, ver, status, updated) = r?;
                output.push_str(&format!("  [{}] {} ({}) v{} {} — {}\n", dt, title, id, ver, status, updated));
            }
            Ok(ToolResult::text(output))
        })
    }
}
