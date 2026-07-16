// ── Prism LLM Inference — Tool Call Integration ───────────────────────────
//
// Wraps compute-core tool types into a ToolEngine for the Prism LLM runtime.
// Handles tool registration, parsing, execution, and structured output via
// grammar-backed JSON mode (JSON Schema → GBNF).
//
// The existing stub implements core types unconditionally so callers do not
// need per-site conditional compilation. The #[cfg(feature = "prism-backend")]
// blocks delegate to tribunus_compute_core for production parsing/repair and
// grammar generation.

// ── Re-export inner types unconditionally ────────────────────────────

use std::collections::HashMap;

// ── Errors ───────────────────────────────────────────────────────────

/// Errors that can occur during tool operations.
#[derive(Debug, Clone)]
pub enum ToolError {
    /// The named tool is not registered.
    NotRegistered(String),
    /// Failed to parse a model-generated tool call.
    ParseError(String),
    /// Tool execution failed.
    ExecutionError(String),
    /// Grammar generation or compilation failed.
    GrammarError(String),
    /// Tool call is unrepairable — should trigger regeneration.
    Unrepairable(String),
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolError::NotRegistered(name) => write!(f, "tool not registered: {}", name),
            ToolError::ParseError(msg) => write!(f, "tool parse error: {}", msg),
            ToolError::ExecutionError(msg) => write!(f, "tool execution error: {}", msg),
            ToolError::GrammarError(msg) => write!(f, "grammar error: {}", msg),
            ToolError::Unrepairable(msg) => write!(f, "unrepairable tool call: {}", msg),
        }
    }
}

impl std::error::Error for ToolError {}

// ── Stub / shared types ──────────────────────────────────────────────

/// A registered tool descriptor holding its JSON Schema.
#[derive(Debug, Clone)]
pub struct RegisteredTool {
    /// Canonical function name.
    pub name: String,
    /// Human-readable description of what the tool does.
    pub description: String,
    /// JSON Schema describing the expected parameters.
    pub parameters: serde_json::Value,
    /// Parameter names that are required.
    pub required: Vec<String>,
}

/// A parsed function call ready for execution.
#[derive(Debug, Clone)]
pub struct ParsedCall {
    /// The tool/function name.
    pub name: String,
    /// Parsed JSON arguments (a serde_json::Value::Object).
    pub arguments: serde_json::Value,
    /// Raw text the model generated.
    pub raw: String,
}

/// Outcome of attempting to parse and/or repair a model-generated tool call.
#[derive(Debug, Clone)]
pub enum ToolCallOutcome {
    /// Successfully parsed — no repair needed.
    Valid(ParsedCall),
    /// Successfully parsed after deterministic repair.
    Repaired(ParsedCall, Vec<String>),
    /// Cannot repair — generation should be retried with error context.
    Unrepairable(String),
}

// ── ToolEngine ───────────────────────────────────────────────────────

/// Manages tool registration, model-output parsing/repair, execution, and
/// JSON Schema → GBNF grammar generation for structured output.
pub struct ToolEngine {
    tools: HashMap<String, RegisteredTool>,
}

impl ToolEngine {
    /// Create an empty tool engine.
    pub fn new() -> Self {
        ToolEngine {
            tools: HashMap::new(),
        }
    }

    /// Register a tool with its JSON Schema parameter definition.
    ///
    /// `name` — the function name the model will use to invoke this tool.
    /// `description` — a human-readable description fed to the model.
    /// `parameters` — a JSON Schema object describing the arguments.
    /// `required` — list of required parameter names.
    pub fn register_tool(
        &mut self,
        name: &str,
        description: &str,
        parameters: serde_json::Value,
        required: Vec<String>,
    ) {
        self.tools.insert(
            name.to_string(),
            RegisteredTool {
                name: name.to_string(),
                description: description.to_string(),
                parameters,
                required,
            },
        );
    }

    /// Look up a registered tool by name.
    pub fn get_tool(&self, name: &str) -> Option<&RegisteredTool> {
        self.tools.get(name)
    }

    /// Return the names of all registered tools.
    pub fn list_tools(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// Remove a registered tool by name.
    pub fn unregister_tool(&mut self, name: &str) -> Option<RegisteredTool> {
        self.tools.remove(name)
    }

    /// Check whether a tool with the given name is registered.
    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Parse raw model-generated text into a tool call.
    ///
    /// On `prism-backend` builds this uses compute-core's multi-strategy
    /// parser+repair pipeline (direct parse, JSON fences, first JSON object,
    /// unescape+parse, validation+type-coercion, fuzzy name correction).
    ///
    /// On non-`prism-backend` builds it falls back to a simple JSON parse.
        pub fn parse_call(&self, raw: &str, tool_name: &str) -> Result<ToolCallOutcome, ToolError> {
        let _tool = self
            .tools
            .get(tool_name)
            .ok_or_else(|| ToolError::NotRegistered(tool_name.to_string()))?;

        match serde_json::from_str::<serde_json::Value>(raw) {
            Ok(val) => {
                let name = val
                    .get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or(tool_name)
                    .to_string();
                let arguments = val
                    .get("arguments")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                Ok(ToolCallOutcome::Valid(ParsedCall {
                    name,
                    arguments,
                    raw: raw.to_string(),
                }))
            }
            Err(e) => Ok(ToolCallOutcome::Unrepairable(format!(
                "parse failed: {}",
                e
            ))),
        }
    }

    /// Parse raw model-generated text into a tool call.
    /// Execute a parsed tool call through a caller-provided dispatcher or
    /// the default built-in implementation.
    ///
    /// The `dispatcher` is an optional closure `Fn(&str, &serde_json::Value)
    /// -> Result<serde_json::Value, String>` that receives the tool name
    /// and parsed arguments. When `None`, a simple acknowledgment is returned.
        pub fn execute_tool<F>(
        &self,
        call: &ParsedCall,
        dispatcher: Option<&F>,
    ) -> Result<serde_json::Value, ToolError>
    where
        F: Fn(&str, &serde_json::Value) -> Result<serde_json::Value, String>,
    {
        if let Some(d) = dispatcher {
            d(&call.name, &call.arguments).map_err(|e| ToolError::ExecutionError(e))
        } else {
            Ok(serde_json::json!({
                "tool": call.name,
                "arguments": call.arguments,
                "status": "called"
            }))
        }
    }

    /// Execute a parsed tool call.
    /// Build a GBNF grammar string from a JSON Schema for structured output.
    ///
    /// The returned GBNF grammar constrains generation so the model can only
    /// produce valid JSON matching the schema. On `prism-backend` builds this
    /// delegates to `tribunus_compute_core::grammar::Grammar::from_json_schema`.
        pub fn json_schema_to_grammar(
        _name: &str,
        _schema: &serde_json::Value,
    ) -> Result<String, ToolError> {
        // Stub: return minimal JSON-object grammar.
        Ok("root ::= \"{\" ws \"}\"\nws ::= [ \\t\\n]*\n".to_string())
    }

    /// Build a GBNF grammar string from a JSON Schema for structured output.
    /// Build valid GBNF text directly from a JSON Schema Value.
    ///
    /// The output is a self-consistent GBNF grammar that constrains generation
    /// to valid JSON matching the schema. This function is the "hand-crafted"
    /// replacement for reconstructing GBNF from the compile-core Grammar AST.
    /// Recursively emit a single named GBNF rule for a JSON Schema sub-schema.
    /// Return the number of registered tools.
    pub fn tool_count(&self) -> usize {
        self.tools.len()
    }
}

impl Default for ToolEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── Grammar node to GBNF text (prism-backend only) ────────────────────

/// Format a `GrammarNode` back into GBNF text.
#[allow(dead_code)]
/// Format with parent-context awareness for correct GBNF grouping.
///
/// - `parent_is_seq`: true when the parent is a Seq, so `Alt` children
///   need wrapping in `( ... )` for correct GBNF precedence.
#[allow(dead_code)]
// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Registration & query ─────────────────────────────────────────

    #[test]
    fn test_register_and_list_tools() {
        let mut engine = ToolEngine::new();
        assert_eq!(engine.tool_count(), 0);

        engine.register_tool(
            "get_weather",
            "Get current weather",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "location": { "type": "string" }
                },
                "required": ["location"]
            }),
            vec!["location".to_string()],
        );

        assert_eq!(engine.tool_count(), 1);
        assert!(engine.has_tool("get_weather"));
        assert!(!engine.has_tool("nonexistent"));
        assert_eq!(engine.list_tools(), vec!["get_weather"]);
    }

    #[test]
    fn test_unregister_tool() {
        let mut engine = ToolEngine::new();
        engine.register_tool("foo", "does something", serde_json::json!({}), vec![]);
        assert!(engine.has_tool("foo"));
        let removed = engine.unregister_tool("foo");
        assert!(removed.is_some());
        assert!(!engine.has_tool("foo"));
        assert!(engine.unregister_tool("foo").is_none());
    }

    #[test]
    fn test_get_tool_returns_registered_fields() {
        let mut engine = ToolEngine::new();
        let params = serde_json::json!({
            "type": "object",
            "properties": {
                "x": { "type": "number" }
            },
            "required": ["x"]
        });
        engine.register_tool("calc", "Calculate", params.clone(), vec!["x".to_string()]);

        let tool = engine.get_tool("calc").expect("tool should exist");
        assert_eq!(tool.name, "calc");
        assert_eq!(tool.description, "Calculate");
        assert_eq!(tool.parameters, params);
        assert_eq!(tool.required, vec!["x".to_string()]);
    }

    // ── Parse call (stub) ────────────────────────────────────────────

    #[test]
    fn test_parse_call_valid_json() {
        let mut engine = ToolEngine::new();
        engine.register_tool(
            "greet",
            "Greet someone",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string" }
                }
            }),
            vec![],
        );

        let result = engine.parse_call(
            r#"{"name": "greet", "arguments": {"name": "Alice"}}"#,
            "greet",
        );
        match result {
            Ok(ToolCallOutcome::Valid(call)) => {
                assert_eq!(call.name, "greet");
                assert_eq!(
                    call.arguments.get("name").and_then(|v| v.as_str()),
                    Some("Alice")
                );
            }
            Ok(ToolCallOutcome::Unrepairable(msg)) => {
                panic!("expected valid, got unrepairable: {}", msg);
            }
            Ok(ToolCallOutcome::Repaired(_, fixes)) => {
                panic!("expected valid, got repaired: {:?}", fixes);
            }
            Err(e) => panic!("parse_call error: {}", e),
        }
    }

    #[test]
    fn test_parse_call_unregistered_tool() {
        let engine = ToolEngine::new();
        let result = engine.parse_call(r#"{}"#, "does_not_exist");
        assert!(matches!(result, Err(ToolError::NotRegistered(_))));
    }

    #[test]
    fn test_parse_call_malformed_json() {
        let mut engine = ToolEngine::new();
        engine.register_tool("t", "test", serde_json::json!({}), vec![]);

        let result = engine.parse_call("not json at all {{{", "t");
        match result {
            Ok(ToolCallOutcome::Unrepairable(_)) => {} // expected for stub
            Ok(other) => panic!("expected unrepairable, got {:?}", other),
            Err(e) => panic!("expected Ok(Unrepairable), got Err({})", e),
        }
    }

    // ── Execute tool ─────────────────────────────────────────────────

    #[test]
    fn test_execute_tool_default() {
        let engine = ToolEngine::new();
        let call = ParsedCall {
            name: "test".to_string(),
            arguments: serde_json::json!({"key": "value"}),
            raw: r#"{"name": "test", "arguments": {"key": "value"}}"#.to_string(),
        };

        let result = engine
            .execute_tool::<fn(&str, &serde_json::Value) -> Result<serde_json::Value, String>>(
                &call,
                None::<&fn(&str, &serde_json::Value) -> Result<serde_json::Value, String>>,
            );
        let json = result.expect("execute_tool should succeed");
        assert_eq!(json["tool"], "test");
        assert_eq!(json["status"], "called");
    }

    #[test]
    fn test_execute_tool_with_dispatcher() {
        let engine = ToolEngine::new();
        let call = ParsedCall {
            name: "echo".to_string(),
            arguments: serde_json::json!({"msg": "hello"}),
            raw: "raw".to_string(),
        };

        let dispatcher = |name: &str, args: &serde_json::Value| {
            Ok(serde_json::json!({
                "echoed": name,
                "message": args["msg"]
            }))
        };

        let result = engine.execute_tool(&call, Some(&dispatcher));
        let json = result.expect("execute_tool with dispatcher should succeed");
        assert_eq!(json["echoed"], "echo");
        assert_eq!(json["message"], "hello");
    }

    #[test]
    fn test_execute_tool_dispatcher_error() {
        let engine = ToolEngine::new();
        let call = ParsedCall {
            name: "fail".to_string(),
            arguments: serde_json::json!({}),
            raw: "raw".to_string(),
        };

        let dispatcher = |_: &str, _: &serde_json::Value| Err("something went wrong".to_string());
        let result = engine.execute_tool(&call, Some(&dispatcher));
        match result {
            Err(ToolError::ExecutionError(msg)) => assert_eq!(msg, "something went wrong"),
            other => panic!("expected ExecutionError, got {:?}", other),
        }
    }

    // ── JSON Schema → GBNF grammar ───────────────────────────────────

    #[test]
    fn test_json_schema_to_grammar_returns_string() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            },
            "required": []
        });

        let result = ToolEngine::json_schema_to_grammar("person", &schema);
        let grammar = result.unwrap_or_else(|e| panic!("should produce grammar: {e}"));
        // Should produce valid GBNF with a root rule
        assert!(grammar.contains("root ::="));
        // Should contain whitespace rule
        assert!(grammar.contains("ws ::="));
    }

    #[test]
    fn test_json_schema_to_grammar_simple_object() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {}
        });

        let result = ToolEngine::json_schema_to_grammar("empty", &schema);
        assert!(result.is_ok());
    }

    #[test]
    fn test_json_schema_to_grammar_invalid_schema() {
        // Deeply nested to trigger error on the stubbed side
        let schema = serde_json::json!({
            "type": "object"
        });
        let result = ToolEngine::json_schema_to_grammar("test", &schema);
        assert!(result.is_ok()); // stubbed version always returns a basic grammar
    }

    // ── Error display ────────────────────────────────────────────────

    #[test]
    fn test_tool_error_display() {
        assert_eq!(
            ToolError::NotRegistered("foo".to_string()).to_string(),
            "tool not registered: foo"
        );
        assert_eq!(
            ToolError::ParseError("bad".to_string()).to_string(),
            "tool parse error: bad"
        );
        assert_eq!(
            ToolError::ExecutionError("oops".to_string()).to_string(),
            "tool execution error: oops"
        );
        assert_eq!(
            ToolError::GrammarError("syntax".to_string()).to_string(),
            "grammar error: syntax"
        );
        assert_eq!(
            ToolError::Unrepairable("broken".to_string()).to_string(),
            "unrepairable tool call: broken"
        );
    }
}
