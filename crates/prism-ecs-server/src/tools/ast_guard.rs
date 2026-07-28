//! AST-level security guard for JavaScript code run in the V8 sandbox
//! (constitutional home).
//!
//! Parses the JS source with swc and rejects code containing dangerous
//! dynamic execution patterns like `eval()`, `new Function()`, or
//! string-argument `setTimeout`/`setInterval`.
//!
//! # Authority boundary
//!
//! This is a static-analysis gate. It runs on the JS source before
//! `js_runtime::run_javascript` boots a V8 isolate. It does not mutate
//! ECS world state; it is a precondition for the execution effect.

use swc_common::sync::Lrc;
use swc_common::{FileName, SourceMap};
use swc_ecma_ast::*;
use swc_ecma_parser::lexer::Lexer;
use swc_ecma_parser::{Parser, StringInput, Syntax};

/// Validate a JavaScript snippet before V8 execution.
///
/// Returns `Ok(())` if the code passes all security checks, or `Err(reason)`
/// with a description of the violation.
pub fn validate_agent_script(js_code: &str) -> Result<(), String> {
    let cm: Lrc<SourceMap> = Default::default();
    let fm = cm.new_source_file(
        Lrc::new(FileName::Custom("sandbox.js".into())),
        js_code.to_owned(),
    );
    let lexer = Lexer::new(
        Syntax::Es(Default::default()),
        Default::default(),
        StringInput::from(&*fm),
        None,
    );
    let mut parser = Parser::new_from(lexer);
    let module = parser
        .parse_module()
        .map_err(|e| format!("Failed to parse JS AST: {e:?}"))?;

    let mut visitor = SecurityVisitor {
        has_violation: false,
        reason: String::new(),
    };
    visitor.visit_module(&module);

    if visitor.has_violation {
        Err(visitor.reason)
    } else {
        Ok(())
    }
}

// ── AST Visitor ───────────────────────────────────────────────────────

struct SecurityVisitor {
    has_violation: bool,
    reason: String,
}

impl SecurityVisitor {
    fn reject(&mut self, msg: String) {
        if !self.has_violation {
            self.has_violation = true;
            self.reason = format!("SECURITY BLOCKED: {msg}");
        }
    }

    /// Resolve a callee expression to an identifier name, walking through
    /// indirect-eval wrappers: `(0, eval)`, `window.eval`, `window["eval"]`,
    /// `globalThis.eval`. Does not resolve dynamically computed property names
    /// (e.g., `window[atob("ZXZhbA==")]`) — that requires runtime analysis.
    fn callee_name(expr: &Expr) -> Option<&str> {
        match expr {
            Expr::Ident(ident) => Some(ident.sym.as_str()),
            Expr::Paren(paren) => Self::callee_name(&paren.expr),
            Expr::Seq(seq) => seq.exprs.last().and_then(|e| Self::callee_name(e)),
            Expr::Member(member) => match &member.prop {
                MemberProp::Ident(ident) => Some(ident.sym.as_str()),
                MemberProp::Computed(computed) => Self::callee_name(&computed.expr),
                _ => None,
            },
            Expr::Lit(Lit::Str(s)) => Some(s.value.as_str().unwrap_or_default()),
            _ => None,
        }
    }

    fn visit_expr(&mut self, expr: &Expr) {
        if self.has_violation {
            return;
        }
        match expr {
            // 1. Block eval() calls
            Expr::Call(call_expr) => {
                if let Callee::Expr(callee) = &call_expr.callee {
                    if Self::callee_name(callee) == Some("eval") {
                        return self.reject("eval() is not allowed".into());
                    }
                }
            }
            // 2. Block new Function(...)
            Expr::New(new_expr) => {
                if Self::callee_name(&new_expr.callee) == Some("Function") {
                    return self.reject("new Function() is not allowed".into());
                }
            }
            _ => {}
        }
        // Recurse into children
        match expr {
            Expr::Call(call) => {
                // Check setTimeout/setInterval with string literal first arg
                if let Callee::Expr(callee) = &call.callee {
                    if let Some(name) = Self::callee_name(callee) {
                        if matches!(name, "setTimeout" | "setInterval") {
                            if let Some(first_arg) = call.args.first() {
                                if matches!(&*first_arg.expr, Expr::Lit(Lit::Str(_))) {
                                    return self.reject(format!(
                                        "{}() with string argument is not allowed",
                                        name
                                    ));
                                }
                            }
                        }
                    }
                }
                for arg in &call.args {
                    self.visit_expr(&arg.expr);
                }
            }
            Expr::New(new_expr) => {
                self.visit_expr(&new_expr.callee);
                if let Some(args) = &new_expr.args {
                    for arg in args {
                        self.visit_expr(&arg.expr);
                    }
                }
            }
            Expr::Bin(bin) => {
                self.visit_expr(&bin.left);
                self.visit_expr(&bin.right);
            }
            Expr::Unary(unary) => {
                self.visit_expr(&unary.arg);
            }
            Expr::Assign(assign) => {
                self.visit_expr(&assign.right);
            }
            Expr::Member(member) => {
                if let MemberProp::Computed(computed) = &member.prop {
                    self.visit_expr(&computed.expr);
                }
            }
            Expr::Cond(cond) => {
                self.visit_expr(&cond.test);
                self.visit_expr(&cond.cons);
                self.visit_expr(&cond.alt);
            }
            Expr::Seq(seq) => {
                for e in &seq.exprs {
                    self.visit_expr(e);
                }
            }
            Expr::Paren(paren) => {
                self.visit_expr(&paren.expr);
            }
            Expr::Tpl(tpl) => {
                for e in &tpl.exprs {
                    self.visit_expr(e);
                }
            }
            Expr::Await(await_expr) => {
                self.visit_expr(&await_expr.arg);
            }
            Expr::Yield(yield_expr) => {
                if let Some(arg) = &yield_expr.arg {
                    self.visit_expr(arg);
                }
            }
            Expr::Array(arr) => {
                for elem in &arr.elems {
                    if let Some(e) = elem {
                        self.visit_expr(&e.expr);
                    }
                }
            }
            Expr::Object(obj) => {
                for prop in &obj.props {
                    if let PropOrSpread::Prop(boxed) = prop {
                        if let Prop::KeyValue(kv) = boxed.as_ref() {
                            self.visit_expr(&kv.value);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn visit_stmt(&mut self, stmt: &Stmt) {
        if self.has_violation {
            return;
        }
        match stmt {
            Stmt::Expr(expr_stmt) => self.visit_expr(&expr_stmt.expr),
            Stmt::Decl(decl) => match decl {
                Decl::Var(var_decl) => {
                    for declarator in &var_decl.decls {
                        if let Some(init) = &declarator.init {
                            self.visit_expr(init);
                        }
                    }
                }
                Decl::Fn(fn_decl) => {
                    if let Some(body) = &fn_decl.function.body {
                        for s in &body.stmts {
                            self.visit_stmt(s);
                        }
                    }
                }
                _ => {}
            },
            Stmt::Return(return_stmt) => {
                if let Some(arg) = &return_stmt.arg {
                    self.visit_expr(arg);
                }
            }
            Stmt::If(if_stmt) => {
                self.visit_expr(&if_stmt.test);
                self.visit_stmt(&if_stmt.cons);
                if let Some(alt) = &if_stmt.alt {
                    self.visit_stmt(alt);
                }
            }
            Stmt::Block(block) => {
                for s in &block.stmts {
                    self.visit_stmt(s);
                }
            }
            Stmt::For(for_stmt) => {
                if let Some(init) = &for_stmt.init {
                    match init {
                        VarDeclOrExpr::VarDecl(var) => {
                            for decl in &var.decls {
                                if let Some(init) = &decl.init {
                                    self.visit_expr(init);
                                }
                            }
                        }
                        VarDeclOrExpr::Expr(e) => self.visit_expr(e),
                    }
                }
                if let Some(test) = &for_stmt.test {
                    self.visit_expr(test);
                }
                if let Some(update) = &for_stmt.update {
                    self.visit_expr(update);
                }
                self.visit_stmt(&for_stmt.body);
            }
            Stmt::While(while_stmt) => {
                self.visit_expr(&while_stmt.test);
                self.visit_stmt(&while_stmt.body);
            }
            Stmt::Try(try_stmt) => {
                for s in &try_stmt.block.stmts {
                    self.visit_stmt(s);
                }
                if let Some(handler) = &try_stmt.handler {
                    for s in &handler.body.stmts {
                        self.visit_stmt(s);
                    }
                }
            }
            Stmt::Switch(switch_stmt) => {
                self.visit_expr(&switch_stmt.discriminant);
                for case in &switch_stmt.cases {
                    if let Some(test) = &case.test {
                        self.visit_expr(test);
                    }
                    for s in &case.cons {
                        self.visit_stmt(s);
                    }
                }
            }
            _ => {}
        }
    }

    fn visit_module(&mut self, module: &Module) {
        for item in &module.body {
            match item {
                ModuleItem::Stmt(stmt) => self.visit_stmt(stmt),
                ModuleItem::ModuleDecl(decl) => match decl {
                    ModuleDecl::ExportDecl(export_decl) => {
                        if let Decl::Var(var_decl) = &export_decl.decl {
                            for declarator in &var_decl.decls {
                                if let Some(init) = &declarator.init {
                                    self.visit_expr(init);
                                }
                            }
                        }
                    }
                    ModuleDecl::ExportDefaultExpr(expr) => self.visit_expr(&expr.expr),
                    _ => {}
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_expression() {
        assert!(validate_agent_script("const x = 1 + 2;").is_ok());
    }

    #[test]
    fn test_eval_blocked() {
        let err = validate_agent_script("eval('bad')").unwrap_err();
        assert!(err.contains("eval"), "error should mention eval: {}", err);
    }

    #[test]
    fn test_new_function_blocked() {
        let err = validate_agent_script("new Function('return 1')").unwrap_err();
        assert!(
            err.contains("Function"),
            "error should mention Function: {}",
            err
        );
    }

    #[test]
    fn test_string_settimeout_blocked() {
        let err = validate_agent_script("setTimeout('code', 100)").unwrap_err();
        assert!(
            err.contains("setTimeout"),
            "error should mention setTimeout: {}",
            err
        );
    }

    #[test]
    fn test_fn_ref_settimeout_allowed() {
        assert!(validate_agent_script("setTimeout(fn, 100)").is_ok());
        assert!(validate_agent_script("setTimeout(() => {}, 100)").is_ok());
    }

    // ── Indirect eval bypasses ───────────────────────────────────────────
    #[test]
    fn blocks_indirect_eval_comma() {
        assert!(validate_agent_script("(0, eval)('code')").is_err());
    }

    #[test]
    fn blocks_indirect_eval_paren() {
        assert!(validate_agent_script("(eval)('code')").is_err());
    }

    #[test]
    fn blocks_member_eval_window() {
        assert!(validate_agent_script("window.eval('code')").is_err());
    }

    #[test]
    fn blocks_member_eval_global_this() {
        assert!(validate_agent_script("globalThis.eval('code')").is_err());
    }

    // ── Indirect Function bypasses ───────────────────────────────────────
    #[test]
    fn blocks_indirect_new_function_comma() {
        assert!(validate_agent_script("new (0, Function)('return 1')").is_err());
    }

    // ── Indirect setTimeout/setInterval bypasses ─────────────────────────
    #[test]
    fn blocks_member_settimeout_string() {
        assert!(validate_agent_script("window.setTimeout('code', 100)").is_err());
    }

    #[test]
    fn blocks_member_setinterval_string() {
        assert!(validate_agent_script("globalThis.setInterval('code', 100)").is_err());
    }

    // ── Legitimate code passes ───────────────────────────────────────────
    #[test]
    fn allows_lambda_call() {
        assert!(validate_agent_script("(() => 42)()").is_ok());
    }

    #[test]
    fn allows_function_declaration() {
        assert!(validate_agent_script("function add(a, b) { return a + b }").is_ok());
    }

    // ── Static computed property access ───────────────────────────────────
    #[test]
    fn blocks_static_computed_eval() {
        assert!(validate_agent_script("window['eval']('code')").is_err());
    }

    #[test]
    fn allows_dynamic_computed_property() {
        assert!(validate_agent_script("window[x]('code')").is_ok());
    }

    /// This _should_ theoretically be blocked but static analysis cannot
    /// resolve runtime-decoded strings like `atob`.  This test documents
    /// the ceiling — future dynamic taint analysis could close this gap.
    #[test]
    fn allows_runtime_decoded_computed_eval() {
        assert!(validate_agent_script("window[atob('ZXZhbA==')]('code')").is_ok());
    }
}
