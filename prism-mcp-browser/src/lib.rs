use anyhow::{anyhow, bail, Result};
use parking_lot::Mutex;
use prism_mcp_core::{ArtifactRepository, DaemonState, EvidenceStore, McpHandler, RequestContext, ToolRequest, ToolResult};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserConfig { pub webdriver_url: String, pub startup_timeout_ms: u64, pub max_script_bytes: usize, pub max_result_bytes: usize, pub allowed_hosts: Vec<String> }

impl Default for BrowserConfig {
    fn default() -> Self { Self { webdriver_url: std::env::var("PRISM_BROWSER_WEBDRIVER_URL").unwrap_or_else(|_| "http://127.0.0.1:4444".into()), startup_timeout_ms: 5000, max_script_bytes: 64 * 1024, max_result_bytes: 8 * 1024 * 1024, allowed_hosts: std::env::var("PRISM_BROWSER_ALLOWED_HOSTS").unwrap_or_default().split(',').filter(|v| !v.is_empty()).map(str::to_owned).collect() } }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tab { pub title: String, pub url: String, pub index: usize }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClickableRegion { pub id: String, pub tag: String, pub text: String, pub selector: String, pub x: f64, pub y: f64, pub width: f64, pub height: f64 }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredPageData { pub title: String, pub url: String, pub text: String, pub links: Vec<ClickableRegion>, pub forms: Vec<ClickableRegion> }

struct Driver { session_id: String, child: Option<Child>, last_used: Instant }

pub struct BrowserSession { config: BrowserConfig, driver: Option<Driver> }

impl BrowserSession {
    pub fn new(config: BrowserConfig) -> Self { Self { config, driver: None } }
    fn ensure(&mut self) -> Result<&mut Driver> {
        if self.driver.is_none() {
            let port = self.config.webdriver_url.rsplit(':').next().unwrap_or("4444");
            let child = Command::new("safaridriver").args(["--port", port]).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn().ok();
            let deadline = Instant::now() + Duration::from_millis(self.config.startup_timeout_ms);
            let agent = ureq::Agent::new();
            while Instant::now() < deadline {
                if let Ok(response) = agent.get(&format!("{}/status", self.config.webdriver_url)).call() { if response.status() < 500 { break; } }
                std::thread::sleep(Duration::from_millis(50));
            }
            let body = self.request("POST", "/session", Some(json!({"capabilities":{"alwaysMatch":{"browserName":"safari"}}})))?;
            let id = body["sessionId"].as_str().or_else(|| body["value"]["sessionId"].as_str()).ok_or_else(|| anyhow!("WebDriver did not return a session id"))?.to_owned();
            self.driver = Some(Driver { session_id: id, child, last_used: Instant::now() });
        }
        Ok(self.driver.as_mut().unwrap())
    }
    fn request(&mut self, method: &str, path: &str, payload: Option<Value>) -> Result<Value> {
        let url = format!("{}{}", self.config.webdriver_url, path);
        let agent = ureq::Agent::new();
        let response = match (method, payload) { ("GET", _) => agent.get(&url).call()?, ("DELETE", _) => agent.delete(&url).call()?, ("POST", Some(body)) => agent.post(&url).send_json(body)?, ("POST", None) => agent.post(&url).call()?, _ => bail!("unsupported WebDriver method {method}") };
        Ok(response.into_json()?)
    }
    fn command(&mut self, method: &str, path: &str, payload: Option<Value>) -> Result<Value> { let id = self.ensure()?.session_id.clone(); let result = self.request(method, &format!("/session/{id}{path}"), payload)?; if result["value"].is_object() && result["value"]["error"].is_string() { bail!("WebDriver error: {}", result["value"]["message"].as_str().unwrap_or("unknown")); } if let Some(d) = self.driver.as_mut() { d.last_used = Instant::now(); } Ok(result["value"].clone()) }
    fn validate_url(&self, url: &str) -> Result<()> { let parsed = url::Url::parse(url)?; if parsed.scheme() != "http" && parsed.scheme() != "https" { bail!("only http and https URLs are allowed") } if !self.config.allowed_hosts.is_empty() && !self.config.allowed_hosts.iter().any(|h| h == parsed.host_str().unwrap_or("")) { bail!("host is not allowed") } Ok(()) }
    fn execute_js(&mut self, script: &str, args: Vec<Value>) -> Result<Value> { if script.len() > self.config.max_script_bytes { bail!("script exceeds configured limit") } let result = self.command("POST", "/execute/sync", Some(json!({"script":script,"args":args})))?; let bytes = serde_json::to_vec(&result)?; if bytes.len() > self.config.max_result_bytes { bail!("script result exceeds configured limit") } Ok(result) }
    fn close(&mut self) -> Result<()> { if let Some(mut d) = self.driver.take() { let _ = self.request("DELETE", &format!("/session/{}", d.session_id), None); if let Some(mut child) = d.child.take() { let _ = child.kill(); } } Ok(()) }
}

impl Drop for BrowserSession { fn drop(&mut self) { let _ = self.close(); } }

pub struct ToolDependencies { pub evidence_ledger: Arc<dyn EvidenceStore>, pub artifact_store: Arc<dyn ArtifactRepository>, pub resource_leases: Arc<dyn prism_mcp_core::LeaseStore>, pub tools: Arc<HashMap<&'static str, Arc<dyn McpHandler + Sync + Send>>> }

static SESSION: OnceLock<Arc<Mutex<BrowserSession>>> = OnceLock::new();
fn session(_state: &DaemonState) -> Result<Arc<Mutex<BrowserSession>>> { Ok(SESSION.get_or_init(|| Arc::new(Mutex::new(BrowserSession::new(BrowserConfig::default())))).clone()) }
fn lease(state: &DaemonState, owner: &str) -> Result<()> { if state.resource_leases.acquire("browser:safari", owner, 30)? { Ok(()) } else { bail!("browser session is leased by another agent") } }

struct BrowserHandler { name: &'static str }
impl McpHandler for BrowserHandler {
    fn name(&self) -> &'static str { self.name }
    fn description(&self) -> &'static str { "MCPD-managed Safari WebDriver browser operation with safety limits and evidence." }
    fn input_schema(&self) -> Value { json!({"type":"object","properties":{"url":{"type":"string"},"code":{"type":"string"},"selector":{"type":"string"},"id":{"type":"string"},"x":{"type":"number"},"y":{"type":"number"},"text":{"type":"string"},"session_owner":{"type":"string"}},"additionalProperties":false}) }
    fn call(&self, request: ToolRequest<'_>, _context: &RequestContext, state: &DaemonState) -> Result<ToolResult> {
        let owner = request.args.get("session_owner").and_then(Value::as_str).unwrap_or("mcp-agent"); lease(state, owner)?; let browser = session(state)?; let mut browser = browser.lock();
        let value = match self.name {
            "browser_session_close" => { browser.close()?; json!({"closed":true}) }
            "browser_navigate" => { let url=request.args.get("url").and_then(Value::as_str).ok_or_else(||anyhow!("url is required"))?; browser.validate_url(url)?; browser.command("POST", "/url", Some(json!({"url":url})))?; json!({"url":url,"status":"navigated"}) }
            "browser_current_url" => browser.command("GET", "/url", None)?,
            "browser_page_source" => browser.command("GET", "/source", None)?,
            "browser_page_text" => browser.execute_js("return document.body ? document.body.innerText : '';", vec![])?,
            "browser_execute_js" => browser.execute_js(request.args.get("code").and_then(Value::as_str).ok_or_else(||anyhow!("code is required"))?, vec![])?,
            "browser_screenshot" => { let raw=browser.command("GET", "/screenshot", None)?; let data=raw.as_str().unwrap_or_default(); json!({"base64_png":data}) }
            "browser_structured_extract" | "browser_structured_view" | "browser_interactive_regions" => browser.execute_js(STRUCTURED_EXTRACT, vec![])? ,
            "browser_click_region" => { let selector=request.args.get("id").and_then(Value::as_str).ok_or_else(||anyhow!("id is required"))?; browser.execute_js("const e=document.querySelector(arguments[0]); if(!e) throw new Error('not found'); e.click(); return true;", vec![json!(selector)])?; json!({"clicked":true}) }
            "browser_type_at" => { let x=request.args.get("x").and_then(Value::as_f64).ok_or_else(||anyhow!("x is required"))?; let y=request.args.get("y").and_then(Value::as_f64).ok_or_else(||anyhow!("y is required"))?; let text=request.args.get("text").and_then(Value::as_str).ok_or_else(||anyhow!("text is required"))?; let keys: Vec<Value> = text.chars().map(|c| json!({"type":"keyDown","value":c.to_string()})).collect(); let mut actions=vec![json!({"type":"pointerMove","x":x,"y":y,"duration":0}),json!({"type":"pointerDown","button":0}),json!({"type":"pointerUp","button":0})]; actions.extend(keys); browser.command("POST", "/actions", Some(json!({"actions":[{"type":"pointer","id":"prism-pointer","parameters":{"pointerType":"mouse"},"actions":actions}]})))?; json!({"typed":true}) }
            "browser_find_element" => { let selector=request.args.get("selector").and_then(Value::as_str).ok_or_else(||anyhow!("selector is required"))?; browser.execute_js("return !!document.querySelector(arguments[0]);", vec![json!(selector)])? }
            "browser_get_tabs" => json!([{"title":"current","url":browser.command("GET", "/url", None)?.as_str().unwrap_or_default(),"index":0}]),
            _ => bail!("unknown browser tool {}", self.name),
        };
        Ok(ToolResult::text(serde_json::to_string(&value)?))
    }
}

const STRUCTURED_EXTRACT: &str = r#"return {title:document.title,url:location.href,text:document.body?.innerText||'',links:[...document.querySelectorAll('a,button,input,textarea,select')].slice(0,500).map((e,i)=>({id:'dom-'+i,tag:e.tagName.toLowerCase(),text:(e.innerText||e.value||'').slice(0,1000),selector:e.id?'#'+e.id:e.tagName.toLowerCase(),x:e.getBoundingClientRect().x,y:e.getBoundingClientRect().y,width:e.getBoundingClientRect().width,height:e.getBoundingClientRect().height})),forms:[]};"#;

pub fn handlers(deps: &ToolDependencies) -> Vec<Arc<dyn McpHandler + Sync + Send>> { let _ = deps; ["browser_navigate","browser_page_source","browser_page_text","browser_current_url","browser_execute_js","browser_screenshot","browser_structured_extract","browser_interactive_regions","browser_structured_view","browser_click_region","browser_type_at","browser_find_element","browser_get_tabs","browser_session_close"].into_iter().map(|name| Arc::new(BrowserHandler{name}) as Arc<dyn McpHandler + Sync + Send>).collect() }

pub fn session_from_env() -> Arc<Mutex<BrowserSession>> { Arc::new(Mutex::new(BrowserSession::new(BrowserConfig::default()))) }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn url_policy_rejects_non_web_and_disallowed_hosts() {
        let config = BrowserConfig { allowed_hosts: vec!["example.com".into()], ..Default::default() };
        let session = BrowserSession::new(config);
        assert!(session.validate_url("file:///etc/passwd").is_err());
        assert!(session.validate_url("https://evil.example/").is_err());
        assert!(session.validate_url("https://example.com/page").is_ok());
    }
    #[test]
    fn script_limit_is_enforced_before_transport() {
        let mut session = BrowserSession::new(BrowserConfig { max_script_bytes: 3, ..Default::default() });
        assert!(session.execute_js("1234", vec![]).is_err());
    }
}
