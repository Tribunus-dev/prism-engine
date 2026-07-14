use anyhow::{anyhow, bail, Result};
use parking_lot::Mutex;
use prism_mcp_core::{
    ArtifactRepository, DaemonState, EvidenceReceipt, EvidenceStatus, EvidenceStore, FileLock,
    McpHandler, MetricSet, RequestContext, ToolInvocationId, ToolRequest, ToolResult,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
pub mod dom;
mod sandbox;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserConfig {
    pub webdriver_url: String,
    pub startup_timeout_ms: u64,
    pub max_script_bytes: usize,
    pub max_result_bytes: usize,
    pub allowed_hosts: Vec<String>,
}

impl Default for BrowserConfig {
    fn default() -> Self {
        Self {
            webdriver_url: std::env::var("PRISM_BROWSER_WEBDRIVER_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:4444".into()),
            startup_timeout_ms: 5000,
            max_script_bytes: 64 * 1024,
            max_result_bytes: 8 * 1024 * 1024,
            allowed_hosts: std::env::var("PRISM_BROWSER_ALLOWED_HOSTS")
                .unwrap_or_default()
                .split(',')
                .filter(|v| !v.is_empty())
                .map(str::to_owned)
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tab {
    pub title: String,
    pub url: String,
    pub index: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClickableRegion {
    pub id: String,
    pub tag: String,
    pub text: String,
    pub selector: String,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredPageData {
    pub title: String,
    pub url: String,
    pub text: String,
    pub links: Vec<ClickableRegion>,
    pub forms: Vec<ClickableRegion>,
}

struct Driver {
    session_id: String,
    child: Option<Child>,
    last_used: Instant,
}

pub struct BrowserSession {
    config: BrowserConfig,
    driver: Option<Driver>,
    dom_revision: dom::DomRevision,
}

impl BrowserSession {
    pub fn new(config: BrowserConfig) -> Self {
        Self {
            config,
            driver: None,
            dom_revision: dom::DomRevision(0),
        }
    }
    fn bump_dom_revision(&mut self) {
        self.dom_revision.0 = self.dom_revision.0.saturating_add(1);
    }
    fn current_dom_revision(&self) -> dom::DomRevision {
        self.dom_revision
    }
    fn ensure(&mut self) -> Result<&mut Driver> {
        if self.driver.is_none() {
            let port = self
                .config
                .webdriver_url
                .rsplit(':')
                .next()
                .unwrap_or("4444");
            let deadline = Instant::now() + Duration::from_millis(self.config.startup_timeout_ms);
            let agent = ureq::Agent::new();
            let mut child = None;
            let startup_lock = FileLock::new(
                &std::env::temp_dir().join("prism-mcp-browser-safaridriver.lock"),
            );
            let _startup_guard = startup_lock.lock()?;
            while Instant::now() < deadline {
                if let Ok(response) = agent
                    .get(&format!("{}/status", self.config.webdriver_url))
                    .call()
                {
                    if response.status() < 500 {
                        break;
                    }
                }
                if child.is_none() {
                    child = Command::new("safaridriver")
                        .args(["--port", port])
                        .stdin(Stdio::null())
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .spawn()
                        .ok();
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            let body = self.request(
                "POST",
                "/session",
                Some(json!({"capabilities":{"alwaysMatch":{"browserName":"safari"}}})),
            )?;
            let id = body["sessionId"]
                .as_str()
                .or_else(|| body["value"]["sessionId"].as_str())
                .ok_or_else(|| anyhow!("WebDriver did not return a session id"))?
                .to_owned();
            self.driver = Some(Driver {
                session_id: id,
                child,
                last_used: Instant::now(),
            });
        }
        Ok(self.driver.as_mut().unwrap())
    }
    fn request(&mut self, method: &str, path: &str, payload: Option<Value>) -> Result<Value> {
        let url = format!("{}{}", self.config.webdriver_url, path);
        let agent = ureq::Agent::new();
        let response = match (method, payload) {
            ("GET", _) => agent.get(&url).call()?,
            ("DELETE", _) => agent.delete(&url).call()?,
            ("POST", Some(body)) => agent.post(&url).send_json(body)?,
            ("POST", None) => agent.post(&url).call()?,
            _ => bail!("unsupported WebDriver method {method}"),
        };
        Ok(response.into_json()?)
    }
    fn command(&mut self, method: &str, path: &str, payload: Option<Value>) -> Result<Value> {
        let id = self.ensure()?.session_id.clone();
        let result = self.request(method, &format!("/session/{id}{path}"), payload)?;
        if result["value"].is_object() && result["value"]["error"].is_string() {
            bail!(
                "WebDriver error: {}",
                result["value"]["message"].as_str().unwrap_or("unknown")
            );
        }
        if let Some(d) = self.driver.as_mut() {
            d.last_used = Instant::now();
        }
        Ok(result["value"].clone())
    }
    fn validate_url(&self, url: &str) -> Result<()> {
        let parsed = url::Url::parse(url)?;
        if parsed.scheme() != "http" && parsed.scheme() != "https" {
            bail!("only http and https URLs are allowed")
        }
        if !self.config.allowed_hosts.is_empty()
            && !self
                .config
                .allowed_hosts
                .iter()
                .any(|h| h == parsed.host_str().unwrap_or(""))
        {
            bail!("host is not allowed")
        }
        Ok(())
    }
    fn execute_js(&mut self, script: &str, args: Vec<Value>) -> Result<Value> {
        if script.len() > self.config.max_script_bytes {
            bail!("script exceeds configured limit")
        }
        let result = self.command(
            "POST",
            "/execute/sync",
            Some(json!({"script":script,"args":args})),
        )?;
        let bytes = serde_json::to_vec(&result)?;
        if bytes.len() > self.config.max_result_bytes {
            bail!("script result exceeds configured limit")
        }
        Ok(result)
    }
    fn close(&mut self) -> Result<()> {
        if let Some(mut d) = self.driver.take() {
            let _ = self.request("DELETE", &format!("/session/{}", d.session_id), None);
            if let Some(mut child) = d.child.take() {
                let _ = child.kill();
            }
        }
        Ok(())
    }

    fn dom_snapshot(&mut self) -> Result<Value> {
        let raw = self.execute_js(DOM_SNAPSHOT, vec![])?;
        let tab = self
            .command("GET", "/window", None)
            .ok()
            .and_then(|v| v.as_str().map(str::to_owned))
            .unwrap_or_else(|| "current".into());
        let nodes = raw["nodes"]
            .as_array()
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .enumerate()
            .map(|(i, v)| dom::node_from_value(&v, tab.clone(), self.dom_revision, i as u32))
            .collect::<Vec<_>>();
        let document = dom::DomDocument {
            revision: self.dom_revision,
            url: raw["url"].as_str().unwrap_or_default().into(),
            title: raw["title"].as_str().unwrap_or_default().into(),
            text: raw["text"].as_str().unwrap_or_default().into(),
            nodes,
        };
        Ok(serde_json::to_value(document)?)
    }

    fn dom_query(&mut self, args: &Value) -> Result<Value> {
        let snapshot: dom::DomDocument = serde_json::from_value(self.dom_snapshot()?)?;
        let query: dom::DomQuery = serde_json::from_value(args.clone())?;
        let nodes: Vec<_> = snapshot
            .nodes
            .into_iter()
            .filter(|node| {
                query.matches(node) && query.css.as_ref().is_none_or(|css| node.selector == *css)
            })
            .collect();
        Ok(json!({"revision":snapshot.revision,"nodes":nodes}))
    }

    fn dom_action(&mut self, operation: &str, args: &Value) -> Result<Value> {
        let node: dom::DomNode = serde_json::from_value(
            args.get("node")
                .cloned()
                .ok_or_else(|| anyhow!("node is required"))?,
        )?;
        if node.id.revision != self.dom_revision {
            bail!(
                "stale DOM node handle: expected revision {}, got {}",
                self.dom_revision.0,
                node.id.revision.0
            );
        }
        if operation == "dom_click" {
            self.execute_js("const e=document.querySelector(arguments[0]); if(!e) throw new Error('node not found'); e.click(); return true;", vec![json!(node.selector)])?;
        } else {
            let text = args
                .get("text")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("text is required"))?;
            self.execute_js("const e=document.querySelector(arguments[0]); if(!e) throw new Error('node not found'); e.focus(); e.value=arguments[1]; e.dispatchEvent(new Event('input',{bubbles:true})); e.dispatchEvent(new Event('change',{bubbles:true})); return true;", vec![json!(node.selector), json!(text)])?;
        }
        self.bump_dom_revision();
        Ok(json!({"ok":true,"revision":self.dom_revision}))
    }
}

impl Drop for BrowserSession {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

pub struct ToolDependencies {
    pub evidence_ledger: Arc<dyn EvidenceStore>,
    pub artifact_store: Arc<dyn ArtifactRepository>,
    pub resource_leases: Arc<dyn prism_mcp_core::LeaseStore>,
    pub tools: Arc<HashMap<&'static str, Arc<dyn McpHandler + Sync + Send>>>,
}

static SESSION: OnceLock<Arc<Mutex<BrowserSession>>> = OnceLock::new();
fn session(_state: &DaemonState) -> Result<Arc<Mutex<BrowserSession>>> {
    Ok(SESSION
        .get_or_init(|| Arc::new(Mutex::new(BrowserSession::new(BrowserConfig::default()))))
        .clone())
}
fn lease(state: &DaemonState, owner: &str) -> Result<()> {
    if state.resource_leases.acquire("browser:safari", owner, 30)? {
        Ok(())
    } else {
        bail!("browser session is leased by another agent")
    }
}

struct BrowserHandler {
    name: &'static str,
    evidence: Arc<dyn EvidenceStore>,
}
impl McpHandler for BrowserHandler {
    fn name(&self) -> &'static str {
        self.name
    }
    fn description(&self) -> &'static str {
        "MCPD-managed Safari WebDriver browser operation with safety limits and evidence."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"url":{"type":"string"},"code":{"type":"string"},"selector":{"type":"string"},"css":{"type":"string"},"role":{"type":"string"},"name":{"type":"string"},"text":{"type":"string"},"node":{"type":"object"},"id":{"type":"string"},"x":{"type":"number"},"y":{"type":"number"},"session_owner":{"type":"string"}},"additionalProperties":false})
    }
    fn call(
        &self,
        request: ToolRequest<'_>,
        _context: &RequestContext,
        state: &DaemonState,
    ) -> Result<ToolResult> {
        let owner = request
            .args
            .get("session_owner")
            .and_then(Value::as_str)
            .unwrap_or("mcp-agent");
        lease(state, owner)?;
        let browser = session(state)?;
        let mut browser = browser.lock();
        let value = match self.name {
            "browser_session_close" => {
                browser.close()?;
                json!({"closed":true})
            }
            "browser_navigate" => {
                let url = request
                    .args
                    .get("url")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("url is required"))?;
                browser.validate_url(url)?;
                browser.command("POST", "/url", Some(json!({"url":url})))?;
                browser.bump_dom_revision();
                json!({"url":url,"status":"navigated","revision":browser.current_dom_revision()})
            }
            "browser_current_url" => browser.command("GET", "/url", None)?,
            "browser_page_source" => browser.command("GET", "/source", None)?,
            "browser_page_text" => browser.execute_js(
                "return document.body ? document.body.innerText : '';",
                vec![],
            )?,
            "browser_execute_js" => browser.execute_js(
                request
                    .args
                    .get("code")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("code is required"))?,
                vec![],
            )?,
            "browser_validate_js" => {
                let code = request
                    .args
                    .get("code")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("code is required"))?;
                sandbox::validate_script(code, browser.config.max_script_bytes)?;
                json!({"valid":true})
            }
            "dom_current_revision" => json!({"revision":browser.current_dom_revision()}),
            "dom_snapshot" => browser.dom_snapshot()?,
            "dom_query" => browser.dom_query(&request.args)?,
            "dom_click" | "dom_type" => browser.dom_action(self.name, &request.args)?,
            "browser_screenshot" => {
                let raw = browser.command("GET", "/screenshot", None)?;
                let data = raw.as_str().unwrap_or_default();
                json!({"base64_png":data})
            }
            "browser_structured_extract"
            | "browser_structured_view"
            | "browser_interactive_regions" => browser.execute_js(STRUCTURED_EXTRACT, vec![])?,
            "browser_click_region" => {
                let selector = request
                    .args
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("id is required"))?;
                browser.execute_js("const e=document.querySelector(arguments[0]); if(!e) throw new Error('not found'); e.click(); return true;", vec![json!(selector)])?;
                json!({"clicked":true})
            }
            "browser_type_at" => {
                let x = request
                    .args
                    .get("x")
                    .and_then(Value::as_f64)
                    .ok_or_else(|| anyhow!("x is required"))?;
                let y = request
                    .args
                    .get("y")
                    .and_then(Value::as_f64)
                    .ok_or_else(|| anyhow!("y is required"))?;
                let text = request
                    .args
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("text is required"))?;
                let keys: Vec<Value> = text
                    .chars()
                    .map(|c| json!({"type":"keyDown","value":c.to_string()}))
                    .collect();
                let mut actions = vec![
                    json!({"type":"pointerMove","x":x,"y":y,"duration":0}),
                    json!({"type":"pointerDown","button":0}),
                    json!({"type":"pointerUp","button":0}),
                ];
                actions.extend(keys);
                browser.command("POST", "/actions", Some(json!({"actions":[{"type":"pointer","id":"prism-pointer","parameters":{"pointerType":"mouse"},"actions":actions}]})))?;
                json!({"typed":true})
            }
            "browser_find_element" => {
                let selector = request
                    .args
                    .get("selector")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("selector is required"))?;
                browser.command(
                    "POST",
                    "/element",
                    Some(json!({"using":"css selector","value":selector})),
                )?
            }
            "browser_get_tabs" => {
                let handles = browser
                    .command("GET", "/window/handles", None)?
                    .as_array()
                    .cloned()
                    .unwrap_or_default();
                let current = browser.command("GET", "/window", None).ok();
                let mut tabs = Vec::new();
                for (index, handle) in handles.iter().enumerate() {
                    browser.command("POST", "/window", Some(json!({"handle":handle})))?;
                    let title = browser
                        .command("GET", "/title", None)?
                        .as_str()
                        .unwrap_or_default()
                        .to_string();
                    let url = browser
                        .command("GET", "/url", None)?
                        .as_str()
                        .unwrap_or_default()
                        .to_string();
                    tabs.push(json!({"title":title,"url":url,"index":index,"handle":handle}));
                }
                if let Some(handle) = current {
                    let _ = browser.command("POST", "/window", Some(json!({"handle":handle})));
                }
                json!(tabs)
            }
            _ => bail!("unknown browser tool {}", self.name),
        };
        let _ = state.resource_leases.release("browser:safari", owner);
        let _ = self.evidence.record(&EvidenceReceipt {
            invocation_id: ToolInvocationId::new(),
            tool: "prism-mcp-browser".into(),
            operation: self.name.into(),
            inputs: vec![],
            outputs: vec![],
            environment: "mcpd".into(),
            target: Some("safari-webdriver".into()),
            source_revision: None,
            status: EvidenceStatus::Success,
            metrics: MetricSet::new(),
            diagnostics: vec![],
            started_at: chrono::Utc::now(),
            duration_ms: 0,
        });
        Ok(ToolResult::text(serde_json::to_string(&value)?))
    }
}

const STRUCTURED_EXTRACT: &str = r#"return {title:document.title,url:location.href,text:document.body?.innerText||'',links:[...document.querySelectorAll('a,button,input,textarea,select')].slice(0,500).map((e)=>{const selector=e.id?'#'+e.id:e.tagName.toLowerCase();return {id:selector,tag:e.tagName.toLowerCase(),text:(e.innerText||e.value||'').slice(0,1000),selector,x:e.getBoundingClientRect().x,y:e.getBoundingClientRect().y,width:e.getBoundingClientRect().width,height:e.getBoundingClientRect().height}}),forms:[]};"#;
const DOM_SNAPSHOT: &str = r#"return (()=>{const visible=e=>{const r=e.getBoundingClientRect(),s=getComputedStyle(e);return !!(r.width&&r.height&&s.visibility!=='hidden'&&s.display!=='none')};const selector=e=>{if(e.id)return '#'+CSS.escape(e.id);let p=[],n=e;while(n&&n.nodeType===1&&p.length<6){let q=n.tagName.toLowerCase();if(n.parentElement){const same=[...n.parentElement.children].filter(x=>x.tagName===n.tagName);if(same.length>1)q+=':nth-of-type('+(same.indexOf(n)+1)+')'}p.unshift(q);n=n.parentElement}return p.join(' > ')};const nodes=[...document.querySelectorAll('a,button,input,textarea,select,[role]')].slice(0,1000).map(e=>{const r=e.getBoundingClientRect(),role=e.getAttribute('role')||({'A':'link','BUTTON':'button','INPUT':'textbox','TEXTAREA':'textbox','SELECT':'combobox'}[e.tagName]||null);return {tag:e.tagName.toLowerCase(),role,name:e.getAttribute('aria-label')||e.getAttribute('name')||e.getAttribute('placeholder')||e.innerText||e.value||'',text:(e.innerText||e.value||'').slice(0,2000),selector:selector(e),visible:visible(e),enabled:!e.disabled,x:r.x,y:r.y,width:r.width,height:r.height}});return {title:document.title,url:location.href,text:document.body?.innerText||'',nodes}})();"#;

pub fn handlers(deps: &ToolDependencies) -> Vec<Arc<dyn McpHandler + Sync + Send>> {
    [
        "browser_navigate",
        "browser_page_source",
        "browser_page_text",
        "browser_current_url",
        "browser_execute_js",
        "browser_validate_js",
        "browser_screenshot",
        "browser_structured_extract",
        "browser_interactive_regions",
        "browser_structured_view",
        "browser_click_region",
        "browser_type_at",
        "browser_find_element",
        "browser_get_tabs",
        "browser_session_close",
        "dom_current_revision",
        "dom_snapshot",
        "dom_query",
        "dom_click",
        "dom_type",
    ]
    .into_iter()
    .map(|name| {
        Arc::new(BrowserHandler {
            name,
            evidence: deps.evidence_ledger.clone(),
        }) as Arc<dyn McpHandler + Sync + Send>
    })
    .collect()
}

pub fn session_from_env() -> Arc<Mutex<BrowserSession>> {
    Arc::new(Mutex::new(BrowserSession::new(BrowserConfig::default())))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn url_policy_rejects_non_web_and_disallowed_hosts() {
        let config = BrowserConfig {
            allowed_hosts: vec!["example.com".into()],
            ..Default::default()
        };
        let session = BrowserSession::new(config);
        assert!(session.validate_url("file:///etc/passwd").is_err());
        assert!(session.validate_url("https://evil.example/").is_err());
        assert!(session.validate_url("https://example.com/page").is_ok());
    }
    #[test]
    fn script_limit_is_enforced_before_transport() {
        let mut session = BrowserSession::new(BrowserConfig {
            max_script_bytes: 3,
            ..Default::default()
        });
        assert!(session.execute_js("1234", vec![]).is_err());
    }
    #[test]
    fn deno_sandbox_rejects_external_capabilities() {
        assert!(sandbox::validate_script("fetch('https://example.com')", 1024).is_err());
        assert!(sandbox::validate_script("const answer = 40 + 2;", 1024).is_ok());
    }
}
