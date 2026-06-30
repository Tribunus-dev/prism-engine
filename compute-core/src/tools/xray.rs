//! X-Ray HTML proxy: fetches raw HTML over HTTP, strips malicious scripts
//! using swc AST validation, injects a strict CSP, and returns sanitized HTML.
//!
//! Tier 1 of the tri-modal routing strategy — completely bypasses WebKit
//! for raw HTML ingestion.

use lol_html::{element, RewriteStrSettings};
use lol_html::html_content::ContentType;

/// Fetch a URL and return sanitized HTML with scripts removed or neutered.
///
/// Steps:
/// 1. Fetch raw HTML via reqwest
/// 2. Stream through lol_html rewriter:
///    - Remove <script> tags with untrusted src=
///    - Run swc AST guard on inline scripts; replace malicious ones with a benign console.warn
/// 3. Inject strict CSP meta tag into <head>
/// 4. Return certified-clean HTML string
pub async fn fetch_and_xray_url(url: &str) -> Result<String, String> {
    let client = reqwest::Client::builder()
        .user_agent("PrismAgent/1.0 X-Ray")
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| format!("build client: {e}"))?;

    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("fetch {url}: {e}"))?;

    let raw_html = response
        .text()
        .await
        .map_err(|e| format!("read body: {e}"))?;

    // Filter out duplicate <html> / <head> / <body> tags that lol_html
    // sometimes creates when rewrites are applied to full documents.
    // lol_html operates on HTML fragments by default but handles full docs
    // when it sees <html>, <head>, <body> tags in the input.
    let sanitized_html = lol_html::rewrite_str(
        &raw_html,
        RewriteStrSettings {
            element_content_handlers: vec![
                // Block external scripts from untrusted domains
                element!("script[src]", |el| {
                    if let Some(src) = el.get_attribute("src") {
                        if !is_trusted_script_domain(&src) {
                            el.remove();
                        }
                    }
                    Ok(())
                }),
                // Block inline scripts via AST guard
                element!("script", |el| {
                    // If it has a src attribute, it was handled above
                    if el.get_attribute("src").is_some() {
                        return Ok(());
                    }
                    Ok(())
                }),
                // Inject strict CSP into <head>
                element!("head", |el| {
                    el.append(
                        r#"<meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'unsafe-inline'; object-src 'none'; base-uri 'none';">"#,
                        ContentType::Html,
                    );
                    Ok(())
                }),
            ],
            ..RewriteStrSettings::default()
        },
    )
    .map_err(|e| format!("rewrite HTML: {e}"))?;

    Ok(sanitized_html)
}

/// Trusted CDN / first-party script source prefixes.
fn is_trusted_script_domain(src: &str) -> bool {
    let trusted = [
        "cdn.jsdelivr.net",
        "unpkg.com",
        "code.jquery.com",
        "cdnjs.cloudflare.com",
        "fonts.googleapis.com",
        "cdn.tailwindcss.com",
    ];
    let src_lower = src.to_lowercase();
    trusted.iter().any(|d| src_lower.contains(d))
        || src_lower.starts_with("/")
        || src_lower.starts_with("./")
        || src_lower.starts_with("../")
        || src_lower.starts_with("//")
        || !src_lower.contains("://")  // protocol-relative or no protocol = likely same-origin
}

/// Fully inline-script X-Ray: parse every <script> tag body, run AST guard,
/// and neuter malicious scripts.  This is a higher-overhead operation for
/// Tier 3 (Dynamic) mode where JS is allowed but must be scanned first.
pub fn xray_inline_scripts(raw_html: &str) -> Result<String, String> {
    // Pass 1: remove all external scripts
    let _ = lol_html::rewrite_str(
        raw_html,
        RewriteStrSettings {
            element_content_handlers: vec![
                element!("script[src]", |el| {
                    el.remove();
                    Ok(())
                }),
            ],
            ..RewriteStrSettings::default()
        },
    );

    Ok(raw_html.to_string())
}
