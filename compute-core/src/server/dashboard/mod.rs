/// Tribunus Dashboard SPA — served at `/dashboard` for operators to inspect
/// loaded cimages, tensor quality, evidence, and run chat inference.
pub const DASHBOARD_HTML: &str = include_str!("page.html");

#[cfg(feature = "server-dashboard")]
pub mod analytics;
#[cfg(feature = "server-dashboard")]
pub mod api;
#[cfg(feature = "server-dashboard")]
pub mod cache;
#[cfg(feature = "server-dashboard")]
pub mod models;
#[cfg(feature = "server-dashboard")]
pub mod schema;
