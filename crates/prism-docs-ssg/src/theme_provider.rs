//! `theme_provider` — the client-side owner of the dark/light
//! theme attribute.
//!
//! Per `OBSERVATORY_V1_SPEC.md` §9 (Visual Direction), the
//! site ships with dark and light themes. The ThemeProvider
//! is the sole writer of the `data-theme` attribute on
//! `<html>`. It reads `localStorage` for the visitor's
//! stored choice, falls back to `prefers-color-scheme`, and
//! wires the theme toggle button in the site header.
//!
//! The provider is a small vanilla module. It does not
//! perform network requests, does not depend on a framework,
//! and does not block the parser. The no-flash guard (a
//! tiny inline script in `<head>` before this module runs)
//! sets the attribute synchronously so the visitor never sees
//! the wrong theme.

/// The ThemeProvider JavaScript source. Loaded as a
/// `<script defer>` after the no-flash guard.
pub const THEME_PROVIDER_JS: &str = include_str!("theme_provider.js");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn theme_provider_is_non_empty() {
        assert!(!THEME_PROVIDER_JS.is_empty());
    }

    #[test]
    fn theme_provider_does_not_perform_network_requests() {
        // The provider reads localStorage and matchMedia; it
        // must not make any network calls. Even same-origin
        // fetches would violate A19.
        let network_calls = [
            "fetch(", "XMLHttpRequest", "WebSocket", "EventSource",
            "importScripts", "sendBeacon", "navigator.sendBeacon",
        ];
        for needle in &network_calls {
            assert!(
                !THEME_PROVIDER_JS.contains(needle),
                "theme provider must not use {}",
                needle
            );
        }
    }

    #[test]
    fn theme_provider_uses_localstorage() {
        // The storage key is the §A22 contract.
        assert!(THEME_PROVIDER_JS.contains("'prism-theme'"));
    }

    #[test]
    fn theme_provider_respects_prefers_color_scheme() {
        assert!(THEME_PROVIDER_JS.contains("prefers-color-scheme"));
    }

    #[test]
    fn theme_provider_wires_toggle_button() {
        assert!(THEME_PROVIDER_JS.contains("theme-toggle"));
    }
}
