//! `transitions_orchestrator` — the client-side dispatcher
//! for state-driven motion.
//!
//! The orchestrator is a thin JS module that
//!   1. Subscribes to the SelectionController.
//!   2. Loads the prism-transitions WASM module on idle.
//!   3. On a state change, calls the WASM for the right
//!      transition string (e.g. dispersion gradient, lift
//!      keyframe, transition shorthand), and applies it to
//!      the matching DOM elements.
//!
//! The CSS in `site.css` carries the static transitions
//! (the no-JS / no-WASM fallback). The orchestrator adds the
//! dynamic layer on top. If the WASM fails to load, the site
//! still renders correctly — just without the choreography.

/// The TransitionsOrchestrator JavaScript source. Loaded as
/// a `<script>` (not `<script type="module">`) so the
/// `import('/transitions/prism_transitions.js')` inside is
/// the only module import and runs after the DOM is parsed.
pub const TRANSITIONS_ORCHESTRATOR_JS: &str = include_str!("transitions_orchestrator.js");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orchestrator_is_non_empty() {
        assert!(!TRANSITIONS_ORCHESTRATOR_JS.is_empty());
    }

    #[test]
    fn orchestrator_uses_idle_callback() {
        // Per §9: the WASM load must not block first paint.
        assert!(TRANSITIONS_ORCHESTRATOR_JS.contains("requestIdleCallback") ||
                TRANSITIONS_ORCHESTRATOR_JS.contains("setTimeout"));
    }

    #[test]
    fn orchestrator_subscribes_to_selection() {
        // The orchestrator's whole reason for being is to
        // listen to the SelectionController and dispatch
        // motion on state change.
        assert!(TRANSITIONS_ORCHESTRATOR_JS.contains("prismSelection.subscribe"));
    }

    #[test]
    fn orchestrator_handles_wasm_load_failure() {
        // If the WASM fails to load, the site continues with
        // CSS-only transitions. The orchestrator must not
        // throw.
        assert!(TRANSITIONS_ORCHESTRATOR_JS.contains(".catch"));
    }
}
