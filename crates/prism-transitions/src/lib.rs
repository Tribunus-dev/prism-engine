//! `prism-transitions` — state-driven motion for Prism
//! Observatory.
//!
//! Per `OBSERVATORY_V1_SPEC.md` §9, the prism effect refracts
//! when the subject crosses a representation or hardware
//! boundary. The motion is real, but it is small: a brief
//! dispersion, a stage-lift, a state-pulse. CSS is the
//! renderer (transitions, keyframes, GPU-accelerated
//! transforms). This crate is the choreographer: it computes
//! the curves, durations, and dispersion angles for each
//! transition, and exposes them as CSS strings.
//!
//! The crate is split into pure Rust (testable on the host)
//! and a thin WASM surface (consumed by the browser). The
//! pure Rust layer has no JS dependencies and runs in
//! `cargo test`. The WASM layer is feature-gated and only
//! compiled with `--features wasm --target wasm32-unknown-unknown`.
//!
//! No-JS fallback: the CSS already carries static transitions
//! for every component. The WASM module adds dynamic,
//! state-driven keyframes. If the WASM does not load, the
//! static transitions still play.

#![cfg_attr(target_arch = "wasm32", allow(dead_code))]

/// The canonical motion vocabulary. These are the only
/// motion tokens the WASM exports. The site uses the same
/// values in CSS (tokens.css) so a visitor with WASM and a
/// visitor without see the same baseline motion; the WASM
/// adds the dynamic layer on top.
pub mod motion {
    pub const FAST_MS: u32 = 120;
    pub const BASE_MS: u32 = 200;
    pub const SLOW_MS: u32 = 360;
    pub const PEAK_MS: u32 = 480;

    pub const EASE: &str = "cubic-bezier(0.2, 0.0, 0.2, 1)";
    pub const EASE_OUT: &str = "cubic-bezier(0.0, 0.0, 0.2, 1)";
}

/// The prism dispersion. Two color stops, cool and warm,
/// anchored by the page background. Each call returns a CSS
/// gradient string suitable for `background-image`.
pub mod dispersion {
    pub const COOL: &str = "#6ad4ff";
    pub const WARM: &str = "#c56ad4";
    pub const COOL_DARK: &str = "#0066aa";
    pub const WARM_DARK: &str = "#8a2d99";

    /// Return the dispersion gradient for a given state
    /// transition. The argument names the kind of boundary
    /// being crossed:
    ///
    /// - `"implemented"` — a presence boundary; the cool
    ///   end dominates.
    /// - `"qualifying"` — a flight boundary; the warm end
    ///   dominates.
    /// - `"validated"` — a materialization boundary; the
    ///   full prism refracts, cool to warm.
    /// - `"boundary"` — a representation or hardware
    ///   boundary (the §9 case); the dispersion is
    ///   perpendicular to the page, sweeping from cool to
    ///   warm at 135°.
    pub fn gradient(boundary: &str) -> String {
        match boundary {
            "implemented" => format!(
                "linear-gradient(135deg, {cool} 0%, {cool} 100%)",
                cool = COOL
            ),
            "qualifying" => format!(
                "linear-gradient(135deg, {warm} 0%, {warm} 100%)",
                warm = WARM
            ),
            "validated" => format!(
                "linear-gradient(135deg, {cool} 0%, {warm} 100%)",
                cool = COOL,
                warm = WARM
            ),
            "boundary" | _ => format!(
                "linear-gradient(135deg, {cool} 0%, {warm} 100%)",
                cool = COOL,
                warm = WARM
            ),
        }
    }
}

/// Build the CSS `transition` shorthand for a state change.
/// Returns `property duration easing[, ...]`.
pub fn transition_for(from: &str, to: &str) -> String {
    use motion::*;
    match (from, to) {
        ("planned", "implemented") | ("implemented", "planned") => {
            format!("background {base}ms {ease}, border-color {fast}ms {ease}",
                base = BASE_MS, fast = FAST_MS, ease = EASE)
        }
        ("qualifying", "validated") | ("validated", "qualifying") => {
            format!("background {slow}ms {ease}, border-color {base}ms {ease}, transform {peak}ms {ease}",
                base = BASE_MS, slow = SLOW_MS, peak = PEAK_MS, ease = EASE)
        }
        ("unreleased", "released") | ("released", "unreleased") => {
            format!("background {base}ms {ease}, border-color {fast}ms {ease}",
                base = BASE_MS, fast = FAST_MS, ease = EASE)
        }
        _ => format!("background {base}ms {ease}, border-color {base}ms {ease}, transform {base}ms {ease}",
            base = BASE_MS, ease = EASE),
    }
}

/// Build a CSS `@keyframes` block (without the `@keyframes
/// <name> {` wrapper) for a one-shot pulse. Used by the
/// prism mark to "throb" when a state changes.
pub fn pulse_keyframes(_name: &str) -> String {
    use motion::*;
    format!(
        "0% {{ transform: scale(1); opacity: 1; }}\n\
         50% {{ transform: scale(1.05); opacity: 0.85; }}\n\
         100% {{ transform: scale(1); opacity: 1; }}"
    )
}

/// Build a CSS `@keyframes` block for an Observatory stage
/// lift. The selected stage rises slightly and gains a soft
/// glow; the previously selected stage returns to base.
pub fn observatory_select_keyframes(_name: &str) -> String {
    use motion::*;
    format!(
        "0% {{ transform: translateY(0); box-shadow: 0 0 0 transparent; }}\n\
         50% {{ transform: translateY(-3px); box-shadow: 0 0 24px rgba(106, 212, 255, 0.35); }}\n\
         100% {{ transform: translateY(-1px); box-shadow: 0 0 12px rgba(106, 212, 255, 0.25); }}"
    )
}

/// The full CSS rule, including the `@keyframes <name> {`
/// wrapper, for a one-shot pulse. Convenience: callers that
/// want to inject a complete rule.
pub fn pulse_rule(name: &str) -> String {
    format!("@keyframes {} {{\n{}\n}}", name, pulse_keyframes(name))
}

/// The full CSS rule, including the `@keyframes <name> {`
/// wrapper, for an Observatory stage lift.
pub fn observatory_select_rule(name: &str) -> String {
    format!(
        "@keyframes {} {{\n{}\n}}",
        name,
        observatory_select_keyframes(name)
    )
}

// ---------- WASM surface ----------
//
// When built for wasm32-unknown-unknown with the `wasm`
// feature, the following functions are exported and callable
// from JavaScript via the wasm-bindgen JS glue. Each returns
// a CSS string the orchestrator can inject into a
// <style> element or apply as a CSS custom property.

#[cfg(feature = "wasm")]
mod wasm_surface {
    use super::*;
    use wasm_bindgen::prelude::*;

    /// Return a CSS gradient for a boundary. The JavaScript
    /// caller names the boundary; this returns the value.
    #[wasm_bindgen]
    pub fn dispersion_for(boundary: &str) -> String {
        dispersion::gradient(boundary)
    }

    /// Return a CSS `transition` shorthand for a state
    /// change. `from` and `to` are the state names.
    #[wasm_bindgen]
    pub fn transition(from: &str, to: &str) -> String {
        transition_for(from, to)
    }

    /// Return a complete CSS `@keyframes` rule for a
    /// one-shot pulse. The `name` is the keyframe name (the
    /// JS orchestrator chooses it).
    #[wasm_bindgen]
    pub fn pulse(name: &str) -> String {
        pulse_rule(name)
    }

    /// Return a complete CSS `@keyframes` rule for an
    /// Observatory stage lift.
    #[wasm_bindgen]
    pub fn observatory_lift(name: &str) -> String {
        observatory_select_rule(name)
    }

    /// Apply a state transition to an element. The element
    /// is identified by the JS side (a CSS selector); this
    /// returns the transition string the JS can set as
    /// `element.style.transition`.
    ///
    /// The actual DOM mutation happens in JS — this crate
    /// is a pure CSS-string factory. The orchestrator owns
    /// the DOM.
    #[wasm_bindgen]
    pub fn state_transition(from: &str, to: &str) -> String {
        transition_for(from, to)
    }

    /// Version string. The JS orchestrator can use this to
    /// check that the WASM matches the CSS.
    #[wasm_bindgen]
    pub fn version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }
}

// ---------- Tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispersion_for_implemented_is_cool() {
        let g = dispersion::gradient("implemented");
        assert!(g.contains(dispersion::COOL));
        assert!(!g.contains(dispersion::WARM));
    }

    #[test]
    fn dispersion_for_qualifying_is_warm() {
        let g = dispersion::gradient("qualifying");
        assert!(g.contains(dispersion::WARM));
    }

    #[test]
    fn dispersion_for_validated_refracts() {
        let g = dispersion::gradient("validated");
        assert!(g.contains(dispersion::COOL));
        assert!(g.contains(dispersion::WARM));
    }

    #[test]
    fn dispersion_for_boundary_refracts() {
        let g = dispersion::gradient("boundary");
        assert!(g.contains(dispersion::COOL));
        assert!(g.contains(dispersion::WARM));
    }

    #[test]
    fn transition_for_implemented_to_planned_is_fast() {
        let t = transition_for("planned", "implemented");
        assert!(t.contains("background"));
        assert!(t.contains("120ms") || t.contains("200ms"));
    }

    #[test]
    fn transition_for_qualifying_to_validated_includes_transform() {
        let t = transition_for("qualifying", "validated");
        assert!(t.contains("transform"));
    }

    #[test]
    fn pulse_rule_is_complete() {
        let r = pulse_rule("prism-pulse");
        assert!(r.starts_with("@keyframes prism-pulse {"));
        assert!(r.contains("scale(1.05)"));
    }

    #[test]
    fn observatory_lift_includes_translateY() {
        let r = observatory_select_rule("prism-obs-lift");
        assert!(r.contains("translateY(-1px)"));
        assert!(r.contains("translateY(-3px)"));
    }

    #[test]
    fn unknown_boundary_falls_back_to_dispersion() {
        // Spec §9: any boundary that isn't one of the four
        // named kinds still gets a dispersion. The unknown
        // case must not panic.
        let g = dispersion::gradient("this-is-unknown");
        assert!(g.contains("linear-gradient"));
    }
}
