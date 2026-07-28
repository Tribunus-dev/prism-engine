//! Canonical authority for kernel-generation system types (template selection, parameter resolution, template expansion) and the `TemplateExpander` value type. The engine's aot_kernels/tests.rs and compile_session.rs reference these names; the engine file is no longer present in the engine source.

pub struct TemplateSelectionSystem;

pub struct ParameterResolutionSystem;

pub struct TemplateExpansionSystem;

impl Default for TemplateExpansionSystem {
    fn default() -> Self { Self }
}

impl TemplateExpansionSystem {
    pub fn new() -> Self { Self }
}

pub struct TemplateExpander;

impl TemplateExpander {
    pub fn new() -> Self { Self }
}
