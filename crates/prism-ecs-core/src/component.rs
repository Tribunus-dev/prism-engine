/// Tag trait for data attached to entities.
pub trait Component: std::fmt::Debug + Send + Sync + 'static {}

// Basic type implementations for common Rust types used as components.
impl Component for String {}
impl Component for u64 {}
impl Component for i64 {}
impl Component for i32 {}
impl Component for f32 {}
impl Component for bool {}
impl Component for usize {}
