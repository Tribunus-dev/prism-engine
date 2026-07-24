//! Optional GGUF provider entrypoints.

#[cfg(feature = "gguf-compile")]
#[allow(dead_code)]
pub const GGUF_PROVIDER_NOTE: &str = "gguf provider feature enabled";

#[cfg(not(feature = "gguf-compile"))]
#[allow(dead_code)]
const _: () = ();
