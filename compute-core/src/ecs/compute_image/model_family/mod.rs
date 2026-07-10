pub mod gemma4_unified;

pub use gemma4_unified::*;
pub mod qwen25_omni;
pub use qwen25_omni::*;

#[cfg(test)]
mod gemma4_12b_schema_fixture;
