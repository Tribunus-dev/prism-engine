pub mod ast;
pub mod emit;
pub mod lexer;
pub mod parser;
pub mod resolve;

pub use ast::*;
pub use emit::emit_rust;
pub use lexer::{Lexer, Token};
pub use parser::parse_document;
pub use resolve::{resolve_document, ResolvedRecord};
