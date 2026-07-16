use serde::{Deserialize, Serialize};

/// A complete TableGen document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TdDocument {
    pub records: Vec<TdRecord>,
}

/// A single record definition (def, class, or multiclass).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TdRecord {
    pub name: String,
    pub kind: RecordKind,
    pub template_args: Vec<TemplateArg>,
    pub superclasses: Vec<SuperclassRef>,
    pub body: Vec<LetBlock>,
}

/// What kind of record this is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecordKind {
    Def,
    Class,
    Multiclass,
}

/// A template argument in angle brackets after the record name.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateArg {
    pub name: String,
    pub type_constraint: Option<String>,
}

/// A superclass reference with optional arguments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperclassRef {
    pub name: String,
    pub args: Vec<Value>,
}

/// A let block (field assignment) in the record body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LetBlock {
    pub name: String,
    pub value: Value,
    /// True when the let name is prefixed with `prism_` or annotated as an
    /// internal annotation (not part of the standard td dialect).
    pub is_prism_annotation: bool,
}

/// Value expressions in TableGen.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Value {
    Ident(String),
    StringLit(String),
    IntLit(i64),
    BitLit(bool),
    List(Vec<Value>),
    Dag {
        root: String,
        args: Vec<DagArg>,
    },
    /// Bare code segment (captured between `[{}]` markers in td files).
    Code(String),
    /// Bang operator: `!<op>(<args>...)`
    Bang {
        op: String,
        args: Vec<Value>,
    },
}

/// An argument inside a dag value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagArg {
    pub name: Option<String>,
    pub value: Box<Value>,
}
