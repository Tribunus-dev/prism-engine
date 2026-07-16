use logos::Logos;

/// MLIR-style TableGen tokens.
#[derive(Logos, Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    #[token("def")]
    Def,
    #[token("class")]
    Class,
    #[token("multiclass")]
    Multiclass,
    #[token("defm")]
    Defm,
    #[token("foreach")]
    ForEach,
    #[token("let")]
    Let,
    #[token("in")]
    In,
    #[token("include")]
    Include,

    // MLIR-specific keywords
    #[token("dag")]
    Dag,
    #[token("ins")]
    Ins,
    #[token("outs")]
    Outs,

    // Identifiers
    #[regex("[a-zA-Z_][a-zA-Z0-9_.]*", |lex| lex.slice().to_string())]
    Ident(String),

    // String literals
    #[regex(r#""[^"]*""#, |lex| {
        let s = lex.slice();
        s[1..s.len()-1].to_string()
    })]
    StringLit(String),

    // Integer literals
    #[regex(r#"-?[0-9]+"#, |lex| lex.slice().parse::<i64>().unwrap())]
    IntLit(i64),

    // Bit literals (0b0 or 0b1)
    #[regex(r"0b[01]", |lex| lex.slice() == "0b1")]
    BitLit(bool),

    // Braces
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("<")]
    LAngle,
    #[token(">")]
    RAngle,

    // Brackets
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,

    // Punctuation
    #[token(":")]
    Colon,
    #[token(";")]
    Semicolon,
    #[token(",")]
    Comma,
    #[token(".")]
    Dot,

    // Operators
    #[token("!")]
    Bang,
    #[token("=")]
    Eq,
    #[token("?")]
    Question,

    /// Any other character or unrecognized token.
    #[regex(r"[ \t\r\n]+", logos::skip)]
    #[regex(r"\$", logos::skip)]
    #[regex(r"//[^\n]*", logos::skip)]
    #[regex(r"/\*[^*]*\*+(?:[^/*][^*]*\*+)*/", logos::skip)]
    Error,
}

/// Lexer type alias.
pub type Lexer<'a> = logos::Lexer<'a, Token>;
