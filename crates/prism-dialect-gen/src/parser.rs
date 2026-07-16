use crate::ast::*;
use crate::lexer::Token;
use logos::Logos;

/// Parse a complete TableGen document from a source string.
pub fn parse_document(source: &str) -> Result<TdDocument, String> {
    let mut lex = Token::lexer(source);
    let mut records = Vec::new();
    loop {
        // Get next non-error token
        let next_tok = loop {
            match lex.next() {
                Some(Ok(Token::Error)) => continue,
                result => break result,
            }
        };
        match next_tok {
            Some(Ok(Token::Def)) | Some(Ok(Token::Class)) | Some(Ok(Token::Multiclass)) => {
                let kind = match next_tok.unwrap() {
                    Ok(Token::Def) => RecordKind::Def,
                    Ok(Token::Class) => RecordKind::Class,
                    Ok(Token::Multiclass) => RecordKind::Multiclass,
                    _ => unreachable!(),
                };
                let record = parse_record(&mut lex, kind)?;
                records.push(record);
            }
            None => break,
            _ => {}
        }
    }
    Ok(TdDocument { records })
}

fn parse_record(lex: &mut logos::Lexer<Token>, kind: RecordKind) -> Result<TdRecord, String> {
    let name = match expect_token(lex)? {
        Token::Ident(s) => s,
        other => return Err(format!("expected identifier after keyword, got {other:?}")),
    };

    // Optional template arguments: <...>
    let template_args = if peek(lex) == Some(Token::LAngle) {
        parse_template_args(lex)?
    } else {
        Vec::new()
    };

    // Optional superclass list after ':'
    let superclasses = if peek(lex) == Some(Token::Colon) {
        lex.next(); // consume ':'
        parse_superclass_list(lex)?
    } else {
        Vec::new()
    };

    // Body
    let body = if peek(lex) == Some(Token::LBrace) {
        parse_body(lex)?
    } else if peek(lex) == Some(Token::Semicolon) {
        lex.next(); // consume ';'
        Vec::new()
    } else if peek(lex) == Some(Token::Eq) {
        // def NAME = expression; (alternative form)
        lex.next();
        parse_value(lex)?;
        expect_token(lex)?; // Semicolon
        Vec::new()
    } else {
        Vec::new()
    };

    Ok(TdRecord {
        name,
        kind,
        template_args,
        superclasses,
        body,
    })
}

/// Try to parse a type constraint from the current position.
///
/// In TableGen, template args can look like:
/// - `name` (bare name, no type constraint)
/// - `Type:name` (colon-separated type:name)
/// - `Type name` (space-separated type and name)
///
/// If a type constraint is found, consume its tokens and return Some(type_string).
/// If not, return None without consuming anything.
fn try_parse_type_constraint(lex: &mut logos::Lexer<Token>) -> Option<String> {
    let first = peek(lex)?;
    let Token::Ident(_) = first else { return None };

    // Check what follows this ident
    let after = peek_after(lex);
    match after {
        // Type:name pattern
        Some(Token::Colon) => {
            let ty = match lex.next() {
                Some(Ok(Token::Ident(s))) => s,
                _ => return None,
            };
            lex.next(); // consume ':'
            Some(ty)
        }
        // Type name pattern (two idents in a row)
        Some(Token::Ident(_)) => {
            let ty = match lex.next() {
                Some(Ok(Token::Ident(s))) => s,
                _ => return None,
            };
            Some(ty)
        }
        // No type constraint -- just a bare name
        _ => None,
    }
}

fn parse_template_args(lex: &mut logos::Lexer<Token>) -> Result<Vec<TemplateArg>, String> {
    let mut args = Vec::new();
    // consume '<'
    lex.next();

    loop {
        // Allow empty arg list
        if peek(lex) == Some(Token::RAngle) {
            lex.next();
            return Ok(args);
        }

        // Template arg patterns:
        //   "name"                     (no type)
        //   "Type name"                (type precedes name, no colon)
        //   "Type:name"                (type:name with colon)
        //   "Type:name = default"      (with default value)
        let type_constraint = try_parse_type_constraint(lex);

        let name = match expect_token(lex)? {
            Token::Ident(s) => s,
            other => return Err(format!("expected template arg name, got {other:?}")),
        };

        args.push(TemplateArg {
            name,
            type_constraint,
        });

        match peek(lex) {
            Some(Token::Comma) => {
                lex.next();
            }
            Some(Token::RAngle) => {
                lex.next();
                return Ok(args);
            }
            Some(Token::Eq) => {
                // Default value: Type:name = value
                lex.next();
                parse_value(lex)?;
                if peek(lex) == Some(Token::Comma) {
                    lex.next();
                } else if peek(lex) == Some(Token::RAngle) {
                    lex.next();
                    return Ok(args);
                }
            }
            other => {
                return Err(format!(
                    "expected ',' or '>' in template args, got {other:?}"
                ))
            }
        }
    }
}

fn parse_superclass_list(lex: &mut logos::Lexer<Token>) -> Result<Vec<SuperclassRef>, String> {
    let mut refs = Vec::new();
    loop {
        let name = match expect_token(lex)? {
            Token::Ident(s) => s,
            other => return Err(format!("expected superclass name, got {other:?}")),
        };

        let args = if peek(lex) == Some(Token::LAngle) {
            parse_value_list(lex)?
        } else if peek(lex) == Some(Token::LParen) {
            parse_value_list_paren(lex)?
        } else {
            Vec::new()
        };

        refs.push(SuperclassRef { name, args });

        match peek(lex) {
            Some(Token::Comma) => {
                lex.next();
            }
            Some(Token::Semicolon) | Some(Token::LBrace) | Some(Token::Eq) => break,
            _ => break,
        }
    }
    Ok(refs)
}

fn parse_value_list(lex: &mut logos::Lexer<Token>) -> Result<Vec<Value>, String> {
    // consume '<'
    lex.next();
    let mut values = Vec::new();
    loop {
        if peek(lex) == Some(Token::RAngle) {
            lex.next();
            return Ok(values);
        }
        values.push(parse_value(lex)?);
        match peek(lex) {
            Some(Token::Comma) => {
                lex.next();
            }
            Some(Token::RAngle) => {
                lex.next();
                return Ok(values);
            }
            other => return Err(format!("expected ',' or '>' in value list, got {other:?}")),
        }
    }
}

fn parse_value_list_paren(lex: &mut logos::Lexer<Token>) -> Result<Vec<Value>, String> {
    // consume '('
    lex.next();
    let mut values = Vec::new();
    loop {
        if peek(lex) == Some(Token::RParen) {
            lex.next();
            return Ok(values);
        }
        values.push(parse_value(lex)?);
        match peek(lex) {
            Some(Token::Comma) => {
                lex.next();
            }
            Some(Token::RParen) => {
                lex.next();
                return Ok(values);
            }
            other => {
                return Err(format!(
                    "expected ',' or ')' in paren value list, got {other:?}"
                ))
            }
        }
    }
}

fn parse_body(lex: &mut logos::Lexer<Token>) -> Result<Vec<LetBlock>, String> {
    // consume '{'
    lex.next();
    let mut blocks = Vec::new();

    loop {
        // Skip semicolons
        while peek(lex) == Some(Token::Semicolon) {
            lex.next();
        }

        match peek(lex) {
            Some(Token::RBrace) => {
                lex.next();
                return Ok(blocks);
            }
            Some(Token::Let) => {
                lex.next(); // consume 'let'
                let block = parse_let(lex)?;
                blocks.push(block);
            }
            Some(Token::Ident(_)) => {
                let block = parse_let(lex)?;
                blocks.push(block);
            }
            // Nested record definitions (def/defm inside multiclass) — skip them
            Some(Token::Def) | Some(Token::Defm) => {
                skip_nested_record(lex)?;
            }
            None => return Err("unexpected EOF in body".to_string()),
            _ => {
                return Err(format!("unexpected token in body: {:?}", peek(lex)));
            }
        }
    }
}

fn parse_let(lex: &mut logos::Lexer<Token>) -> Result<LetBlock, String> {
    let name = match expect_token(lex)? {
        Token::Ident(s) => s,
        other => return Err(format!("expected identifier in let, got {other:?}")),
    };

    let is_prism_annotation = name.starts_with("prism_");

    // '=' or '= value'
    if peek(lex) != Some(Token::Eq) {
        return Err(format!("expected '=' after let name {name}"));
    }
    lex.next(); // consume '='

    let value = parse_value(lex)?;

    // Semicolon optional (allow newline or '}')
    if peek(lex) == Some(Token::Semicolon) {
        lex.next();
    }

    Ok(LetBlock {
        name,
        value,
        is_prism_annotation,
    })
}

fn parse_value(lex: &mut logos::Lexer<Token>) -> Result<Value, String> {
    match peek(lex) {
        Some(Token::Bang) => parse_bang(lex),
        Some(Token::Dag) => {
            lex.next(); // consume 'dag'
            if peek(lex) == Some(Token::LParen) {
                parse_dag(lex)
            } else {
                // bare dag keyword
                Ok(Value::Ident("dag".to_string()))
            }
        }
        Some(Token::LParen) => parse_dag(lex),
        Some(Token::LBracket) => parse_list(lex),
        Some(Token::StringLit(_)) => {
            let s = match lex.next() {
                Some(Ok(Token::StringLit(s))) => s,
                _ => unreachable!(),
            };
            Ok(Value::StringLit(s))
        }
        Some(Token::IntLit(_)) => {
            let n = match lex.next() {
                Some(Ok(Token::IntLit(n))) => n,
                _ => unreachable!(),
            };
            Ok(Value::IntLit(n))
        }
        Some(Token::BitLit(_)) => {
            let b = match lex.next() {
                Some(Ok(Token::BitLit(b))) => b,
                _ => unreachable!(),
            };
            Ok(Value::BitLit(b))
        }
        Some(Token::LBrace) => {
            // Code block: [{ ... }]
            lex.next(); // consume '{'
            parse_code_block(lex)
        }
        Some(Token::Ident(_)) => {
            let s = match lex.next() {
                Some(Ok(Token::Ident(s))) => s,
                _ => unreachable!(),
            };
            Ok(Value::Ident(s))
        }
        Some(Token::Question) => {
            // '?' as a value (unset/undetermined in TableGen)
            lex.next();
            Ok(Value::Ident("?".to_string()))
        }
        Some(Token::Ins) => {
            lex.next();
            Ok(Value::Ident("ins".to_string()))
        }
        Some(Token::Outs) => {
            lex.next();
            Ok(Value::Ident("outs".to_string()))
        }
        Some(other) => Err(format!("unexpected token in value position: {other:?}")),
        None => Err("unexpected EOF in value".to_string()),
    }
}

fn parse_bang(lex: &mut logos::Lexer<Token>) -> Result<Value, String> {
    lex.next(); // consume '!'
    let op = match expect_token(lex)? {
        Token::Ident(s) => s,
        other => return Err(format!("expected identifier after '!', got {other:?}")),
    };

    if peek(lex) == Some(Token::LParen) {
        lex.next(); // consume '('
        let mut args = Vec::new();
        loop {
            if peek(lex) == Some(Token::RParen) {
                lex.next();
                return Ok(Value::Bang { op, args });
            }
            args.push(parse_value(lex)?);
            match peek(lex) {
                Some(Token::Comma) => {
                    lex.next();
                }
                Some(Token::RParen) => {
                    lex.next();
                    return Ok(Value::Bang { op, args });
                }
                other => {
                    return Err(format!(
                        "expected ',' or ')' in bang args for {op}, got {other:?}"
                    ));
                }
            }
        }
    }

    Ok(Value::Bang {
        op,
        args: Vec::new(),
    })
}

fn parse_dag(lex: &mut logos::Lexer<Token>) -> Result<Value, String> {
    // consume '('
    lex.next();

    // The root is the first value (usually an identifier)
    let root_val = parse_value(lex)?;
    let root = match root_val {
        Value::Ident(s) => s,
        Value::StringLit(s) => s,
        _ => {
            return Err(format!(
                "expected identifier/string as dag root, got {root_val:?}"
            ))
        }
    };

    let mut args = Vec::new();
    loop {
        match peek(lex) {
            Some(Token::Comma) => {
                lex.next();
            }
            Some(Token::RParen) => {
                lex.next();
                return Ok(Value::Dag { root, args });
            }
            Some(Token::Colon) => {
                // Named dag arg: :name
                lex.next();
                let name = match expect_token(lex)? {
                    Token::Ident(s) => s,
                    other => return Err(format!("expected dag arg name, got {other:?}")),
                };
                // If there's a value after the name
                match peek(lex) {
                    Some(Token::Comma) | Some(Token::RParen) => {
                        // This is just a name, push a placeholder
                        args.push(DagArg {
                            name: Some(name),
                            value: Box::new(Value::Ident("?".to_string())),
                        });
                    }
                    Some(_) => {
                        let val = parse_value(lex)?;
                        args.push(DagArg {
                            name: Some(name),
                            value: Box::new(val),
                        });
                    }
                    None => {
                        return Err("unexpected EOF in dag".to_string());
                    }
                }
            }
            Some(_) => {
                let val = parse_value(lex)?;
                // Check for :name after the value
                if peek(lex) == Some(Token::Colon) {
                    lex.next(); // consume ':'
                    let name = match expect_token(lex)? {
                        Token::Ident(s) => s,
                        other => return Err(format!("expected dag arg name, got {other:?}")),
                    };
                    args.push(DagArg {
                        name: Some(name),
                        value: Box::new(val),
                    });
                } else {
                    args.push(DagArg {
                        name: None,
                        value: Box::new(val),
                    });
                }
            }
            None => return Err("unexpected EOF in dag".to_string()),
        }
    }
}

fn parse_list(lex: &mut logos::Lexer<Token>) -> Result<Value, String> {
    // consume '['
    lex.next();
    let mut values = Vec::new();
    loop {
        if peek(lex) == Some(Token::RBracket) {
            lex.next();
            return Ok(Value::List(values));
        }
        values.push(parse_value(lex)?);
        match peek(lex) {
            Some(Token::Comma) => {
                lex.next();
            }
            Some(Token::RBracket) => {
                lex.next();
                return Ok(Value::List(values));
            }
            other => {
                return Err(format!("expected ',' or ']' in list, got {other:?}"));
            }
        }
    }
}

fn parse_code_block(lex: &mut logos::Lexer<Token>) -> Result<Value, String> {
    // We're past the initial '{'. Now gather everything until matching '}'.
    let mut depth = 1u32;
    let mut content = String::new();

    // Since logos doesn't tokenize balanced braces well, we work from the raw source.
    // We've already consumed the '{'. Get the current position and scan forward.
    let span = lex.span();
    let source_slice = lex.source();
    let start = span.end;

    let mut pos = start;
    for (i, ch) in source_slice[start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    // End of code block
                    content = source_slice[start..start + i].to_string();
                    // Advance the lexer past the closing '}' — count bytes
                    let end_byte = start + i + 1; // +1 for '}'
                                                  // We need to advance lexer to this position
                                                  // Unfortunately logos doesn't support seeking, so we re-scan
                                                  // and manually advance tokens
                    pos = end_byte;
                    break;
                }
            }
            _ => {}
        }
    }

    if depth > 0 {
        return Err("unterminated code block".to_string());
    }

    // Advance the underlying logos lexer past the code block
    // Since we can't seek, we need to eat tokens until we're past the byte position
    // The safest approach: advance to correct byte position by consuming tokens
    loop {
        match lex.next() {
            Some(Ok(_)) | Some(Err(_)) => {
                if lex.span().start >= pos {
                    break;
                }
            }
            None => break,
        }
    }

    Ok(Value::Code(content.trim().to_string()))
}

fn expect_token(lex: &mut logos::Lexer<Token>) -> Result<Token, String> {
    match lex.next() {
        Some(Ok(tok)) => Ok(tok),
        Some(Err(())) => {
            let rest = &lex.source()[lex.span().start..];
            let end = rest
                .find(|c: char| c.is_whitespace() || c == '{' || c == '}' || c == ';')
                .unwrap_or(rest.len().min(20));
            Err(format!("unexpected token near '{}'", &rest[..end]))
        }
        None => Err("unexpected end of input".to_string()),
    }
}

fn peek(lex: &logos::Lexer<Token>) -> Option<Token> {
    let mut clone = lex.clone();
    // Skip whitespace/error tokens
    loop {
        match clone.next() {
            Some(Ok(Token::Error)) => continue,
            Some(Ok(tok)) => return Some(tok),
            Some(Err(())) => return None,
            None => return None,
        }
    }
}

/// Like peek but advances the clone by exactly one token (no skip).
fn peek_after(lex: &logos::Lexer<Token>) -> Option<Token> {
    let mut clone = lex.clone();
    match clone.next() {
        Some(Ok(tok)) => Some(tok),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_def() {
        let td = r#"
def ADDFOp : Arith_Op<"addf"> {
  let summary = "floating-point addition";
  let arguments = (ins FloatLikeType:$lhs, FloatLikeType:$rhs);
  let results = (outs FloatLikeType:$result);
}
"#;
        let doc = parse_document(td).unwrap();
        assert_eq!(doc.records.len(), 1);
        let rec = &doc.records[0];
        assert_eq!(rec.name, "ADDFOp");
        assert_eq!(rec.kind, RecordKind::Def);
        assert_eq!(rec.superclasses.len(), 1);
        assert_eq!(rec.superclasses[0].name, "Arith_Op");
        assert_eq!(rec.superclasses[0].args.len(), 1);
        assert_eq!(rec.body.len(), 3);
        assert_eq!(rec.body[0].name, "summary");
        assert_eq!(rec.body[1].name, "arguments");
        assert_eq!(rec.body[2].name, "results");
    }

    #[test]
    fn test_parse_class() {
        let td = r#"
class OpBase<string name> {
  let op_name = name;
}
"#;
        let doc = parse_document(td).unwrap();
        assert_eq!(doc.records.len(), 1);
        assert_eq!(doc.records[0].kind, RecordKind::Class);
        assert_eq!(doc.records[0].template_args.len(), 1);
        assert_eq!(doc.records[0].template_args[0].name, "name");
        assert_eq!(
            doc.records[0].template_args[0].type_constraint,
            Some("string".to_string())
        );
    }

    #[test]
    fn test_parse_value_types() {
        let td = r#"
def TestOp {
  let flag = true;
  let count = 42;
  let bitval = 0b1;
  let list_val = [1, 2, 3];
}
"#;
        let doc = parse_document(td).unwrap();
        let body = &doc.records[0].body;
        assert_eq!(body.len(), 4);
        // Check int literal survives
        assert!(matches!(body[1].value, Value::IntLit(42)));
        // Check list
        assert!(matches!(&body[3].value, Value::List(v) if v.len() == 3));
    }
}
/// Skip a nested record (def/defm) inside a multiclass body.
/// This handles the full def/defm syntax including its optional body.
fn skip_nested_record(lex: &mut logos::Lexer<Token>) -> Result<(), String> {
    lex.next(); // consume def/defm

    // Parse optional name
    match peek(lex) {
        Some(Token::Ident(_)) => {
            lex.next(); // consume name
        }
        _ => {}
    }

    // Optional template args: <...>
    if peek(lex) == Some(Token::LAngle) {
        // Skip until matching >
        let mut depth = 1u32;
        loop {
            match lex.next() {
                Some(Ok(Token::LAngle)) => depth += 1,
                Some(Ok(Token::RAngle)) => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                Some(Ok(_)) => {}
                Some(Err(())) => return Err("error token in template args".to_string()),
                None => return Err("unexpected EOF in template args".to_string()),
            }
        }
    }

    // Optional colon + superclass list: skip until { or ;
    if peek(lex) == Some(Token::Colon) {
        lex.next(); // consume ':'
        skip_until_lbrace_or_semicolon(lex)?;
    }

    // Optional body
    if peek(lex) == Some(Token::LBrace) {
        skip_balanced_braces(lex)?;
    }

    Ok(())
}

/// Skip a balanced brace-delimited block { ... }
fn skip_balanced_braces(lex: &mut logos::Lexer<Token>) -> Result<(), String> {
    lex.next(); // consume '{'
    let mut depth = 1u32;
    loop {
        match lex.next() {
            Some(Ok(Token::LBrace)) => depth += 1,
            Some(Ok(Token::RBrace)) => {
                depth -= 1;
                if depth == 0 {
                    return Ok(());
                }
            }
            Some(Ok(_)) => {}
            Some(Err(())) => return Err("error token in block".to_string()),
            None => return Err("unexpected EOF in block".to_string()),
        }
    }
}
/// Skip tokens until we hit '{' or ';' (end of a def/defm header).
/// Properly handles balanced angle brackets.
fn skip_until_lbrace_or_semicolon(lex: &mut logos::Lexer<Token>) -> Result<(), String> {
    let mut angle_depth: i32 = 0;
    loop {
        match lex.next() {
            Some(Ok(Token::LBrace)) if angle_depth <= 0 => return Ok(()),
            Some(Ok(Token::Semicolon)) if angle_depth <= 0 => return Ok(()),
            Some(Ok(Token::LAngle)) => angle_depth += 1,
            Some(Ok(Token::RAngle)) => angle_depth -= 1,
            Some(Ok(_)) => {}
            Some(Err(())) => return Err("error token in skip".to_string()),
            None => return Err("unexpected EOF while skipping".to_string()),
        }
    }
}
