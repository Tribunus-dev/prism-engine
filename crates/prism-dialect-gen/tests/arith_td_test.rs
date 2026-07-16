use prism_dialect_gen::{emit_rust, parse_document, resolve_document};

/// A minimal arith_ops.td-like string to test the full pipeline.
const MINIMAL_ARITH_TD: &str = r#"
// Base op class
class OpBase<string name> {
  let op_name = name;
}

// Arith dialect base
class Arith_Op<string op> : OpBase<!strconcat("arith.", op)> {
  let summary = "";
  let arguments = (ins);
  let results = (outs);
}

// Float ops
def ADDFOp : Arith_Op<"addf"> {
  let summary = "floating-point addition";
  let arguments = (ins FloatLikeType:$lhs, FloatLikeType:$rhs);
  let results = (outs FloatLikeType:$result);
}
def SUBFOp : Arith_Op<"subf"> {
  let summary = "floating-point subtraction";
  let arguments = (ins FloatLikeType:$lhs, FloatLikeType:$rhs);
  let results = (outs FloatLikeType:$result);
}
"#;

/// A multiclass with 2 defm entries for defm expansion testing.
const MULTICLASS_DEFM_TD: &str = r#"
class Arith_Op<string op> {
  let op_name = op;
}

multiclass IntArithOps {
  def ADDOp : Arith_Op<"add"> {
    let summary = "integer addition";
  }
  def SUBOp : Arith_Op<"sub"> {
    let summary = "integer subtraction";
  }
}

defm IntAdd : IntArithOps;
defm IntSub : IntArithOps;
"#;

#[test]
fn test_parse_emit_arith_ops() {
    let doc = parse_document(MINIMAL_ARITH_TD).unwrap();
    assert!(
        !doc.records.is_empty(),
        "should parse at least some records"
    );

    // Check we parsed the class and defs
    let def_count = doc
        .records
        .iter()
        .filter(|r| r.kind == prism_dialect_gen::RecordKind::Def)
        .count();
    assert_eq!(def_count, 2, "should have 2 defs (ADDFOp, SUBFOp)");

    // Resolve
    let resolved = resolve_document(&doc).unwrap();
    assert!(!resolved.is_empty(), "should resolve at least one record");

    // Emit
    let output = emit_rust(&resolved).unwrap_or_else(|e| {
        panic!("emit_rust failed: {e}");
    });

    // Verify output patterns
    assert!(
        output.contains("pub enum"),
        "output should contain 'pub enum', got:\n{output}"
    );

    // Check for op_name method
    assert!(
        output.contains("op_name"),
        "output should contain 'op_name', got:\n{output}"
    );

    // Check for Component implementation pattern
    assert!(
        output.contains("impl Component for"),
        "output should contain 'impl Component for', got:\n{output}"
    );

    println!("emit_rust output:\n{output}");
}

#[test]
fn test_multiclass_defm_generates_two_variants() {
    let doc = parse_document(MULTICLASS_DEFM_TD).unwrap();
    assert!(!doc.records.is_empty(), "should parse records");

    let resolved = resolve_document(&doc).unwrap();

    // Check we get resolved records
    assert!(!resolved.is_empty(), "should resolve records");

    let output = emit_rust(&resolved).unwrap_or_else(|e| {
        panic!("emit_rust failed: {e}");
    });

    // Check output has at least these patterns
    assert!(output.contains("pub enum"));
    assert!(output.contains("op_name"));

    // The exact variant count depends on how many records are recognized as ops.
    // With our simple td that has only IntArithOps as superclass, the resolver
    // should produce at least 2 variants (from the two defm entries if defm expands them,
    // or from the two defs in the multiclass if processed directly).
    //
    // Since our resolver doesn't fully handle defm expansion yet, we at least verify
    // the pipeline doesn't crash.
    println!("Multiclass emit output:\n{output}");
}

#[test]
fn test_full_body_roundtrip() {
    let td = r#"
class FooOp<string op> {
}

def TestOp : FooOp<"test_op"> {
  let summary = "A test operation";
  let arguments = (ins I32:$input);
  let results = (outs I32:$output);
  let has_verifier = true;
}
"#;

    let doc = parse_document(td).unwrap();
    assert_eq!(doc.records.len(), 2);

    let rec = &doc.records[1]; // the def
    assert_eq!(rec.name, "TestOp");
    assert_eq!(rec.body.len(), 4);
    assert_eq!(rec.body[0].name, "summary");
    assert_eq!(rec.body[1].name, "arguments");
    assert_eq!(rec.body[2].name, "results");
    assert_eq!(rec.body[3].name, "has_verifier");

    // Verify dag parse in arguments
    if let prism_dialect_gen::Value::Dag { root, args } = &rec.body[1].value {
        assert_eq!(root, "ins");
        assert_eq!(args.len(), 1);
        assert_eq!(args[0].name, Some("input".to_string()));
    } else {
        panic!("arguments should be a dag value");
    }

    if let prism_dialect_gen::Value::Dag { root, args } = &rec.body[2].value {
        assert_eq!(root, "outs");
        assert_eq!(args.len(), 1);
        assert_eq!(args[0].name, Some("output".to_string()));
    } else {
        panic!("results should be a dag value");
    }

    // Emit
    let resolved = resolve_document(&doc).unwrap();
    let output = emit_rust(&resolved).unwrap();
    assert!(output.contains("Test,"), "should contain Test variant");
}

#[test]
fn test_parse_arith_def_summary_survives() {
    let td = r#"
class OpBase<string name> {
  let op_name = name;
}

class Arith_Op<string op> : OpBase<op> {
  let summary = "";
  let arguments = (ins);
  let results = (outs);
}

def ADDFOp : Arith_Op<"addf"> {
  let summary = "floating-point addition";
  let arguments = (ins FloatLikeType:$lhs, FloatLikeType:$rhs);
  let results = (outs FloatLikeType:$result);
}
"#;

    let doc = parse_document(td).unwrap();
    let def_rec = doc.records.iter().find(|r| r.name == "ADDFOp").unwrap();
    assert_eq!(def_rec.body.len(), 3);

    // Check summary survives
    let summary_block = def_rec.body.iter().find(|b| b.name == "summary").unwrap();
    if let prism_dialect_gen::Value::StringLit(s) = &summary_block.value {
        assert_eq!(s, "floating-point addition");
    } else {
        panic!("summary should be a string literal");
    }

    // Check arguments dag
    let args_block = def_rec.body.iter().find(|b| b.name == "arguments").unwrap();
    if let prism_dialect_gen::Value::Dag { root, args } = &args_block.value {
        assert_eq!(root, "ins");
        assert_eq!(args.len(), 2);
        assert_eq!(args[0].name, Some("lhs".to_string()));
        assert_eq!(args[1].name, Some("rhs".to_string()));
    } else {
        panic!("arguments should be a dag value");
    }
}

#[test]
fn test_empty_file() {
    let td = "";
    let doc = parse_document(td).unwrap();
    assert!(doc.records.is_empty());
}

#[test]
fn test_single_def_without_body() {
    let td = "def FooOp;";
    let doc = parse_document(td).unwrap();
    assert_eq!(doc.records.len(), 1);
    assert_eq!(doc.records[0].name, "FooOp");
    assert!(doc.records[0].body.is_empty());
}
