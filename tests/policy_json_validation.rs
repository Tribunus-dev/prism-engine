/// Verify compiler_policy.json structure: valid JSON, every policy has
/// evidence nested correctly with operator (when present) inside evidence.
#[test]
fn verify_compiler_policy_json() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("compiler_policy.json");
    assert!(path.exists(), "compiler_policy.json not found");

    let content = std::fs::read_to_string(&path).expect("read failed");
    let value: serde_json::Value = serde_json::from_str(&content).expect("invalid JSON");

    let policies = value["model_policies"]
        .as_array()
        .expect("model_policies not an array");

    for (i, policy) in policies.iter().enumerate() {
        let name = policy["scope"].as_str().unwrap_or("unnamed");

        // Every policy must have evidence with weight_space
        let evidence = &policy["evidence"];
        assert!(evidence.is_object(), "{name}: evidence not an object");
        assert!(evidence.get("weight_space").is_some(), "{name}: missing weight_space");

        // operator, when present, must be inside evidence
        if evidence.get("operator").is_some() {
            assert!(
                evidence["operator"].is_object(),
                "{name}: evidence.operator not an object"
            );
        }

        // operator must NOT appear at policy level
        assert!(
            policy.get("operator").is_none(),
            "{name}: operator found at policy level — must be inside evidence"
        );

        // validation_status must be present
        assert!(
            policy.get("validation_status").is_some(),
            "{name}: missing validation_status"
        );

        // promotion_gates_met must have an entry for this scope, or the scope is the Qwen policy
        let gates = &value["auto_promotion_rules"]["promotion_gates_met"];
        if gates.get(name).is_none() && !name.contains("Qwen") {
            // Qwen is weight_space_only, no gate needed yet
            // All others must have a gate
            assert!(gates.get(name).is_some(), "{name}: missing from promotion_gates_met");
        }
    }

    eprintln!("compiler_policy.json: {} policies validated", policies.len());
}
