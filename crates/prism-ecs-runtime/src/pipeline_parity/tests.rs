// ── Tests ──────────────────────────────────────────────────────────────────

use super::*;

#[test]
fn all_phases_have_contracts() {
    for phase in PipelinePhase::all() {
        let found = PHASE_CONTRACTS.iter().any(|c| c.phase == *phase);
        assert!(found, "phase '{phase}' has no contract in PHASE_CONTRACTS");
    }
}

#[test]
    fn all_contracts_have_inputs() {
        for contract in PHASE_CONTRACTS {
            assert!(
                !contract.inputs.is_empty(),
                "phase '{}' has no inputs",
                contract.phase
            );
        }
    }

    #[test]
    fn all_contracts_have_outputs() {
        for contract in PHASE_CONTRACTS {
            assert!(
                !contract.outputs.is_empty(),
                "phase '{}' has no outputs",
                contract.phase
            );
        }
    }

    #[test]
    fn all_phases_have_non_empty_descriptions() {
        for contract in PHASE_CONTRACTS {
            assert!(
                !contract.description.is_empty(),
                "phase '{}' has empty description",
                contract.phase
            );
        }
    }

    #[test]
    fn all_phases_roundtrip_serde() {
        for phase in PipelinePhase::all() {
            let s = phase.to_string();
            let parsed: PipelinePhase = s
                .parse()
                .unwrap_or_else(|e| panic!("cannot parse '{s}': {e}"));
            assert_eq!(*phase, parsed, "roundtrip failed for '{s}'");
        }
    }

    #[test]
    fn all_phases_roundtrip_json() {
        for phase in PipelinePhase::all() {
            let s = serde_json::to_string(phase).expect("serialize");
            let parsed: PipelinePhase = serde_json::from_str(&s).expect("deserialize");
            assert_eq!(*phase, parsed, "json roundtrip failed for '{s}'");
        }
    }

    #[test]
    fn display_snake_case_no_whitespace() {
        for phase in PipelinePhase::all() {
            let s = phase.to_string();
            assert!(
                !s.contains(' '),
                "Display of '{phase:?}' contains whitespace: '{s}'"
            );
            assert!(
                s.chars()
                    .all(|c| c.is_ascii_lowercase() || c == '_' || c.is_ascii_digit()),
                "Display of '{:?}' has non-snake_case chars: '{s}'",
                phase
            );
        }
    }

    #[test]
    fn support_matrix_covers_all_phases() {
        let matrices = [
            coreai_support_matrix(),
            mlx_support_matrix(),
            accelerate_support_matrix(),
            reference_support_matrix(),
        ];
        for matrix in &matrices {
            let matrix_phases: std::collections::BTreeSet<PipelinePhase> =
                matrix.phases.iter().map(|(p, _)| *p).collect();
            for phase in PipelinePhase::all() {
                assert!(
                    matrix_phases.contains(phase),
                    "phase '{phase}' missing from {} support matrix",
                    matrix.backend,
                );
            }
        }
    }

    #[test]
    fn support_matrix_sorted_by_phase_order() {
        let matrices = [
            coreai_support_matrix(),
            mlx_support_matrix(),
            accelerate_support_matrix(),
            reference_support_matrix(),
        ];
        for matrix in &matrices {
            for (i, (phase, _)) in matrix.phases.iter().enumerate() {
                if i > 0 {
                    let prev = &matrix.phases[i - 1].0;
                    let prev_idx = PipelinePhase::all().iter().position(|p| *p == *prev).unwrap();
                    let cur_idx = PipelinePhase::all().iter().position(|p| *p == *phase).unwrap();
                    assert!(
                        cur_idx >= prev_idx,
                        "{} support matrix phases not sorted: {} (idx {}) before {} (idx {})",
                        matrix.backend,
                        prev,
                        prev_idx,
                        phase,
                        cur_idx,
                    );
                }
            }
        }
    }

    #[test]
    fn graph_family_to_phase_coverage() {
        // All 16 graph families must map to a valid phase or explicitly return Err.
        let families = [
            "matmul",
            "chain_matmul_add_silu",
            "branch_rejoin",
            "multi_output",
            "constant_heavy",
            "reshape_transpose_matmul",
            "softmax_tail",
            "identity_passthrough",
            "add_standalone",
            "mul_standalone",
            "sigmoid_standalone",
            "silu_standalone",
            "matmul_projection",
            "matmul_residual_add",
            "two_matmul_add",
            "matmul_add_silu",
        ];
        for &family in &families {
            match graph_family_to_phase(family) {
                Ok(phase) => {
                    assert!(
                        phase != PipelinePhase::TokenEmbedding,
                        "family '{family}' mapped to TokenEmbedding — should never happen via graph_family_to_phase"
                    );
                }
                Err(e) => {
                    // identity_passthrough is expected to return Err.
                    assert_eq!(
                        family, "identity_passthrough",
                        "unexpected Err for family '{family}': {e}"
                    );
                }
            }
        }
    }

    #[test]
    fn graph_family_identity_passthrough_excluded() {
        let result = graph_family_to_phase("identity_passthrough");
        assert!(result.is_err(), "identity_passthrough should be excluded");
        let err = result.unwrap_err();
        assert!(err.reason.contains("harness control"));
    }

    #[test]
    fn graph_family_unknown_fails_closed() {
        let result = graph_family_to_phase("nonexistent_family");
        assert!(result.is_err(), "unknown family should fail closed");
    }

    #[test]
    fn semantic_contract_id_is_deterministic() {
        let id1 = graph_family_semantic_contract_id("matmul");
        let id2 = graph_family_semantic_contract_id("matmul");
        assert_eq!(id1, id2, "semantic contract ID must be deterministic");
        assert_eq!(id1, "qkv_projection/generic_projection");

        let excluded = graph_family_semantic_contract_id("identity_passthrough");
        assert_eq!(excluded, "excluded/harness_control");
    }

    #[test]
    fn phase_variant_distinguishes_same_phase_families() {
        // Both are Activation but different phase variants.
        let v1 = graph_family_phase_variant("silu_standalone");
        let v2 = graph_family_phase_variant("chain_matmul_add_silu");
        assert_ne!(
            v1, v2,
            "different activation families should have distinct phase variants"
        );
    }

    #[test]
    fn support_matrix_for_returns_non_empty() {
        for backend in &[
            BackendId::CoreAi,
            BackendId::Mlx,
            BackendId::Accelerate,
            BackendId::Reference,
        ] {
            let matrix = support_matrix_for(*backend);
            assert!(
                !matrix.phases.is_empty(),
                "support matrix for {backend} is empty"
            );
            assert_eq!(matrix.backend, *backend);
        }
    }

    #[test]
    fn support_matrix_support_for_returns_some() {
        let matrix = mlx_support_matrix();
        let status = matrix.support_for(PipelinePhase::QkvProjection);
        assert!(
            status.is_some(),
            "MLX should report support for QkvProjection"
        );
        assert_eq!(*status.unwrap(), PhaseSupportStatus::Native);
    }

    #[test]
    fn support_matrix_unsupported_has_code_and_reason() {
        let matrix = accelerate_support_matrix();
        if let Some(PhaseSupportStatus::Unsupported { code, reason }) =
            matrix.support_for(PipelinePhase::AttentionScores)
        {
            assert_eq!(*code, UnsupportedCode::NeedsGraphScheduling);
            assert!(!reason.is_empty(), "Unsupported reason must not be empty");
        } else {
            panic!("Accelerate AttentionScores should be Unsupported");
        }
    }

    #[test]
    fn support_matrix_pending_has_code_and_reason() {
        let matrix = coreai_support_matrix();
        if let Some(PhaseSupportStatus::Pending { code, reason }) =
            matrix.support_for(PipelinePhase::TokenEmbedding)
        {
            assert_eq!(*code, PendingCode::MilOpNotWired);
            assert!(!reason.is_empty(), "Pending reason must not be empty");
        } else {
            panic!("CoreML TokenEmbedding should be Pending");
        }
    }

    #[test]
    fn kv_phases_roundtrip_serde() {
        for phase in &[
            PipelinePhase::KvWrite,
            PipelinePhase::KvAppend,
            PipelinePhase::KvView,
        ] {
            let s = phase.to_string();
            let parsed: PipelinePhase = s
                .parse()
                .unwrap_or_else(|e| panic!("cannot parse '{s}': {e}"));
            assert_eq!(*phase, parsed, "roundtrip failed for '{s}'");
        }
    }

    #[test]
    fn all_phases_count_is_21() {
        assert_eq!(
            PipelinePhase::all().len(),
            21,
            "PipelinePhase::all() must have exactly 21 entries after KV phase addition"
        );
    }

    #[test]
    fn kv_contracts_have_all_phases() {
        let matrices = [
            coreai_support_matrix(),
            mlx_support_matrix(),
            accelerate_support_matrix(),
        ];
        for matrix in &matrices {
            for &phase in &[
                PipelinePhase::KvWrite,
                PipelinePhase::KvAppend,
                PipelinePhase::KvView,
            ] {
                let found = matrix.phases.iter().any(|(p, _)| *p == phase);
                assert!(
                    found,
                    "phase '{phase}' missing from {} support matrix",
                    matrix.backend
                );
            }
        }
    }

    #[test]
    fn kv_phases_display_snake_case() {
        for phase in &[
            PipelinePhase::KvWrite,
            PipelinePhase::KvAppend,
            PipelinePhase::KvView,
        ] {
            let s = phase.to_string();
            assert!(
                !s.contains(' '),
                "Display of '{phase:?}' contains whitespace: '{s}'"
            );
            assert!(
                s.chars().all(|c| c.is_ascii_lowercase() || c == '_'),
                "Display of '{:?}' has non-snake_case chars: '{s}'",
                phase
            );
        }
    }

    #[test]
    fn coreai_kv_all_unsupported() {
        let matrix = coreai_support_matrix();
        for &phase in &[
            PipelinePhase::KvWrite,
            PipelinePhase::KvAppend,
            PipelinePhase::KvView,
        ] {
            let status = matrix
                .support_for(phase)
                .expect("CoreML must have entry for {phase}");
            match status {
                PhaseSupportStatus::Unsupported { code, .. } => {
                    assert_eq!(
                        *code,
                        UnsupportedCode::StatefulBoundary,
                        "CoreML {phase} should be StatefulBoundary"
                    );
                }
                _ => panic!("CoreML {phase} should be Unsupported, got {status:?}"),
            }
        }
    }

    #[test]
    fn mlx_kv_all_composed() {
        let matrix = mlx_support_matrix();
        for &phase in &[
            PipelinePhase::KvWrite,
            PipelinePhase::KvAppend,
            PipelinePhase::KvView,
        ] {
            let status = matrix
                .support_for(phase)
                .expect("MLX must have entry for {phase}");
            assert_eq!(
                *status,
                PhaseSupportStatus::Composed,
                "MLX {phase} should be Composed"
            );
        }
    }

    #[test]
    fn accelerate_kv_all_unsupported() {
        let matrix = accelerate_support_matrix();
        for &phase in &[
            PipelinePhase::KvWrite,
            PipelinePhase::KvAppend,
            PipelinePhase::KvView,
        ] {
            let status = matrix
                .support_for(phase)
                .expect("Accelerate must have entry for {phase}");
            match status {
                PhaseSupportStatus::Unsupported { code, .. } => {
                    assert_eq!(
                        *code,
                        UnsupportedCode::StatefulBoundary,
                        "Accelerate {phase} should be StatefulBoundary"
                    );
                }
                _ => panic!("Accelerate {phase} should be Unsupported, got {status:?}"),
            }
        }
    }

    // ── BackendId ──────────────────────────────────────────────────────

    #[test]
    fn backend_id_roundtrip() {
        for id in &[
            BackendId::CoreAi,
            BackendId::Mlx,
            BackendId::Accelerate,
            BackendId::Reference,
        ] {
            let s = id.to_string();
            let parsed: BackendId = s.parse().expect("parse");
            assert_eq!(*id, parsed);
        }
    }

    #[test]
    fn backend_id_unknown_fails() {
        let err = "unknown-backend".parse::<BackendId>().unwrap_err();
        assert!(err.contains("unknown BackendId"));
    }

    // ── BTreeMap for canonical collections ────────────────────────────

    #[test]
    fn kv_phase_support_uses_btreemap() {
        let map = kv_phase_support_for(BackendId::Mlx);
        // BTreeMap iteration order is sorted by key (PipelinePhase enum order).
        let keys: Vec<_> = map.keys().copied().collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(
            keys, sorted,
            "kv_phase_support_for must return a BTreeMap (sorted iteration)"
        );
    }

    // ── ComparisonReceiptView + grouping ──────────────────────────────

    fn view_with_phase(
        phase: &str,
        backend: &str,
        variant: &str,
        semantic: &str,
    ) -> ComparisonReceiptView {
        ComparisonReceiptView {
            pipeline_phase: Some(phase.to_string()),
            phase_variant: variant.to_string(),
            semantic_contract_id: semantic.to_string(),
            shape_profile: "small".to_string(),
            dtype: "float32".to_string(),
            backend: backend.to_string(),
            backend_runtime_policy: format!("{backend}_default"),
            predict_status: "pass".to_string(),
            tolerance: 1e-3,
            input_shape: vec![1, 4],
            weight_shape: vec![4, 1],
        }
    }

    #[test]
    fn comparison_grouping_filters_empty_phase() {
        let mut r = view_with_phase(
            "qkv_projection",
            "mlx",
            "generic_projection",
            "qkv_projection/generic_projection",
        );
        r.pipeline_phase = None;
        let groups = group_for_comparison(&[r]);
        assert!(
            groups.is_empty(),
            "receipt with None pipeline_phase should be filtered out"
        );
    }

    #[test]
    fn comparison_grouping_requires_same_phase() {
        let r1 = view_with_phase(
            "qkv_projection",
            "mlx",
            "generic_projection",
            "qkv_projection/generic_projection",
        );
        let r2 = view_with_phase(
            "softmax",
            "accelerate",
            "softmax_after_matmul",
            "softmax/softmax_after_matmul",
        );
        let groups = group_for_comparison(&[r1, r2]);
        // Different semantic_contract_id -> different groups
        assert_eq!(
            groups.len(),
            2,
            "different phases should produce separate groups"
        );
    }

    #[test]
    fn comparison_grouping_requires_same_semantic_contract() {
        let r1 = view_with_phase("activation", "mlx", "silu", "activation/silu");
        let r2 = view_with_phase(
            "activation",
            "accelerate",
            "matmul_add_silu",
            "activation/matmul_add_silu",
        );
        let groups = group_for_comparison(&[r1, r2]);
        assert_eq!(
            groups.len(),
            2,
            "same phase but different semantic contract IDs should produce separate groups"
        );
    }

    #[test]
    fn comparison_grouping_merges_same_semantic_contract() {
        let r1 = view_with_phase(
            "qkv_projection",
            "mlx",
            "generic_projection",
            "qkv_projection/generic_projection",
        );
        let r2 = view_with_phase(
            "qkv_projection",
            "accelerate",
            "generic_projection",
            "qkv_projection/generic_projection",
        );
        let groups = group_for_comparison(&[r1, r2]);
        assert_eq!(
            groups.len(),
            1,
            "same phase + variant + shape should produce one group"
        );
        assert_eq!(
            groups[0].rows.len(),
            2,
            "both backends should appear in the same group"
        );
    }

    // ── tolerance profile classification ──────────────────────────────

    #[test]
    fn tolerance_profile_thresholds() {
        assert_eq!(PhaseComparisonGroup::tolerance_profile_for(1e-6), "strict");
        assert_eq!(PhaseComparisonGroup::tolerance_profile_for(1e-5), "strict");
        assert_eq!(
            PhaseComparisonGroup::tolerance_profile_for(1e-4),
            "standard"
        );
        assert_eq!(PhaseComparisonGroup::tolerance_profile_for(1e-3), "relaxed");
    }

    // ── is_fully_covered ───────────────────────────────────────────────

    #[test]
    fn reference_is_fully_covered() {
        let m = reference_support_matrix();
        assert!(m.is_fully_covered());
    }

    #[test]
    fn coreai_not_fully_covered() {
        let m = coreai_support_matrix();
        assert!(!m.is_fully_covered());
    }

    // ── PipelineParityError display ────────────────────────────────────

    #[test]
    fn pipeline_parity_error_display() {
        let err = PipelineParityError {
            family_name: "identity_passthrough".to_string(),
            reason: "harness control family, not an inference pipeline phase",
        };
        let s = err.to_string();
        assert!(s.contains("identity_passthrough"));
        assert!(s.contains("harness control"));
    }

    // ── Dim Display ────────────────────────────────────────────────────

    #[test]
    fn dim_display() {
        assert_eq!(Dim::Known(1).to_string(), "1");
        assert_eq!(Dim::Symbol("hidden_dim").to_string(), "{hidden_dim}");
        assert_eq!(Dim::Any.to_string(), "*");
    }
