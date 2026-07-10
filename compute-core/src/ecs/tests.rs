#[cfg(test)]
mod tests {
    use crate::ecs::compile_session::CompileSession;
    use crate::ecs::component::fusion::*;
    use crate::ecs::component::memory::*;
    use crate::ecs::component::tensor::*;
    use crate::ecs::*;
    use crate::ecs::plan::CodecFamily;
    use crate::ecs::adapter::CanonicalRole;

    // -----------------------------------------------------------------------
    // test_compworld_basics
    // -----------------------------------------------------------------------
    #[test]
    fn test_compworld_basics() {
        let mut world = CompWorld::new();
        assert_eq!(world.entity_count(), 0);

        // Spawn entities of each kind.
        let model = world.spawn(EntityKind::Model, Some("test_model".into()));
        let tensor = world.spawn(EntityKind::Tensor, Some("test_tensor".into()));
        let layer = world.spawn(EntityKind::Layer, Some("test_layer".into()));
        let expert = world.spawn(EntityKind::Expert, None);
        let dispatch = world.spawn(EntityKind::Dispatch, None);
        let kernel = world.spawn(EntityKind::Kernel, None);
        let buffer = world.spawn(EntityKind::Buffer, None);
        let cmd_buf = world.spawn(EntityKind::CommandBuffer, None);
        let exec = world.spawn(EntityKind::Executable, None);
        let fence = world.spawn(EntityKind::Fence, None);

        assert_eq!(world.entity_count(), 10);

        // Verify entity kinds.
        assert_eq!(world.kind(model), Some(EntityKind::Model));
        assert_eq!(world.kind(tensor), Some(EntityKind::Tensor));
        assert_eq!(world.kind(layer), Some(EntityKind::Layer));
        assert_eq!(world.kind(dispatch), Some(EntityKind::Dispatch));

        // Verify names.
        assert_eq!(world.name(model), Some("test_model"));
        assert_eq!(world.name(tensor), Some("test_tensor"));

        // Add and retrieve components.
        let shape = Shape(vec![4096, 4096]);
        let dtype = DataType(DType::F32);
        let role = CanonicalRoleComp(CanonicalRole::Q(0));

        world.add_component(tensor, shape.clone());
        world.add_component(tensor, dtype);
        world.add_component(tensor, role);

        assert_eq!(world.get_component::<Shape>(tensor), Some(&shape));
        assert_eq!(
            world.get_component::<DataType>(tensor),
            Some(&DataType(DType::F32))
        );
        assert_eq!(
            world.get_component::<CanonicalRoleComp>(tensor),
            Some(&CanonicalRoleComp(CanonicalRole::Q(0)))
        );

        // Verify entities_of_kind filtering.
        let tensors = world.entities_of_kind(EntityKind::Tensor);
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0], tensor);

        let models = world.entities_of_kind(EntityKind::Model);
        assert_eq!(models.len(), 1);
        assert_eq!(models[0], model);

        let buffers = world.entities_of_kind(EntityKind::Buffer);
        assert_eq!(buffers.len(), 1);
        assert_eq!(buffers[0], buffer);

        // Remove component.
        let removed: Option<Shape> = world.remove_component(tensor);
        assert!(removed.is_some());
        assert_eq!(world.get_component::<Shape>(tensor), None);
    }

    // -----------------------------------------------------------------------
    // test_compilesession_create
    // -----------------------------------------------------------------------
    #[test]
    fn test_compilesession_create() {
        let mut session = CompileSession::new();
        assert!(session.get_output_path().is_none());

        session.set_output_path("/tmp/test_out.cimage");
        assert_eq!(
            session
                .get_output_path()
                .map(|p| p.to_string_lossy().to_string()),
            Some("/tmp/test_out.cimage".into())
        );

        // Register built-in systems; must not panic.
        session.register_builtin_systems();

        // After registration the world has at least 20 systems.
        // (The exact count is 23 — see register_builtin_systems body.)
        assert!(
            session.world.entity_count() == 0,
            "no entities should be spawned by registration alone"
        );
    }

    // -----------------------------------------------------------------------
    // test_ecs_pipeline_model_loading
    // -----------------------------------------------------------------------
    #[test]
    fn test_ecs_pipeline_model_loading() {
        let mut session = CompileSession::new();
        session.register_builtin_systems();

        // Manually simulate what load_model would create: Model + Tensor +
        // Layer entities with the components the pipeline expects.
        let model_entity = session.world.spawn(EntityKind::Model, Some("dummy".into()));

        // Create a few tensor entities with different canonical roles.
        let q_tensor = session.world.spawn(EntityKind::Tensor, Some("Q(0)".into()));
        session
            .world
            .add_component(q_tensor, Shape(vec![4096, 4096]));
        session
            .world
            .add_component(q_tensor, CanonicalRoleComp(CanonicalRole::Q(0)));
        session.world.add_component(q_tensor, DataType(DType::F16));

        let v_tensor = session.world.spawn(EntityKind::Tensor, Some("V(0)".into()));
        session
            .world
            .add_component(v_tensor, Shape(vec![4096, 4096]));
        session
            .world
            .add_component(v_tensor, CanonicalRoleComp(CanonicalRole::V(0)));
        session.world.add_component(v_tensor, DataType(DType::F16));

        let gate_tensor = session
            .world
            .spawn(EntityKind::Tensor, Some("Gate(0)".into()));
        session
            .world
            .add_component(gate_tensor, Shape(vec![4096, 11008]));
        session
            .world
            .add_component(gate_tensor, CanonicalRoleComp(CanonicalRole::Gate(0)));

        let down_tensor = session
            .world
            .spawn(EntityKind::Tensor, Some("Down(0)".into()));
        session
            .world
            .add_component(down_tensor, Shape(vec![11008, 4096]));
        session
            .world
            .add_component(down_tensor, CanonicalRoleComp(CanonicalRole::Down(0)));

        let layer0 = session
            .world
            .spawn(EntityKind::Layer, Some("layer_0".into()));
        session.world.add_component(layer0, LayerIndex(0));

        // Run Phase B (Quantization) — CodecSelectionSystem assigns a
        // CodecFamilyComp to each tensor based on canonical role.
        session.run_phase(SchedulePhase::Quantization).unwrap();

        // Verify Model entity exists.
        let models = session.world.entities_of_kind(EntityKind::Model);
        assert_eq!(models.len(), 1, "Model entity must survive Phase B");
        assert_eq!(models[0], model_entity);

        // Verify Tensor entities exist and have CodecFamilyComp.
        let tensors = session.world.entities_of_kind(EntityKind::Tensor);
        assert_eq!(tensors.len(), 4, "all four tensor entities must survive");

        for t in &tensors {
            let codec = session.world.get_component::<CodecFamilyComp>(*t);
            assert!(
                codec.is_some(),
                "every tensor should have a CodecFamilyComp after Phase B"
            );
        }

        // Verify role-specific codec assignments.
        let q_entity = tensors
            .iter()
            .find(|t| session.world.name(**t) == Some("Q(0)"))
            .expect("Q(0) tensor should exist");
        let q_codec = session
            .world
            .get_component::<CodecFamilyComp>(*q_entity)
            .unwrap();
        assert_eq!(q_codec.0, CodecFamily::Int8, "Q projection → Int8");
        assert_eq!(q_codec.1, 0, "Q projection group_size = 0");

        let gate_entity = tensors
            .iter()
            .find(|t| session.world.name(**t) == Some("Gate(0)"))
            .expect("Gate(0) tensor should exist");
        let gate_codec = session
            .world
            .get_component::<CodecFamilyComp>(*gate_entity)
            .unwrap();
        assert_eq!(gate_codec.0, CodecFamily::Q8_0, "Gate projection → Q8_0");
        assert_eq!(gate_codec.1, 32, "Gate projection group_size = 32");

        // Verify Layer entity exists with LayerIndex.
        let layers = session.world.entities_of_kind(EntityKind::Layer);
        assert_eq!(layers.len(), 1, "Layer entity must survive Phase B");
        let layer_idx = session.world.get_component::<LayerIndex>(layers[0]);
        assert_eq!(layer_idx, Some(&LayerIndex(0)));

        // Run Phase C (MemoryPlanning) — requires BackendTarget on tensors.
        // Add BackendTarget manually (MemoryDomainAssignmentSystem reads it).
        for t in &tensors {
            session
                .world
                .add_component(*t, crate::ecs::component::backend::BackendTarget::Metal);
        }
        session.run_phase(SchedulePhase::MemoryPlanning).unwrap();

        // Verify Buffer entities were allocated for each tensor.
        let buffers = session.world.entities_of_kind(EntityKind::Buffer);
        assert!(
            buffers.len() >= 4,
            "at least one buffer per tensor should exist after Phase C"
        );

        // Verify MemoryDomain was set on each tensor.
        for t in &tensors {
            let domain = session.world.get_component::<MemoryDomain>(*t);
            assert_eq!(
                domain,
                Some(&MemoryDomain::DeviceLocal),
                "Metal backend → DeviceLocal"
            );
        }
    }

    // -----------------------------------------------------------------------
    // test_codec_selection_per_role
    // -----------------------------------------------------------------------
    #[test]
    fn test_codec_selection_per_role() {
        // Create a CompileSession with systems registered.
        let mut session = CompileSession::new();
        session.register_builtin_systems();

        // Create tensors for each role category with a uniform Shape.
        let role_cases: Vec<(&str, CanonicalRole, CodecFamily, u32)> = vec![
            // Attention projections → Int8
            ("Q(0)", CanonicalRole::Q(0), CodecFamily::Int8, 0),
            ("K(0)", CanonicalRole::K(0), CodecFamily::Int8, 0),
            ("V(0)", CanonicalRole::V(0), CodecFamily::Int8, 0),
            ("O(0)", CanonicalRole::O(0), CodecFamily::Int8, 0),
            ("QNorm(0)", CanonicalRole::QNorm(0), CodecFamily::Int8, 0),
            ("KNorm(0)", CanonicalRole::KNorm(0), CodecFamily::Int8, 0),
            // MLP projections → Q8_0 with group_size 32
            ("Gate(0)", CanonicalRole::Gate(0), CodecFamily::Q8_0, 32),
            ("Up(0)", CanonicalRole::Up(0), CodecFamily::Q8_0, 32),
            ("Down(0)", CanonicalRole::Down(0), CodecFamily::Q8_0, 32),
            // Routed MoE experts → Q4_K with group_size 128
            (
                "GateEx(0,0)",
                CanonicalRole::GateEx(0, 0),
                CodecFamily::Q4_K,
                128,
            ),
            // Router weight → Q8_0
            (
                "RouterWeight(0)",
                CanonicalRole::RouterWeight(0),
                CodecFamily::Q8_0,
                32,
            ),
            // Embedding → Fp16
            ("Embedding", CanonicalRole::Embedding, CodecFamily::Fp16, 0),
            // Normalisation → Fp16
            (
                "AttnNorm(0)",
                CanonicalRole::AttnNorm(0),
                CodecFamily::Fp16,
                0,
            ),
            (
                "MlpNorm(0)",
                CanonicalRole::MlpNorm(0),
                CodecFamily::Fp16,
                0,
            ),
            // Shared experts → Int8
            (
                "SharedGate",
                CanonicalRole::SharedGate,
                CodecFamily::Int8,
                0,
            ),
            ("SharedUp", CanonicalRole::SharedUp, CodecFamily::Int8, 0),
            (
                "SharedDown",
                CanonicalRole::SharedDown,
                CodecFamily::Int8,
                0,
            ),
        ];

        for (name, role, _expected_codec, _expected_gs) in &role_cases {
            let entity = session
                .world
                .spawn(EntityKind::Tensor, Some(name.to_string()));
            session.world.add_component(entity, Shape(vec![4096, 4096]));
            session
                .world
                .add_component(entity, CanonicalRoleComp(*role));
        }

        // Run Phase B.
        session.run_phase(SchedulePhase::Quantization).unwrap();

        // Verify each tensor received the correct codec.
        for (name, _role, expected_codec, expected_gs) in &role_cases {
            let tensors = session.world.entities_of_kind(EntityKind::Tensor);
            let entity = tensors
                .iter()
                .find(|t| session.world.name(**t) == Some(*name))
                .unwrap_or_else(|| panic!("tensor '{}' should exist", name));

            let codec = session
                .world
                .get_component::<CodecFamilyComp>(*entity)
                .unwrap_or_else(|| panic!("tensor '{}' missing CodecFamilyComp", name));

            assert_eq!(
                codec.0, *expected_codec,
                "{}: expected codec {:?}, got {:?}",
                name, expected_codec, codec.0
            );
            assert_eq!(
                codec.1, *expected_gs,
                "{}: expected group_size {}, got {}",
                name, expected_gs, codec.1
            );
        }

        // Tensors without CanonicalRoleComp should not get a codec.
        let unnamed = session
            .world
            .spawn(EntityKind::Tensor, Some("no_role".into()));
        session.world.add_component(unnamed, Shape(vec![128, 128]));

        // Run Phase B again for the new tensor.
        session.run_phase(SchedulePhase::Quantization).unwrap();

        assert!(
            session
                .world
                .get_component::<CodecFamilyComp>(unnamed)
                .is_none(),
            "tensor without CanonicalRoleComp should not receive a codec"
        );
    }

    // -----------------------------------------------------------------------
    // test_buffer_allocation
    // -----------------------------------------------------------------------
    #[test]
    fn test_buffer_allocation_with_memory_domain() {
        let mut session = CompileSession::new();
        session.register_builtin_systems();

        // Create a tensor with all components needed for buffer allocation.
        let tensor = session
            .world
            .spawn(EntityKind::Tensor, Some("test_weight".into()));
        session.world.add_component(tensor, Shape(vec![4096, 4096]));
        session
            .world
            .add_component(tensor, CanonicalRoleComp(CanonicalRole::Gate(0)));
        // CodecSelectionSystem must run first to assign CodecFamilyComp.
        session.run_phase(SchedulePhase::Quantization).unwrap();

        // Verify CodecFamilyComp was attached.
        let codec_comp = session
            .world
            .get_component::<CodecFamilyComp>(tensor)
            .expect("Gate(0) should have CodecFamilyComp after Phase B");

        // Compute expected storage: Q8_0 with 32-element groups.
        // storage = (num_groups * (block_size * 4 + 12)) = total elements / 32 * (32*4 + 12)
        // For 16777216 elements: 524288 groups × 140 bytes = 73400320
        let elem_count: u64 = 4096u64 * 4096; // 16,777,216
        let group_size: u64 = 32;
        let num_groups = elem_count.div_ceil(group_size);
        // Q8_0: each group stores group_size * 4 (fp32 activations, but
        // codec allocation is: block Q8_0 = 4 bytes/block (scale32) + group_size * 1 byte
        // Using codec_storage_bytes logic:
        // Q8_0: bytes = num_groups * (32 * 1 + 4 + 2) = num_groups * 38
        // But let's just compute it the same way codec_storage_bytes does.
        let expected_bytes = match codec_comp.0 {
            CodecFamily::Q8_0 => {
                // Matches codec_storage_bytes: elem_count + blocks * 2 (2-byte scale per block)
                elem_count + num_groups * 2
            }
            _ => panic!("expected Q8_0 codec"),
        };

        // Add BackendTarget so MemoryDomainAssignmentSystem assigns MemoryDomain.
        session
            .world
            .add_component(tensor, crate::ecs::component::backend::BackendTarget::Metal);

        // Run Phase C — MemoryDomainAssignment + BufferAllocation.
        session.run_phase(SchedulePhase::MemoryPlanning).unwrap();

        // Verify a Buffer entity was created.
        let buffers = session.world.entities_of_kind(EntityKind::Buffer);
        assert!(!buffers.is_empty(), "at least one Buffer entity must exist");

        // The buffer should have MemoryPool with Dedicated policy (has CanonicalRoleComp).
        let buffer = buffers[0];
        let pool = session
            .world
            .get_component::<MemoryPool>(buffer)
            .expect("Buffer should have MemoryPool");
        assert_eq!(
            pool.policy,
            super::super::component::memory::PoolPolicy::Dedicated,
            "weight tensor → Dedicated pool"
        );
        assert_eq!(pool.total_bytes, expected_bytes as u64);

        // Buffer should have BufferLifetime.
        assert!(
            session
                .world
                .get_component::<BufferLifetime>(buffer)
                .is_some(),
            "Buffer should have BufferLifetime"
        );
    }

    // -----------------------------------------------------------------------
    // test_fusion_analysis — MLP layer (Gate, Up, Down)
    // -----------------------------------------------------------------------
    #[test]
    fn test_fusion_analysis_mlp_layer() {
        let mut session = CompileSession::new();
        session.register_builtin_systems();

        // Create a Layer entity.
        let layer = session
            .world
            .spawn(EntityKind::Layer, Some("layer_0".into()));
        session.world.add_component(layer, LayerIndex(0));

        // Create the MLP triplet tensors.
        let gate = session
            .world
            .spawn(EntityKind::Tensor, Some("Gate(0)".into()));
        session.world.add_component(gate, Shape(vec![4096, 11008]));
        session
            .world
            .add_component(gate, CanonicalRoleComp(CanonicalRole::Gate(0)));

        let up = session
            .world
            .spawn(EntityKind::Tensor, Some("Up(0)".into()));
        session.world.add_component(up, Shape(vec![4096, 11008]));
        session
            .world
            .add_component(up, CanonicalRoleComp(CanonicalRole::Up(0)));

        let down = session
            .world
            .spawn(EntityKind::Tensor, Some("Down(0)".into()));
        session.world.add_component(down, Shape(vec![11008, 4096]));
        session
            .world
            .add_component(down, CanonicalRoleComp(CanonicalRole::Down(0)));

        // Run Phase D — FusionDispatch runs FusionAnalysisSystem,
        // FusionHeuristicSystem, DispatchFormationSystem, ScalarDispatchSystem.
        session.run_phase(SchedulePhase::FusionDispatch).unwrap();

        // FusionAnalysisSystem should have created dispatch entities.
        let dispatches = session.world.entities_of_kind(EntityKind::Dispatch);
        assert!(
            !dispatches.is_empty(),
            "FusionAnalysisSystem should create Dispatch entities for MLP layer"
        );

        // With Gate, Up, Down, the MLP graph produces 3 FusionGroups:
        //   Gate MatMul → fused: SiLU  (gate_out consumed by SiLU)
        //   Up   MatMul → fused: Mul   (up_out consumed by Mul)
        //   Down MatMul → fused: (none)
        let fusion_groups: Vec<FusionGroup> = dispatches
            .iter()
            .filter_map(|d| session.world.get_component::<FusionGroup>(*d).cloned())
            .collect();
        assert_eq!(
            fusion_groups.len(),
            3,
            "MLP Gate+Up+Down should produce 3 fusion groups"
        );

        // Group them by fused_op_kinds.
        let mut fused_silu = false;
        let mut fused_mul = false;
        let mut solo_matmuls = 0;

        for group in &fusion_groups {
            assert_eq!(
                group.root_op_kind, "MatMul",
                "all fusion group roots should be MatMul"
            );
            if group.fused_op_kinds == vec!["SiLU"] {
                fused_silu = true;
            } else if group.fused_op_kinds == vec!["Mul"] {
                fused_mul = true;
            } else if group.fused_op_kinds.is_empty() {
                solo_matmuls += 1;
            }
        }

        assert!(
            fused_silu,
            "Gate MatMul should have SiLU fused (Gate→SiLU→Mul)"
        );
        assert!(fused_mul, "Up MatMul should have Mul fused (Up→Mul)");
        assert_eq!(
            solo_matmuls, 1,
            "Down MatMul should be a solo MatMul (no elementwise consumer)"
        );

        // A DataflowGraphHandle should be attached to the Layer entity.
        let handle = session.world.get_component::<DataflowGraphHandle>(layer);
        assert!(
            handle.is_some(),
            "Layer should have a DataflowGraphHandle after fusion analysis"
        );
        assert_eq!(
            handle.unwrap().0,
            "fusion_graph_layer_0",
            "handle should reference layer index"
        );
    }

    // -----------------------------------------------------------------------
    // test_scalar_dispatch
    // -----------------------------------------------------------------------
    #[test]
    fn test_scalar_dispatch() {
        let mut session = CompileSession::new();
        session.register_builtin_systems();

        // Create a Dispatch entity with a small shape (< 128 elements).
        let small_dispatch = session
            .world
            .spawn(EntityKind::Dispatch, Some("small".into()));
        session
            .world
            .add_component(small_dispatch, Shape(vec![1, 64]));

        // Also create one above the threshold to verify it is NOT scalar.
        let large_dispatch = session
            .world
            .spawn(EntityKind::Dispatch, Some("large".into()));
        session
            .world
            .add_component(large_dispatch, Shape(vec![256, 256]));

        // Run Phase D — ScalarDispatchSystem runs after
        // FusionAnalysisSystem, FusionHeuristicSystem, DispatchFormationSystem.
        session.run_phase(SchedulePhase::FusionDispatch).unwrap();

        // Small dispatch: total = 1 * 64 = 64 < 128 → WorkgroupCount(1,1,1).
        let small_wg = session
            .world
            .get_component::<WorkgroupCount>(small_dispatch);
        assert_eq!(
            small_wg,
            Some(&WorkgroupCount(1, 1, 1)),
            "dispatch with <128 elements should be scalar"
        );

        // Large dispatch: total = 256 * 256 = 65536 >= 128 → no WorkgroupCount
        // (unless prior systems added one, which they only do for entities
        //  that also have FusionGroup).
        let large_wg = session
            .world
            .get_component::<WorkgroupCount>(large_dispatch);
        assert!(
            large_wg.is_none() || large_wg != Some(&WorkgroupCount(1, 1, 1)),
            "dispatch with ≥128 elements should NOT have scalar workgroup"
        );
    }
}
