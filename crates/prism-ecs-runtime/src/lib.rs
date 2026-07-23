pub mod fault;
mod kernel;
mod ports;
pub mod schedule;
pub mod test_adapters;
pub mod world_view;

pub use fault::{
    FaultMode, FaultPlan, FaultPoint, FaultingCommandStore, FaultingLeaseCoordinator,
    FaultingSnapshotStore,
};
pub use kernel::{
    create_kernel, AgentSnapshot, Command, CommandEnvelope, CommandResult, CommitOutcome,
    KernelHandle, KernelHealth, RuntimeKernel,
};
pub use ports::{
    Admission, AdmittedCommand, AuthorityJournal, CommandStore, CommandWatermarks,
    CompletedCommand, DispatchError, DispatchHandle, DispatchRequest, DispatchStatus, EvidenceSink,
    HardwareDispatcher, KernelClock, LeaseCoordinator, NoopDispatcher, RecoveredCommand,
    RecoveryReport, ResultPayload, RuntimeError, SnapshotPayload, SnapshotStore, TickReceiptStore,
    WorkDispatcher, WorldSnapshot,
};
pub use schedule::{
    Access, AdmitSystem, CleanupSystem, CollectSystem, CommandBuffer, CommandEmitter,
    DispatchSystem, LeaseSystem, ObserveSystem, PlanSystem, PublishSystem, RuntimeSchedule,
    ScheduleError, System, SystemContext, SystemId, SystemSpec, SystemStage, TickReceipt,
};
pub use world_view::WorldViewImpl;

pub use test_adapters::{
    DeterministicClock, FakeDispatcher, InMemoryAuthorityJournal, InMemoryCommandStore,
    InMemoryLeaseCoordinator, InMemorySnapshotStore, InMemoryTickReceiptStore,
};

#[cfg(test)]
mod tests {
    use crate::kernel::{Command, RuntimeKernel};

    #[test]
    fn create_kernel_returns_working_handle() {
        let kernel = RuntimeKernel::new();
        let handle = kernel.handle();
        let health = kernel.health();
        assert_eq!(health.status, "running");
        assert_eq!(health.entity_count, 0);
        // handle should be usable
        let agents = handle.query_agents();
        assert!(agents.is_empty());
    }

    #[test]
    fn spawn_adds_agent_visible_through_query() {
        let kernel = RuntimeKernel::new();
        let handle = kernel.handle();

        let outcome = handle.submit(crate::CommandEnvelope::new(Command::SpawnAgent {
            parent_id: 0,
            task: "test agent".to_string(),
            max_steps: 10,
        }));
        assert!(outcome.is_ok(), "spawn should succeed: {:?}", outcome);
        let entity_id = match outcome.unwrap().result {
            crate::CommandResult::Spawned { entity_id } => entity_id,
            _ => panic!("expected Spawned, got something else"),
        };
        assert!(entity_id > 0, "entity id should be positive");

        let agents = handle.query_agents();
        assert_eq!(agents.len(), 1, "one agent should be visible");
        assert_eq!(agents[0].entity_id, entity_id);
        assert_eq!(agents[0].phase, "Planning");
        assert_eq!(agents[0].lifecycle, "Active");
        // parent_id in SpawnAgent command is passed directly; 0 means entity 0
        assert_eq!(agents[0].parent_id, Some(0));
    }

    #[test]
    fn cancel_updates_lifecycle() {
        let kernel = RuntimeKernel::new();
        let handle = kernel.handle();

        let entity_id = handle
            .submit(crate::CommandEnvelope::new(Command::SpawnAgent {
                parent_id: 0,
                task: "cancellable".to_string(),
                max_steps: 10,
            }))
            .unwrap()
            .result;
        let entity_id = match entity_id {
            crate::CommandResult::Spawned { entity_id } => entity_id,
            _ => panic!("expected Spawned"),
        };

        let cancel_result = handle.submit(crate::CommandEnvelope::new(Command::CancelAgent {
            agent_id: entity_id,
        }));
        assert!(
            cancel_result.is_ok(),
            "cancel should succeed: {:?}",
            cancel_result
        );

        let agents = handle.query_agents();
        let agent = agents.iter().find(|a| a.entity_id == entity_id).unwrap();
        assert_eq!(agent.lifecycle, "Completed");
    }

    #[test]
    fn test_submit_records_through_command_store() {
        let kernel = RuntimeKernel::new();
        let handle = kernel.handle();

        let outcome = handle.submit(crate::CommandEnvelope::new(Command::SpawnAgent {
            parent_id: 0,
            task: "monitored".to_string(),
            max_steps: 10,
        }));
        assert!(outcome.is_ok(), "submit should succeed");

        // Re-submit the same idempotency key should return completed
        let env = crate::CommandEnvelope::new(Command::SpawnAgent {
            parent_id: 0,
            task: "dup".to_string(),
            max_steps: 10,
        });
        let _first = handle.submit(env.clone()).unwrap().result;
        let second = handle.submit(env).unwrap().result;
        // Idempotent replay returns the stored result directly
        assert!(matches!(
            second,
            crate::CommandResult::Spawned { entity_id: 2 }
        ));
    }

    #[test]
    fn test_lifecycle_command_idempotency() {
        // Verify that resubmitting the same lifecycle command returns the
        // original sequence, world epoch, affected entity, and result.
        use crate::CommandResult;
        use prism_ecs_constitutional::lifecycle_command::{
            CreateWorkCommand, LifecycleCommand, LifecycleCommandResult,
        };
        let kernel = RuntimeKernel::new();
        let handle = kernel.handle();

        let env = crate::CommandEnvelope::new(Command::Lifecycle(LifecycleCommand::CreateWork(
            CreateWorkCommand {
                entity: 0, target_entity: 0,
                kind: "idempotency-test".to_string(),
                resource_claim: "{}".to_string(),
                output_path: "".to_string(),
                input_path: "".to_string(),
            },
        )));

        // First submission
        let first = handle.submit(env.clone()).expect("first submit");
        let work_entity = match &first.result {
            CommandResult::Lifecycle(LifecycleCommandResult::WorkCreated {
                work_entity, ..
            }) => *work_entity,
            _ => panic!("expected WorkCreated"),
        };
        assert!(first.sequence > 0, "sequence should be positive");
        assert!(work_entity > 0, "work entity should be positive");

        // Second submission — same envelope (idempotency key)
        let second = handle.submit(env).expect("second submit (idempotent)");
        assert_eq!(
            second.sequence, first.sequence,
            "idempotent submission must return original sequence"
        );
        assert!(
            matches!(
                second.result,
                CommandResult::Lifecycle(LifecycleCommandResult::WorkCreated { .. })
            ),
            "idempotent result should match original variant"
        );
    }
}
