//! This module owns the canonical structural validator that runs all ten
//! admission gates against an assistant graph manifest and produces a
//! validation receipt.

use std::collections::{HashMap, HashSet};

use super::bridge::BridgeValueType;
use super::manifest::{
    AssistantGraphManifest, AssistantRegionDecl, AssistantRegionKind, RegionOutputAuthority,
};
use super::receipts::{AssistantGraphValidationReceipt, AssistantGraphValidationStatus};
use super::state::StateStoreKind;

pub struct AssistantGraphValidator;

impl AssistantGraphValidator {
    pub fn validate(graph: &AssistantGraphManifest) -> AssistantGraphValidationReceipt {
        let mut errors: Vec<String> = Vec::new();
        let mut warnings: Vec<String> = Vec::new();

        // Gate 1: Only one region may emit CommittedAssistantResponse per phase
        Self::gate_only_one_committed_response(graph, &mut errors);

        // Gate 2: TTS/audio region must not mutate semantic response state
        Self::gate_tts_no_semantic_mutation(graph, &mut errors);

        // Gate 3: Vision region must not emit committed user-facing text
        Self::gate_vision_no_committed_text(graph, &mut errors);

        // Gate 4: Embedding region must not write assistant identity/style state
        Self::gate_embedding_no_identity_write(graph, &mut errors);

        // Gate 5: Two regions must not write the same shared state path without arbitration
        Self::gate_no_conflicting_writes(graph, &mut errors);

        // Gate 6: Route must not consume DraftText where CommittedAssistantResponse is required
        Self::gate_route_draft_text_mismatch(graph, &mut errors);

        // Gate 7: Contract checks
        Self::gate_contract_checks(graph, &mut errors, &mut warnings);

        // Gate 8: All bridge source/target regions exist
        Self::gate_bridge_regions_exist(graph, &mut errors);

        // Gate 9: All region input/output types resolve to declared bridge types
        Self::gate_region_types_resolve(graph, &mut errors);

        // Gate 10: No duplicate bridge IDs
        Self::gate_no_duplicate_bridge_ids(graph, &mut errors);

        // Determine status
        let contract_valid = errors.is_empty();
        let region_count = graph.regions.len() as u32;
        let bridge_count = graph.bridges.len() as u32;
        let route_edges = graph.route_graph.edges.len() as u32;

        let validation_status = if !errors.is_empty() {
            AssistantGraphValidationStatus::Invalid
        } else if !warnings.is_empty() {
            AssistantGraphValidationStatus::ValidWithWarnings
        } else {
            AssistantGraphValidationStatus::Valid
        };

        AssistantGraphValidationReceipt {
            graph_id: graph.graph_id.clone(),
            contract_valid,
            region_count,
            bridge_count,
            route_edges,
            errors,
            warnings,
            validation_status,
        }
    }

    /// Gate 1: Reject if more than one region can emit CommittedAssistantResponse
    /// in the same phase (same partition_id).
    fn gate_only_one_committed_response(graph: &AssistantGraphManifest, errors: &mut Vec<String>) {
        // Group regions by partition_id (None counts as one group)
        let mut by_partition: HashMap<Option<&String>, Vec<&AssistantRegionDecl>> = HashMap::new();
        for region in &graph.regions {
            by_partition
                .entry(region.partition_id.as_ref())
                .or_default()
                .push(region);
        }

        for (partition, regions) in &by_partition {
            let committed: Vec<&str> = regions
                .iter()
                .filter(|r| {
                    r.authority
                        .contains(&RegionOutputAuthority::CommittedAssistantResponse)
                })
                .map(|r| r.region_id.as_str())
                .collect();

            if committed.len() > 1 {
                let phase = match partition {
                    Some(id) => format!("partition {}", id),
                    None => "global phase".to_string(),
                };
                errors.push(format!(
                    "Gate 1: More than one region can emit CommittedAssistantResponse in {}: {}",
                    phase,
                    committed.join(", ")
                ));
            }
        }
    }

    /// Gate 2: Reject if a TTS/audio region can mutate semantic response state.
    fn gate_tts_no_semantic_mutation(graph: &AssistantGraphManifest, errors: &mut Vec<String>) {
        let store_map: HashMap<&str, &StateStoreKind> = graph
            .shared_state_schema
            .stores
            .iter()
            .map(|s| (s.store_id.as_str(), &s.store_kind))
            .collect();

        for region in &graph.regions {
            if region.region_kind != AssistantRegionKind::SpeechSynthesis {
                continue;
            }
            for write_store in &region.state_access.write_stores {
                if let Some(StateStoreKind::AssistantResponseState) =
                    store_map.get(write_store.as_str())
                {
                    errors.push(format!(
                        "Gate 2: SpeechSynthesis region '{}' mutates semantic response state via store '{}'",
                        region.region_id, write_store
                    ));
                }
            }
        }
    }

    /// Gate 3: Reject if a vision region can emit committed user-facing text.
    fn gate_vision_no_committed_text(graph: &AssistantGraphManifest, errors: &mut Vec<String>) {
        for region in &graph.regions {
            if region.region_kind != AssistantRegionKind::VisionPerception {
                continue;
            }
            if region
                .authority
                .contains(&RegionOutputAuthority::CommittedAssistantResponse)
            {
                errors.push(format!(
                    "Gate 3: VisionPerception region '{}' may emit CommittedAssistantResponse",
                    region.region_id
                ));
            }
        }
    }

    /// Gate 4: Reject if an embedding region can write assistant identity/style state.
    fn gate_embedding_no_identity_write(
        graph: &AssistantGraphManifest,
        errors: &mut Vec<String>,
    ) {
        let store_map: HashMap<&str, &StateStoreKind> = graph
            .shared_state_schema
            .stores
            .iter()
            .map(|s| (s.store_id.as_str(), &s.store_kind))
            .collect();

        for region in &graph.regions {
            if region.region_kind != AssistantRegionKind::EmbeddingRetrieval {
                continue;
            }
            for write_store in &region.state_access.write_stores {
                if let Some(StateStoreKind::AssistantResponseState) =
                    store_map.get(write_store.as_str())
                {
                    errors.push(format!(
                        "Gate 4: EmbeddingRetrieval region '{}' writes assistant identity/\
                         style state via store '{}'",
                        region.region_id, write_store
                    ));
                }
            }
        }
    }

    /// Gate 5: Reject if two active regions write the same shared state path
    /// without arbitration (same write_store_id from different regions).
    fn gate_no_conflicting_writes(graph: &AssistantGraphManifest, errors: &mut Vec<String>) {
        // Build writer -> store_ids
        let mut store_writers: HashMap<&str, Vec<&str>> = HashMap::new();
        for region in &graph.regions {
            for write_store in &region.state_access.write_stores {
                store_writers
                    .entry(write_store.as_str())
                    .or_default()
                    .push(region.region_id.as_str());
            }
        }

        for (store_id, writers) in &store_writers {
            if writers.len() > 1 {
                errors.push(format!(
                    "Gate 5: Multiple regions write the same shared state store '{}' \
                     without arbitration: {}",
                    store_id,
                    writers.join(", ")
                ));
            }
        }
    }

    /// Gate 6: Reject if a route consumes DraftText where CommittedAssistantResponse
    /// is required (target region has CommittedAssistantResponse authority but the
    /// edge allows DraftText).
    fn gate_route_draft_text_mismatch(
        graph: &AssistantGraphManifest,
        errors: &mut Vec<String>,
    ) {
        // Build region_id -> region map
        let region_map: HashMap<&str, &AssistantRegionDecl> = graph
            .regions
            .iter()
            .map(|r| (r.region_id.as_str(), r))
            .collect();

        for edge in &graph.route_graph.edges {
            if !edge
                .allowed_types
                .contains(&BridgeValueType::DraftReasoningTrace)
            {
                continue;
            }
            if let Some(target) = region_map.get(edge.to_region.as_str()) {
                if target
                    .authority
                    .contains(&RegionOutputAuthority::CommittedAssistantResponse)
                {
                    errors.push(format!(
                        "Gate 6: Route edge from '{}' to '{}' allows DraftReasoningTrace \
                         but target region requires CommittedAssistantResponse",
                        edge.from_region, edge.to_region
                    ));
                }
            }
        }
    }

    /// Gate 7: Contract checks — max_active_regions, requires_bridge_types, requires_authority.
    fn gate_contract_checks(
        graph: &AssistantGraphManifest,
        errors: &mut Vec<String>,
        warnings: &mut Vec<String>,
    ) {
        let contract = &graph.assistant_contract;

        // 7a: max_active_regions
        if contract.max_active_regions > 0
            && graph.regions.len() as u32 > contract.max_active_regions
        {
            errors.push(format!(
                "Gate 7a: Contract max_active_regions={} but graph has {} regions",
                contract.max_active_regions,
                graph.regions.len()
            ));
        }

        // 7b: requires_bridge_types
        if contract.requires_bridge_types && graph.bridges.is_empty() {
            errors.push(
                "Gate 7b: Contract requires_bridge_types but no bridges are declared".to_string(),
            );
        }

        // 7c: requires_authority
        if contract.requires_authority {
            let missing_authority: Vec<&str> = graph
                .regions
                .iter()
                .filter(|r| r.authority.is_empty())
                .map(|r| r.region_id.as_str())
                .collect();
            if !missing_authority.is_empty() {
                errors.push(format!(
                    "Gate 7c: Contract requires_authority but regions lack authority: {}",
                    missing_authority.join(", ")
                ));
            }
        }

        // Warnings
        if contract.max_active_regions == 0 {
            warnings.push("Contract max_active_regions is 0 (unbounded)".to_string());
        }
        if contract.requires_bridge_types && graph.bridges.is_empty() {
            warnings.push(
                "Contract requires_bridge_types enabled but bridges list is empty".to_string(),
            );
        }
        if contract.requires_authority && graph.authority_policy.rules.is_empty() {
            warnings.push(
                "Contract requires_authority enabled but authority_policy has no rules"
                    .to_string(),
            );
        }
    }

    /// Gate 8: All bridge source/target regions exist in the regions list.
    fn gate_bridge_regions_exist(graph: &AssistantGraphManifest, errors: &mut Vec<String>) {
        let region_ids: HashSet<&str> =
            graph.regions.iter().map(|r| r.region_id.as_str()).collect();

        for bridge in &graph.bridges {
            if !region_ids.contains(bridge.source_region.as_str()) {
                errors.push(format!(
                    "Gate 8: Bridge '{}' references unknown source region '{}'",
                    bridge.bridge_id, bridge.source_region
                ));
            }
            if !region_ids.contains(bridge.target_region.as_str()) {
                errors.push(format!(
                    "Gate 8: Bridge '{}' references unknown target region '{}'",
                    bridge.bridge_id, bridge.target_region
                ));
            }
        }
    }

    /// Gate 9: All region input/output types must resolve to at least one bridge
    /// of that value type.
    fn gate_region_types_resolve(graph: &AssistantGraphManifest, errors: &mut Vec<String>) {
        let bridge_types: HashSet<BridgeValueType> =
            graph.bridges.iter().map(|b| b.value_type.clone()).collect();

        for region in &graph.regions {
            for input_type in &region.input_types {
                if !bridge_types.contains(input_type) {
                    errors.push(format!(
                        "Gate 9: Region '{}' has input type {:?} with no matching bridge",
                        region.region_id, input_type
                    ));
                }
            }
            for output_type in &region.output_types {
                if !bridge_types.contains(output_type) {
                    errors.push(format!(
                        "Gate 9: Region '{}' has output type {:?} with no matching bridge",
                        region.region_id, output_type
                    ));
                }
            }
        }
    }

    /// Gate 10: No duplicate bridge IDs.
    fn gate_no_duplicate_bridge_ids(graph: &AssistantGraphManifest, errors: &mut Vec<String>) {
        let mut seen = HashSet::new();
        for bridge in &graph.bridges {
            if !seen.insert(bridge.bridge_id.as_str()) {
                errors.push(format!(
                    "Gate 10: Duplicate bridge ID '{}'",
                    bridge.bridge_id
                ));
            }
        }
    }
}
