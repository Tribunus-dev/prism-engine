import Foundation

/// AgentOrchestrator moved to Rust ECS. The Swift app now communicates
/// with the orchestrator via MCP; this typealias preserves existing API
/// surface while deferring all orchestration logic to the engine bridge.
public typealias AgentOrchestrator = Any

/// Minimal inference provider stub used by the agent loop.
/// In production this is replaced by an MCP-based bridge to the Rust engine.
public protocol InferenceProviderProtocol {
    func runTurn(prompt: String, context: [String: Any]) async throws -> String
}

public actor InferenceProvider: InferenceProviderProtocol {
    public func runTurn(prompt: String, context: [String: Any] = [:]) async throws -> String {
        // Stub: forward to Rust engine via MCP bridge
        return try await mcpInference(prompt: prompt, context: context)
    }

    private func mcpInference(prompt: String, context: [String: Any]) async throws -> String {
        // TODO: Wire to prism-mcpd inference handler
        throw PrismAgentError.featureNotImplemented("MCP inference bridge")
    }
}

public enum PrismAgentError: Error, LocalizedError {
    case featureNotImplemented(String)
    case agentFailed(String)

    public var errorDescription: String? {
        switch self {
        case .featureNotImplemented(let feature):
            return "Feature not implemented: \(feature)"
        case .agentFailed(let reason):
            return "Agent failed: \(reason)"
        }
    }
}

/// Runs a swarm of sub-agents against a set of prompts.
public actor SwarmEngine {
    private var agents: [String: Any] = [:]

    public init() {}

    // MARK: - Public API

    /// Run a single agent with the given orchestrator and prompt.
    @discardableResult
    public func runAgent(
        name: String,
        orchestrator: AgentOrchestrator,
        prompt: String
    ) async throws -> String {
        let provider = InferenceProvider()
        let context: [String: Any] = [
            "agent_name": name,
            "orchestrator": orchestrator,
        ]
        return try await provider.runTurn(prompt: prompt, context: context)
    }

    /// Run a multi-agent swarm. Each task gets a fresh orchestrator reference.
    @discardableResult
    public func runSwarm(
        tasks: [(name: String, prompt: String)],
        orchestrator: AgentOrchestrator
    ) async throws -> [String: String] {
        var results: [String: String] = [:]
        for task in tasks {
            let result = try await runAgent(
                name: task.name,
                orchestrator: orchestrator,
                prompt: task.prompt
            )
            results[task.name] = result
        }
        return results
    }

    /// List registered agent names.
    public func registeredAgents() -> [String] {
        return Array(agents.keys)
    }

    // MARK: - Internal

    /// Register an agent (for future use).
    internal func registerAgent(name: String, agent: Any) {
        agents[name] = agent
    }
}
