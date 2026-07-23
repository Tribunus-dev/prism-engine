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
        let daemonURL = ProcessInfo.processInfo.environment["PRISM_DAEMON_HTTP"] ?? "http://127.0.0.1:8080"
        let model = (context["model"] as? String)
            ?? ProcessInfo.processInfo.environment["PRISM_MODEL"]
            ?? "Gemma4_Unified"
        guard let url = URL(string: daemonURL + "/api/generate") else {
            throw PrismAgentError.agentFailed("Invalid Prism daemon URL")
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(GenerateRequest(model: model, prompt: prompt, maxTokens: 256))

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw PrismAgentError.agentFailed("Prism daemon returned an invalid response")
        }
        let payload = try JSONDecoder().decode(GenerateResponse.self, from: data)
        return payload.text
    }
}

private struct GenerateRequest: Encodable {
    let model: String
    let prompt: String
    let maxTokens: Int

    enum CodingKeys: String, CodingKey {
        case model, prompt
        case maxTokens = "max_tokens"
    }
}

private struct GenerateResponse: Decodable {
    let text: String
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
