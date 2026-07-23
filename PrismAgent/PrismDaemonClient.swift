import Foundation

/// Shared HTTP client for native surfaces that need the local Prism daemon.
struct PrismDaemonClient: Sendable {
    static let shared = PrismDaemonClient()

    private let baseURL: URL

    private init() {
        baseURL = URL(string: ProcessInfo.processInfo.environment["PRISM_DAEMON_HTTP"] ?? "http://127.0.0.1:8080")!
    }

    func compilerLabEvents() -> AsyncThrowingStream<DaemonCompilerEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = URLSession.shared.webSocketTask(with: baseURL.appendingPathComponent("api/compiler-lab/ws"))
            task.resume()
            func receive() {
                task.receive { result in
                    switch result {
                    case .failure(let error): continuation.finish(throwing: error)
                    case .success(let message):
                        if case .string(let text) = message,
                           let data = text.data(using: .utf8),
                           let event = try? JSONDecoder().decode(DaemonCompilerEvent.self, from: data) {
                            continuation.yield(event)
                        }
                        receive()
                    }
                }
            }
            receive()
            continuation.onTermination = { _ in task.cancel(with: .goingAway, reason: nil) }
        }
    }

    func generate(prompt: String, model: String? = nil, maxTokens: Int = 256) async throws -> String {
        let modelName = model
            ?? ProcessInfo.processInfo.environment["PRISM_MODEL"]
            ?? "Gemma4_Unified"
        let endpoint = baseURL.appendingPathComponent("api/generate")
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(Request(model: modelName, prompt: prompt, maxTokens: maxTokens))
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw NSError(domain: "PrismDaemon", code: 1, userInfo: [NSLocalizedDescriptionKey: "Prism daemon is unavailable"])
        }
        return try JSONDecoder().decode(Response.self, from: data).text
    }

    func listModels() async throws -> [String] {
        let (data, response) = try await URLSession.shared.data(from: baseURL.appendingPathComponent("api/models"))
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw NSError(domain: "PrismDaemon", code: 2, userInfo: [NSLocalizedDescriptionKey: "Prism daemon model registry is unavailable"])
        }
        return try JSONDecoder().decode([String].self, from: data)
    }

    func startEvolutionSearch(model: String, population: Int = 24, generations: Int = 12) async throws {
        var request = URLRequest(url: baseURL.appendingPathComponent("api/compiler-lab/search"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: ["model": model, "population": population, "generations": generations])
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw NSError(domain: "PrismDaemon", code: 3, userInfo: [NSLocalizedDescriptionKey: "Evolution search could not be started"])
        }
        if let body = try? JSONSerialization.jsonObject(with: data) as? [String: Any], body["error"] != nil {
            throw NSError(domain: "PrismDaemon", code: 4, userInfo: [NSLocalizedDescriptionKey: body["error"] as? String ?? "Evolution search rejected"])
        }
    }

    func generate(prompt: String, model: String? = nil, maxTokens: Int = 256, completion: @escaping (Result<String, Error>) -> Void) {
        let modelName = model
            ?? ProcessInfo.processInfo.environment["PRISM_MODEL"]
            ?? "Gemma4_Unified"
        var request = URLRequest(url: baseURL.appendingPathComponent("api/generate"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        do {
            request.httpBody = try JSONEncoder().encode(Request(model: modelName, prompt: prompt, maxTokens: maxTokens))
        } catch {
            completion(.failure(error)); return
        }
        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error { completion(.failure(error)); return }
            guard let data, let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                completion(.failure(NSError(domain: "PrismDaemon", code: 1, userInfo: [NSLocalizedDescriptionKey: "Prism daemon is unavailable"])))
                return
            }
            do { completion(.success(try JSONDecoder().decode(Response.self, from: data).text)) }
            catch { completion(.failure(error)) }
        }.resume()
    }

    private struct Request: Encodable {
        let model: String
        let prompt: String
        let maxTokens: Int
        enum CodingKeys: String, CodingKey { case model, prompt; case maxTokens = "max_tokens" }
    }

    private struct Response: Decodable { let text: String }
}

struct DaemonCompilerEvent: Decodable, Sendable {
    let type: String
    let phase: String
    let detail: String
    let progress: Double?
    let generation: Int?
    let generations: Int?
    let population: Int?
    let bestFitness: Double?
    let candidates: [DaemonSearchCandidate]?
}

struct DaemonSearchCandidate: Decodable, Sendable, Identifiable {
    let id: Int
    let fitness: Double
    let representation: String
}
