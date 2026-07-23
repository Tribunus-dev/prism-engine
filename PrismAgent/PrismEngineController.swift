import Foundation
import PrismEngineC

enum PrismEngineError: Error {
    case initializationFailed
    case compilationFailed(code: Int32)
    case sandboxingViolation
}

@MainActor
final class PrismEngineController: ObservableObject {
    @Published var isCompiling: Bool = false
    @Published private(set) var compilePhase: String = "Ready"
    @Published private(set) var compileProgress: Double = 0
    @Published private(set) var compileEvents: [CompilerLabEvent] = []
    @Published private(set) var searchGeneration = 0
    @Published private(set) var searchGenerations = 0
    @Published private(set) var searchPopulation = 0
    @Published private(set) var searchBestFitness: Double?
    @Published private(set) var searchCandidates: [DaemonSearchCandidate] = []
    @Published var isRunning: Bool = false
    @Published var lastError: String? = nil
    @Published private(set) var daemonModels: [String] = []
    static let shared = PrismEngineController()

    private nonisolated(unsafe) var multiplexerPtr: OpaquePointer? = nil

    private var appSupportDir: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
    }

    var activeCImagePath: String {
        appSupportDir.appendingPathComponent("Gemma4_Unified.cimage").path
    }

    var activeModelDirectory: String {
        appSupportDir.appendingPathComponent("downloads").path
    }

    deinit {
        if let ptr = multiplexerPtr {
            prism_runtime_free(ptr)
        }
    }

    /// Compile the .cimage from downloaded safetensors and bundled resources.
    func compileDownloadedWeights() async throws {
        await MainActor.run {
            isCompiling = true; lastError = nil; compileProgress = 0
            compilePhase = "Preparing"
            compileEvents = [CompilerLabEvent(phase: "Preparing", detail: "Locating downloaded model weights", progress: 0)]
        }

        let safetensorsURL = appSupportDir.appendingPathComponent("downloads")
        let outputURL = URL(fileURLWithPath: activeCImagePath)

        guard let resourceURL = Bundle.main.resourceURL else {
            throw PrismEngineError.sandboxingViolation
        }

        try FileManager.default.createDirectory(at: safetensorsURL, withIntermediateDirectories: true)

        await MainActor.run {
            compilePhase = "Compiling"
            compileProgress = 0.35
            compileEvents.append(CompilerLabEvent(phase: "Compiling", detail: "Packing model into a cimage artifact", progress: 0.35))
        }

        let status = prism_compile_and_pack(
            safetensorsURL.path,
            outputURL.path,
            resourceURL.path
        )

        await MainActor.run {
            isCompiling = false
            compileProgress = status == 0 ? 1 : compileProgress
            compilePhase = status == 0 ? "Verified" : "Failed"
            compileEvents.append(CompilerLabEvent(phase: status == 0 ? "Verified" : "Failed", detail: status == 0 ? "Artifact emitted successfully" : "Compiler returned an error", progress: status == 0 ? 1 : compileProgress))
        }

        guard status == 0 else {
            throw PrismEngineError.compilationFailed(code: status)
        }
    }

    /// Boot the zero-copy CoreML + Metal runtime scheduler.
    func bootEngineRuntime() throws {
        let outputURL = appSupportDir.appendingPathComponent("Gemma4_Unified.cimage")

        guard FileManager.default.fileExists(atPath: outputURL.path) else {
            throw PrismEngineError.initializationFailed
        }

        guard let ptr = prism_runtime_init(outputURL.path) else {
            throw PrismEngineError.initializationFailed
        }

        multiplexerPtr = ptr
        isRunning = true
    }

    func refreshDaemonModels() async {
        do {
            daemonModels = try await PrismDaemonClient.shared.listModels()
        } catch {
            daemonModels = []
            lastError = error.localizedDescription
        }
    }

    func recordDaemonEvent(_ event: DaemonCompilerEvent) {
        compilePhase = event.phase
        if let progress = event.progress { compileProgress = progress }
        if event.type == "search" {
            searchGeneration = event.generation ?? searchGeneration
            searchGenerations = event.generations ?? searchGenerations
            searchPopulation = event.population ?? searchPopulation
            searchBestFitness = event.bestFitness ?? searchBestFitness
            searchCandidates = event.candidates ?? searchCandidates
        }
        compileEvents.append(CompilerLabEvent(phase: event.phase, detail: event.detail, progress: event.progress ?? compileProgress))
        if compileEvents.count > 100 { compileEvents.removeFirst(compileEvents.count - 100) }
    }
}
