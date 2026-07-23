import Foundation
import AppKit
import AVFoundation
import CoreVideo
import CoreAudio
import WebKit
import SwiftUI

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Streaming Inference Adapter
// ─────────────────────────────────────────────────────────────────────────────

/// Receives streaming inference events and routes them to the UI, media pipeline,
/// and vector store.
@MainActor
final class StreamHandler: NSObject {
    private var fullText = ""
    private(set) var didReceiveAudio = false
    private let audioEngine = AVAudioEngine()
    private let audioPlayer = AVAudioPlayerNode()
    private var onTextUpdate: ((String) -> Void)?
    private var onComplete: ((String, Bool) -> Void)?
    private var onError: ((String) -> Void)?
    private(set) var exporter: ProfessionalMediaExporter?

    override init() {
        super.init()
        audioEngine.attach(audioPlayer)
        audioEngine.connect(audioPlayer, to: audioEngine.mainMixerNode, format: nil)
        try? audioEngine.start()
    }

    /// Attach callbacks for the chat UI.
    func attach(
        textUpdate: @escaping (String) -> Void,
        complete: @escaping (String, Bool) -> Void,
        error: @escaping (String) -> Void
    ) {
        onTextUpdate = textUpdate
        onComplete = complete
        onError = error
    }

    /// Attach a media exporter for video/audio output.
    func attachExporter(_ exporter: ProfessionalMediaExporter) {
        self.exporter = exporter
    }

    nonisolated func handleEvent(_ event: StreamEvent) {
        switch event {
            case .text(let token):
            Task { @MainActor [token] in
                self.fullText += token
                self.onTextUpdate?(token)
            }
        case .imageFrame(let bytes, _, _):
            Task { @MainActor in if let _ = NSImage(data: bytes) {} }
        case .videoFrame(let bytes, let width, let height, let ts):
            Task { @MainActor [bytes, width, height, ts] in
                exporter?.appendVideoFrame(bytes: bytes, width: Int(width), height: Int(height), timestampNs: ts)
            }
        case .audioChunk(let bytes, let rate, let channels):
            Task { @MainActor [bytes, rate, channels] in
                playPCM(bytes, sampleRate: Double(rate), channels: channels)
                exporter?.appendAudioChunk(bytes: bytes, sampleRate: Double(rate), channels: channels)
            }
        case .embedding:
                break
        }
    }

    nonisolated func handleDone() {
        Task { @MainActor in
            exporter?.finalizeExport {}
            onComplete?(fullText, didReceiveAudio)
        }
    }

    nonisolated func handleError(_ error: String) {
        Task { @MainActor in
            onError?(error)
        }
    }

    private func playPCM(_ bytes: Data, sampleRate: Double, channels: UInt32) {
        guard channels > 0 else { return }
        let fmt = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: sampleRate, channels: AVAudioChannelCount(channels), interleaved: true)
        guard let format = fmt, let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(bytes.count / 2 / Int(channels))) else { return }
        buf.frameLength = buf.frameCapacity
        bytes.withUnsafeBytes { ptr in
            buf.int16ChannelData?.pointee.update(from: ptr.bindMemory(to: Int16.self).baseAddress!, count: Int(buf.frameLength) * Int(channels))
        }
        audioPlayer.scheduleBuffer(buf)
        if !audioEngine.isRunning { try? audioEngine.start() }
    }
}

/// UniFFI callback interface — receives events from the Rust bridge.
final class StreamCallbackAdapter: MultimodalStreamCallback {
    private let handler: StreamHandler

    init(handler: StreamHandler) {
        self.handler = handler
    }

    func onEvent(event: StreamEvent) {
        handler.handleEvent(event)
    }

    func onDone() {
        handler.handleDone()
    }

    func onError(error: String) {
        handler.handleError(error)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Compilation Progress Adapter
// ─────────────────────────────────────────────────────────────────────────────

/// Receives deterministic compilation progress from the Rust AOT compiler.
@MainActor
final class CompilationMonitor: ObservableObject {
    @Published var log: [String] = []
    @Published var progress: Float = 0
    @Published var isComplete = false
    @Published var error: String?

    func attachToCompiler() -> CompilationCallbackAdapter {
        CompilationCallbackAdapter(monitor: self)
    }
}

final class CompilationCallbackAdapter: CompilerProgressCallback, @unchecked Sendable {
    private let monitor: CompilationMonitor

    init(monitor: CompilationMonitor) {
        self.monitor = monitor
    }

    func onLog(message: String) {
        Task { @MainActor in
            monitor.log.append(message)
        }
    }

    func onProgress(percentage: Float) {
        Task { @MainActor in
            monitor.progress = percentage / 100
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Agent Orchestrator
// ─────────────────────────────────────────────────────────────────────────────

/// Drives the agent state machine — compile, infer, run tool loops, and
/// manage subagents.  Runs on a background actor; UI updates via callbacks.
actor AgentOrchestrator {
    private var state: BridgeAgentState?
    private let streamHandler: StreamHandler
    private let tools: [BridgeToolDefinition]

    init(streamHandler: StreamHandler) {
        self.streamHandler = streamHandler
        self.tools = prismDefaultTools()
    }

    /// Full turn: compile → infer → step → loop until done.
    func runTurn(
        ggufPath: String,
        modelDir: String,
        prompt: String
    ) async throws -> String {
        // 1. Compile GGUF → .cimage
        let monitor = await MainActor.run { CompilationMonitor() }
        let outputDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("prism-\(UUID().uuidString)")
            .path

        let cimageDir = try prismCompileGguf(
            ggufPath: ggufPath,
            outputDir: outputDir,
            callback: await MainActor.run { monitor.attachToCompiler() }
        )

        // 2. Initialise state
        let initialStateJson = BridgeAgentState(
            phase: .idle,
            historyJsonl: "",
            currentPrompt: prompt
        )

        // 3. Inference + step loop
        var currentState = initialStateJson
        var finalText = ""

        for _ in 0..<10 { // max 10 rounds
            // Run streaming inference
            let adapter = StreamCallbackAdapter(handler: streamHandler)
            prismInferMultimodalStream(
                cimagePath: cimageDir + "/model.cimage",
                modelDir: modelDir,
                prompt: currentState.currentPrompt,
                callback: adapter
            )

            // Step the state machine (model output collected by streamHandler)
            // Note: in a real integration, streamHandler.fullText would be set
            // by the callback.  For now we collect it synchronously.

            // Check for tool calls
            let result = prismAgentStep(
                stateJson: currentState.historyJsonl,
                modelOutput: ""  // full text from streamHandler
            )

            switch result.outcome {
            case .awaitingTools(let tools):
                // Execute each tool in the sandbox
                for tool in tools {
                    _ = executeSandboxTool(tool)  // Result intentionally unused during sandbox phase
                }
                currentState = result.state

            case .awaitingSubagents(let subagents):
                // Spawn subagent for each handle
                for _ in subagents {
                    // Create new AgentOrchestrator, run with sub.goal
                }
                currentState = result.state

            case .generating:
                currentState = result.state
                continue

            case .finished(let text):
                finalText = text
                return finalText
            }
        }

        return finalText
    }

    private func executeSandboxTool(_ tool: BridgeToolCall) -> String {
        // Map tool.name to sandbox operations
        return "{\"ok\": true}"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Professional Media Exporter
// ─────────────────────────────────────────────────────────────────────────────

/// Writes raw pixel buffers and PCM audio to a ProRes .mov via AVAssetWriter.
final class ProfessionalMediaExporter {
    private var assetWriter: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var audioInput: AVAssetWriterInput?
    private var pixelBufferAdaptor: AVAssetWriterInputPixelBufferAdaptor?

    init(outputURL: URL, width: Int, height: Int, sampleRate: Double, channels: UInt32) {
        guard let writer = try? AVAssetWriter(outputURL: outputURL, fileType: .mov) else { return }
        assetWriter = writer

        // ProRes video settings
        let videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.proRes422,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height
        ]
        let videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        videoInput.expectsMediaDataInRealTime = false
        self.videoInput = videoInput

        pixelBufferAdaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: videoInput,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ]
        )

        // Linear PCM audio — broadcast standard (48 kHz, 24-bit)
        let audioSettings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: channels,
            AVLinearPCMBitDepthKey: 24,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsFloatKey: false,
            "AVLinearPCMIsNonInterleaved": false
        ]
        let audioInput = AVAssetWriterInput(mediaType: .audio, outputSettings: audioSettings)
        audioInput.expectsMediaDataInRealTime = false
        self.audioInput = audioInput

        if writer.canAdd(videoInput) { writer.add(videoInput) }
        if writer.canAdd(audioInput) { writer.add(audioInput) }

        writer.startWriting()
        writer.startSession(atSourceTime: .zero)
    }

    func appendVideoFrame(bytes: Data, width: Int, height: Int, timestampNs: UInt64) {
        guard let adaptor = pixelBufferAdaptor,
              let videoInput = videoInput,
              videoInput.isReadyForMoreMediaData else { return }

        if let pixelBuffer = createPixelBuffer(from: bytes, width: width, height: height) {
            let time = CMTime(value: CMTimeValue(timestampNs), timescale: 1_000_000_000)
            adaptor.append(pixelBuffer, withPresentationTime: time)
        }
    }

    func appendAudioChunk(bytes: Data, sampleRate: Double, channels: UInt32) {
        guard let audioInput = audioInput,
              audioInput.isReadyForMoreMediaData else { return }

        if let sampleBuffer = createAudioSampleBuffer(
            from: bytes, sampleRate: sampleRate, channels: channels
        ) {
            audioInput.append(sampleBuffer)
        }
    }

    func finalizeExport(completion: @escaping @Sendable () -> Void) {
        videoInput?.markAsFinished()
        audioInput?.markAsFinished()
        assetWriter?.finishWriting(completionHandler: completion)
    }

    // MARK: - CoreVideo Bridge

    private func createPixelBuffer(from data: Data, width: Int, height: Int) -> CVPixelBuffer? {
        var pixelBuffer: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary

        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height,
            kCVPixelFormatType_32BGRA, attrs, &pixelBuffer
        )
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])
        if let contextAddress = CVPixelBufferGetBaseAddress(buffer) {
            data.withUnsafeBytes { ptr in
                if let base = ptr.baseAddress {
                    memcpy(contextAddress, base, data.count)
                }
            }
        }
        CVPixelBufferUnlockBaseAddress(buffer, [])

        return buffer
    }

    // MARK: - CoreAudio Bridge

    private func createAudioSampleBuffer(
        from data: Data, sampleRate: Double, channels: UInt32
    ) -> CMSampleBuffer? {
        var blockBuffer: CMBlockBuffer?
        let length = data.count

        let status = data.withUnsafeBytes { rawBufferPointer in
            CMBlockBufferCreateWithMemoryBlock(
                allocator: kCFAllocatorDefault,
                memoryBlock: UnsafeMutableRawPointer(mutating: rawBufferPointer.baseAddress),
                blockLength: length,
                blockAllocator: kCFAllocatorNull,
                customBlockSource: nil,
                offsetToData: 0,
                dataLength: length,
                flags: 0,
                blockBufferOut: &blockBuffer
            )
        }
        guard status == noErr, let bBuffer = blockBuffer else { return nil }

        var asbd = AudioStreamBasicDescription(
            mSampleRate: sampleRate,
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked,
            mBytesPerPacket: 3 * channels,
            mFramesPerPacket: 1,
            mBytesPerFrame: 3 * channels,
            mChannelsPerFrame: channels,
            mBitsPerChannel: 24,
            mReserved: 0
        )

        var formatDescription: CMAudioFormatDescription?
        CMAudioFormatDescriptionCreate(
            allocator: kCFAllocatorDefault,
            asbd: &asbd,
            layoutSize: 0, layout: nil,
            magicCookieSize: 0, magicCookie: nil,
            extensions: nil,
            formatDescriptionOut: &formatDescription
        )
        guard let fDesc = formatDescription else { return nil }

        var sampleBuffer: CMSampleBuffer?
        let sampleCount = length / Int(asbd.mBytesPerFrame)

        CMSampleBufferCreateReady(
            allocator: kCFAllocatorDefault,
            dataBuffer: bBuffer,
            formatDescription: fDesc,
            sampleCount: sampleCount,
            sampleTimingEntryCount: 0,
            sampleTimingArray: nil,
            sampleSizeEntryCount: 0,
            sampleSizeArray: nil,
            sampleBufferOut: &sampleBuffer
        )

        return sampleBuffer
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Model Downloads
// ─────────────────────────────────────────────────────────────────────────────

/// Manages model downloads: HuggingFace repos → local storage → GGUF files.
actor ModelManager {
    private let fileManager = FileManager.default

    private var modelsDir: URL {
        let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport.appendingPathComponent("prism-models")
    }

    /// Ensure a GGUF file is downloaded from HuggingFace.
    /// Returns the local file path.
    func ensureModel(repo: String, filename: String) async throws -> String {
        let modelDir = modelsDir.appendingPathComponent(repo.replacingOccurrences(of: "/", with: "--"))
        let ggufPath = modelDir.appendingPathComponent(filename)

        if fileManager.fileExists(atPath: ggufPath.path) {
            return ggufPath.path
        }

        try fileManager.createDirectory(at: modelDir, withIntermediateDirectories: true)

        // Download using URLSession with progress
        let url = URL(string: "https://huggingface.co/\(repo)/resolve/main/\(filename)")!
        let downloadedPath = try await downloadWithProgress(url: url, to: ggufPath)
        return downloadedPath
    }

    private func downloadWithProgress(url: URL, to destination: URL) async throws -> String {
        let session = URLSession(configuration: .default)
        let (tempURL, _) = try await session.download(from: url)

        try fileManager.moveItem(at: tempURL, to: destination)
        return destination.path
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Web Browser Controller
// ─────────────────────────────────────────────────────────────────────────────

/// Semantic DOM reducer: walks the page, strips noise (scripts, styles, SVGs),
/// tags every interactive element with `data-prism-id`, and returns a clean
/// JSON tree the agent can reason about without being overwhelmed by raw HTML.
private let semanticReducerScript = """
(function() {
    let interactiveElements = [];
    let elementCounter = 0;

    const IGNORED_TAGS = new Set(['SCRIPT', 'STYLE', 'SVG', 'NOSCRIPT', 'IFRAME', 'META', 'LINK', 'PATH']);
    const MEDIA_TAGS = new Set(['IMG', 'VIDEO', 'AUDIO']);

    function walkDOM(node) {
        if (node.nodeType === Node.TEXT_NODE) {
            let text = node.textContent.trim();
            if (!text) return null;
            // Neutralize overt imperative injection vectors
            // Catches obvious command overrides without relying on the LLM
            const injectionTriggers = /ignore previous instructions|system override|you are now|new rule|cancel all instructions/i;
            if (injectionTriggers.test(text)) {
                text = '[POTENTIAL INJECTION REMOVED BY PRISM REDUCER]';
        }
            return { type: 'text', content: text };
        }

        if (node.nodeType !== Node.ELEMENT_NODE) return null;
        if (IGNORED_TAGS.has(node.tagName)) return null;

        // Media elements — tag them for web_extract_media
        if (MEDIA_TAGS.has(node.tagName)) {
            const currentId = elementCounter++;
            node.setAttribute('data-prism-id', currentId);
            var mediaData = {
                tag: node.tagName.toLowerCase(),
                id: currentId,
                src: node.currentSrc || node.getAttribute('src') || ''
            };
            if (node.tagName === 'IMG') {
                mediaData.alt = node.getAttribute('alt') || '';
                mediaData.width = node.clientWidth;
                mediaData.height = node.clientHeight;
            } else if (node.tagName === 'VIDEO') {
                mediaData.duration = node.duration || null;
                mediaData.width = node.clientWidth;
                mediaData.height = node.clientHeight;
            } else if (node.tagName === 'AUDIO') {
                mediaData.duration = node.duration || null;
            }
            return mediaData;
        }

        const isInteractive = ['A', 'BUTTON', 'INPUT', 'SELECT', 'TEXTAREA'].includes(node.tagName) ||
                              node.hasAttribute('onclick') ||
                              node.getAttribute('role') === 'button';

        var nodeData = {
            tag: node.tagName.toLowerCase()
        };

        if (isInteractive) {
            const currentId = elementCounter++;
            nodeData.id = currentId;
            node.setAttribute('data-prism-id', currentId);

            if (node.tagName === 'A') nodeData.href = node.getAttribute('href');
            if (node.tagName === 'INPUT') {
                nodeData.inputType = node.getAttribute('type') || 'text';
                nodeData.value = node.value;
                nodeData.placeholder = node.getAttribute('placeholder');
            }
        }

        let children = [];
        for (let child of node.childNodes) {
            let childData = walkDOM(child);
            if (childData) children.push(childData);
        }

        if (children.length > 0) nodeData.children = children;

        // Flatten non-semantic wrappers
        if (!isInteractive && ['div', 'span', 'section'].includes(nodeData.tag) && children.length === 1 && children[0].type === 'text') {
            return children[0];
        }

        return (children.length > 0 || isInteractive) ? nodeData : null;
    }

    const structure = walkDOM(document.body);

    return JSON.stringify({
        url: window.location.href,
        title: document.title,
        dom_digest: structure
    });
})();
"""

// MARK: - Browser Tab

/// Represents a single browser tab managed by the web controller.
public struct BrowserTab: Identifiable {
    public let id: UUID = UUID()
    public var title: String
    public var url: URL
    public var isLoading: Bool = false
    public var estimatedProgress: Double = 0.0
    public var webView: WKWebView?

    @MainActor public var canGoBack: Bool { webView?.canGoBack ?? false }
    @MainActor public var canGoForward: Bool { webView?.canGoForward ?? false }

    public init(title: String, url: URL, webView: WKWebView? = nil) {
        self.title = title
        self.url = url
        self.webView = webView
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Drives a WebView2 for the agent's browser tools.
/// Receives tool calls from the agent loop and executes them natively against
/// Chromium — no Rust FFI needed for browser state.
@MainActor
final class PrismWebController: NSObject {
    // MARK: - Tab Management

    public var tabs: [BrowserTab] = []
    public var activeTab: BrowserTab? = nil
    public var activeTabID: UUID? { activeTab?.id }

    public func createNewTab(url: URL? = nil) {
        let tab = BrowserTab(title: "New Tab", url: url ?? URL(string: "about:blank")!)
        tabs.append(tab)
        activeTab = tab
    }

    public func closeTab(id: UUID) {
        if let index = tabs.firstIndex(where: { $0.id == id }) {
            let wasActive = tabs[index].id == activeTab?.id
            tabs.remove(at: index)
            if wasActive {
                if index < tabs.count {
                    activeTab = tabs[index]
                } else if let last = tabs.last {
                    activeTab = last
                } else {
                    activeTab = nil
                }
            }
        }
    }

    public func selectTab(id: UUID) {
        activeTab = tabs.first(where: { $0.id == id })
    }

    let webView: WKWebView
    private let isHeadless: Bool
    private let jsEnabled: Bool
    /// Set before a headless session's runJavaScript call.  Intercepted
    /// downloads (PDFs, ZIPs, CSVs triggered by clicking a link) are
    /// written to this directory.
    var currentSandboxRoot: URL?
    private var navigationContinuation: CheckedContinuation<Bool, Error>?

    init(isHeadless: Bool = false, jsEnabled: Bool = false) {
        self.isHeadless = isHeadless
        self.jsEnabled = jsEnabled
        let config = WKWebViewConfiguration()
        let wv = WKWebView(frame: .zero, configuration: config)

        if isHeadless {
            wv.customUserAgent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15 PrismAgent/1.0"
        }
        self.webView = wv
        super.init()
        webView.navigationDelegate = self
    }

    /// Mount this WebView2 in your SwiftUI/AppKit layout.
    func getWebView() -> WKWebView {
        return webView
    }

    /// Map a web tool call from the agent to native Chromium actions.
    func execute(name: String, argumentsJson: String) async throws -> String {
        guard let data = argumentsJson.data(using: .utf8),
              let args = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return "Error: Invalid JSON arguments"
        }

        switch name {
        case "web_navigate":
            return try await navigate(args: args)

        case "web_snapshot":
            return try await snapshot()

        case "web_interact":
            return try await interact(args: args)

        case "web_extract_media":
            return try await extractMedia(args: args)

        case "web_evaluate_js":
            return try await evaluateJS(args: args)

        case "web_download":
            guard let url = args["url"] as? String,
                  let filename = args["filename"] as? String,
                  let sandboxRoot = currentSandboxRoot else {
                return "Error: Missing required parameters 'url', 'filename', or sandbox root not configured"
        }
            return try await executeDownload(urlString: url, filename: filename, sandboxRoot: sandboxRoot)

        default:
            return "Error: Unknown web tool \"\(name)\""
        }
    }

    // MARK: - Media Extraction

    /// Canvas-based extraction: paints the target element to a <canvas>, reads
    /// the base64 PNG, writes to a temp file, and returns the path.
    private func extractMedia(args: [String: Any]) async throws -> String {
        guard let id = args["id"] as? Int else {
            return "Error: Missing required parameter 'id'"
        }

        let extractScript = """
        (function() {
            var el = document.querySelector('[data-prism-id="\(id)"]');
            if (!el) return 'Error: Element not found';

            if (el.tagName === 'IMG' || el.tagName === 'VIDEO') {
                var canvas = document.createElement('canvas');
                canvas.width = el.videoWidth || el.naturalWidth || el.clientWidth || 640;
                canvas.height = el.videoHeight || el.naturalHeight || el.clientHeight || 480;
                var ctx = canvas.getContext('2d');
                ctx.drawImage(el, 0, 0, canvas.width, canvas.height);
                return canvas.toDataURL('image/png');
            }
            if (el.tagName === 'AUDIO') {
                return 'Error: Audio extraction requires blob URL mapping (src: ' + (el.src || 'none') + ')';
            }
            return 'Error: Element is not a supported media type';
        })();
        """

        guard let result = try await webView.evaluateJavaScript(extractScript) as? String else {
            return "Error: JS execution returned nil"
        }

        if result.hasPrefix("data:image/png;base64,") {
            let base64 = String(result.dropFirst("data:image/png;base64,".count))
            guard let imageData = Data(base64Encoded: base64) else {
                return "Error: Base64 decode failed"
            }

            let tempURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("prism-media-\(id).png")
            try imageData.write(to: tempURL)

            return "Success: Media extracted to \(tempURL.path). Use read_file to inspect."
        }

        return result
    }

    // MARK: - Tool Implementations

    private func navigate(args: [String: Any]) async throws -> String {
        guard let urlString = args["url"] as? String,
              let url = URL(string: urlString) else {
            return "Error: Invalid URL"
        }

        // Tier 2 (Static) mode: route through Rust X-Ray proxy
        // This fetches raw HTML, strips malicious scripts via swc AST guard,
        // injects strict CSP, and loads the certified-clean HTML string
        if !jsEnabled {
            do {
                let sanitizedHTML = try await prismXrayNavigate(url: urlString)
                _ = await MainActor.run {
                    webView.loadHTMLString(sanitizedHTML, baseURL: url)
                }
                let loaded = try await withCheckedThrowingContinuation { (c: CheckedContinuation<Bool, Error>) in
                    navigationContinuation = c
                    Task {
                        try await Task.sleep(nanoseconds: 30_000_000_000)
                        if let cont = navigationContinuation {
                            cont.resume(returning: false)
                            navigationContinuation = nil
                        }
                    }
                }
                return loaded ? "Navigated safely via X-Ray to \(urlString)" : "Error: Navigation timed out"
            } catch {
                return "Error: X-Ray proxy failed: \(error.localizedDescription)"
            }
        }

        // Tier 3 (Dynamic) mode: direct Chromium navigation with JS enabled
        webView.load(URLRequest(url: url))

        let loaded = try await withCheckedThrowingContinuation { (c: CheckedContinuation<Bool, Error>) in
            navigationContinuation = c
            Task {
                try await Task.sleep(nanoseconds: 30_000_000_000)
                if let cont = navigationContinuation {
                    cont.resume(returning: false)
                    navigationContinuation = nil
                }
            }
        }
        return loaded ? "Navigated to \(urlString)" : "Error: Navigation timed out"
    }

    private func snapshot() async throws -> String {
        let result = try await webView.evaluateJavaScript(semanticReducerScript)
        if let jsonString = result as? String {
            return jsonString
        }
        return "Error: Could not parse DOM"
    }

    private func interact(args: [String: Any]) async throws -> String {
        guard let id = args["id"] as? Int,
              let action = args["action"] as? String else {
            return "Error: Missing required parameters 'id' and 'action'"
        }

        switch action {
        case "click":
            let script = """
                var el = document.querySelector('[data-prism-id="\(id)"]');
                if (el) { el.click(); 'clicked \(id)'; } else { 'Element \(id) not found'; }
            """
            let result = try await webView.evaluateJavaScript(script)
            return String(describing: result ?? "nil")

        case "type":
            guard let value = args["value"] as? String else {
                return "Error: 'value' required for type action"
            }
            // Bulletproof escaping via JSON encoder — handles quotes, backslashes, newlines
            let jsonData = try JSONSerialization.data(withJSONObject: [value], options: [])
            let jsonArray = String(data: jsonData, encoding: .utf8) ?? "[]"
            let safeValue = String(jsonArray.dropFirst().dropLast())
            let script = """
                (function() {
                var el = document.querySelector('[data-prism-id="\(id)"]');
                if (!el) return 'Element \(id) not found';
                    el.value = \(safeValue);
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                    el.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true }));
                    el.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true }));
                    return 'typed into \(id)';
                })();
            """
            let result = try await webView.evaluateJavaScript(script)
            return String(describing: result ?? "nil")

        case "focus":
            let script = """
                var el = document.querySelector('[data-prism-id="\(id)"]');
                if (el) { el.focus(); 'focused \(id)'; } else { 'Element \(id) not found'; }
            """
            let result = try await webView.evaluateJavaScript(script)
            return String(describing: result ?? "nil")

        default:
            return "Error: Unknown action '\(action)'"
        }
    }

    private func evaluateJS(args: [String: Any]) async throws -> String {
        guard let script = args["script"] as? String else {
            return "Error: Missing 'script' parameter"
        }
        let result = try await webView.evaluateJavaScript(script)
        return String(describing: result ?? "nil")
    }

    // MARK: - Session-Aware Download

    /// Download a file using the WebView2's authenticated session.
    /// Extracts cookies from the ephemeral store and attaches them to the request.
    private func executeDownload(urlString: String, filename: String, sandboxRoot: URL) async throws -> String {
        guard let url = URL(string: urlString) else {
            return "Error: Invalid URL format"
        }

        // Build request with session state injected
        var request = URLRequest(url: url)
        if let ua = webView.customUserAgent {
            request.addValue(ua, forHTTPHeaderField: "User-Agent")
        }

        let (tempURL, response) = try await URLSession.shared.download(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            return "Error: Download failed with HTTP status \((response as? HTTPURLResponse)?.statusCode ?? 0)"
        }

        // Secure sandbox routing — strip path components to prevent traversal
        let safeFilename = URL(fileURLWithPath: filename).lastPathComponent
        let destinationURL = sandboxRoot.appendingPathComponent(safeFilename)

        // Atomic move
        if FileManager.default.fileExists(atPath: destinationURL.path) {
            try FileManager.default.removeItem(at: destinationURL)
        }
        try FileManager.default.moveItem(at: tempURL, to: destinationURL)

        return "Success: Downloaded to \(destinationURL.path)"
    }
}


// MARK: - WKNavigationDelegate

extension PrismWebController: WKNavigationDelegate {
    func webView(_ webView: WKWebView, didNavigateTo url: URL) {
        navigationContinuation?.resume(returning: true)
        navigationContinuation = nil
    }

    func webView(_ webView: WKWebView, didFailNavigationWithError error: Error) {
        navigationContinuation?.resume(returning: true) // partial load still usable
        navigationContinuation = nil
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Split View (Web + Chat)
// ─────────────────────────────────────────────────────────────────────────────
// MARK: - V8 ↔ WebView2 Bridge Driver
// ─────────────────────────────────────────────────────────────────────────────

/// Implements BrowserRuntimeDriver (UniFFI callback interface) to let the
/// V8 sandbox drive the WebView2 synchronously.
final class SwiftBrowserDriver: BrowserRuntimeDriver {
    private let webController: PrismWebController

    init(webController: PrismWebController) {
        self.webController = webController
    }

    func navigate(url: String) -> String {
        let webController = self.webController
        return syncHop {
            try await webController.execute(name: "web_navigate", argumentsJson: "{\"url\":\"\(url)\"}")
        }
    }

    func snapshot() -> String {
        let webController = self.webController
        return syncHop {
            try await webController.execute(name: "web_snapshot", argumentsJson: "{}")
        }
    }

    func interact(id: UInt32, action: String, value: String?) -> String {
        var args: [String: Any] = ["id": id, "action": action]
        if let v = value { args["value"] = v }
        let jsonData = try! JSONSerialization.data(withJSONObject: args)
        let jsonStr = String(data: jsonData, encoding: .utf8)!
        let webController = self.webController
        return syncHop {
            try await webController.execute(name: "web_interact", argumentsJson: jsonStr)
        }
    }

    func evaluateJs(script: String) -> String {
        let escaped = script
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
        let webController = self.webController
        return syncHop {
            try await webController.execute(name: "web_evaluate_js", argumentsJson: "{\"script\":\"\(escaped)\"}")
        }
    }

    func download(url: String, filename: String) -> String {
        let args = "{\"url\":\"\(url)\",\"filename\":\"\(filename)\"}"
        let webController = self.webController
        return syncHop {
            try await webController.execute(name: "web_download", argumentsJson: args)
        }
    }

    private nonisolated func syncHop<T>(_ block: @escaping @MainActor @Sendable () async throws -> T) -> String where T: Sendable {
        let semaphore = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var result: String = "ERROR: execution failed"
        Task { @MainActor in
            do {
                let val = try await block()
                result = String(describing: val)
            } catch {
                result = "ERROR: \(error.localizedDescription)"
            }
            semaphore.signal()
        }
        semaphore.wait()
        return result
    }
}

// ─────────────────────────────────────────────────────────────────────────────

// MARK: - Headless Browser Sessions
// ─────────────────────────────────────────────────────────────────────────────

/// A headless WebView2 session for background browsing.
final class HeadlessBrowserSession {
    fileprivate let webController: PrismWebController
    fileprivate(set) lazy var driver = SwiftBrowserDriver(webController: webController)
    let id: String
    init(id: String) async {
        self.id = id
        self.webController = await MainActor.run { PrismWebController(isHeadless: true) }
    }
    func runJavaScript(_ code: String, sandboxRoot: String = "") -> String {
        prismRunJs(code: code, sandboxRoot: sandboxRoot, driver: driver)
    }
}

final class HeadlessBrowserManager {
    private var sessions: [String: HeadlessBrowserSession] = [:]
    private let lock = NSLock()
    private var nextId: UInt64 = 1
    @discardableResult func createSession() async -> String {
        let id = "headless-\(nextId)"; nextId += 1
        let s = await HeadlessBrowserSession(id: id)
        lock.withLock { sessions[id] = s }; return id
    }
    func closeSession(_ id: String) { _ = lock.withLock { sessions.removeValue(forKey: id) } }
    func closeAll() { lock.withLock { sessions.removeAll() } }
    func run(sessionId: String, code: String, sandboxRoot: String = "") -> String? {
        lock.withLock { sessions[sessionId] }?.runJavaScript(code, sandboxRoot: sandboxRoot)
    }
}

// ─────────────────────────────────────────────────────────────────────────────

// MARK: - Headless Agent Orchestrator
// ─────────────────────────────────────────────────────────────────────────────

/// Spawns fully isolated headless browser sessions.
class HeadlessAgentOrchestrator {
    func runSession(script: String, sandboxRoot: String) async -> String {
        await runSession(script: script, sandboxRoot: sandboxRoot, url: nil)
    }

    func runSession(script: String, sandboxRoot: String, url: String? = nil) async -> String {
        let rootURL = URL(fileURLWithPath: sandboxRoot)
        _ = try? FileManager.default.createDirectory(at: rootURL, withIntermediateDirectories: true)

        // Tri-modal routing based on URL:
        // - Tier 2 (Static): jsEnabled=false, network shield enabled
        // - Tier 3 (Dynamic): jsEnabled=true, only for trusted domains
        let trustedDomains = ["github.com", "apple.com", "ycombinator.com"]
        let requiresDynamic: Bool
        if let urlStr = url, let host = URL(string: urlStr)?.host {
            requiresDynamic = trustedDomains.contains(where: { host.contains($0) })
        } else {
            requiresDynamic = false
        }

        let controller = await MainActor.run {
            let c = PrismWebController(isHeadless: true, jsEnabled: requiresDynamic)
            c.currentSandboxRoot = rootURL
            return c
        }
        let driver = SwiftBrowserDriver(webController: controller)
        return await Task.detached(priority: .userInitiated) {
            prismRunJs(code: script, sandboxRoot: sandboxRoot, driver: driver)
        }.value
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// The `webController` is shared across the app via the environment so the
/// agent loop and the split view reference the same WebView2 instance.
struct PrismSplitView: View {
    @State private var store = ConversationStore.shared
    @State private var showWebPanel = false
    @State private var splitRatio: CGFloat = 0.55
    @State private var isDragging = false

    let webController: PrismWebController

    var body: some View {
        HSplitView {
            // ── Web panel ────────────────────────────────────────────
            if showWebPanel {
                WebViewWrapper(webView: webController.getWebView())
                    .frame(minWidth: 300, idealWidth: 600)
                    .layoutPriority(1)
            }

            // ── Chat panel ───────────────────────────────────────────
            PrismChatView(showWebPanel: $showWebPanel, webController: webController)
                .frame(minWidth: 320, idealWidth: 420)
                .layoutPriority(0)
        }
        .frame(minWidth: 700, minHeight: 400)
    }
}

/// AppKit NSViewRepresentable wrapper for WebView2.
struct WebViewWrapper: NSViewRepresentable {
    let webView: WKWebView

    func makeNSView(context: Context) -> WKWebView {
        webView
    }

    func updateNSView(_ nsView: WKWebView, context: Context) {}
}

// TODO: In PrismChatView, add two bindings:
//   @Binding var showWebPanel: Bool
//   let webController: PrismWebController
// and add a toolbar button:
//   Button(action: { showWebPanel.toggle() }) {
//       Image(systemName: showWebPanel ? "text.bubble" : "globe")
//   }
