import SwiftUI
import AppKit
import PrismCore
import AVFoundation
import CoreImage
import IOSurface
import Speech

// MARK: - PrismChatView

struct PrismChatView: View {
    @State private var store = ConversationStore.shared
    @State private var inputText: String = ""
    @State private var attachedImage: NSImage? = nil
    @State private var isTargetedForDrop: Bool = false
    @State private var isStreaming: Bool = false
    @State private var inferenceSpeed: Double? = nil
    @State private var voiceState: VoiceState = .idle
    @State private var audioCapture = AudioCaptureService()
    @State private var speechSynth = AVSpeechSynthesizer()
    @State private var speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    @State private var recognitionTask: SFSpeechRecognitionTask?
    @State private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    @Binding var showWebPanel: Bool
    let webController: PrismWebController

    var body: some View {
        VStack(spacing: 0) {
            // Toolbar
            HStack {
                Button(action: { showWebPanel.toggle() }) {
                    Label(showWebPanel ? "Chat" : "Web", systemImage: showWebPanel ? "text.bubble" : "globe")
                }
                .buttonStyle(.borderless)
                .help("Toggle web browser panel")
                Spacer()
            }
            .padding(.horizontal)
            .padding(.top, 4)

            // Content
            if store.visibleMessages.isEmpty {
                emptyState
            } else {
                chatArea
            }

            // Input bar
            inputBar
        }
    }

    // MARK: - Input Bar

    @ViewBuilder
    private var inputBar: some View {
        HStack(alignment: .bottom, spacing: 8) {
            // Voice button
            VoiceButton(state: voiceState) {
                toggleVoiceInput()
            }

            if let img = attachedImage {
                Image(nsImage: img)
                    .resizable()
                    .scaledToFill()
                    .frame(width: 48, height: 48)
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .overlay(alignment: .topTrailing) {
                        Button(action: { attachedImage = nil }) {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundColor(.white)
                                .background(Circle().fill(.black.opacity(0.5)))
                        }
                        .buttonStyle(.plain)
                        .offset(x: 8, y: -8)
                    }
            }

            TextField("Message Prism...", text: $inputText, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...6)
                .padding(8)
                .background(Color(nsColor: .controlBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .onSubmit(submitMessage)

            Button(action: submitMessage) {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.title2)
            }
            .buttonStyle(.plain)
            .disabled(inputText.isEmpty && attachedImage == nil)
        }
        .padding()
        .background(Color(nsColor: .windowBackgroundColor))
        .onDrop(of: [.image], isTargeted: $isTargetedForDrop) { providers in
            guard let provider = providers.first else { return false }
            provider.loadObject(ofClass: NSImage.self) { img, _ in
                if let image = img as? NSImage {
                    DispatchQueue.main.async { attachedImage = image }
                }
            }
            return true
        }
        .overlay(isTargetedForDrop ?
            RoundedRectangle(cornerRadius: 8).stroke(Color.blue, lineWidth: 2) : nil)
    }

    // MARK: - Empty State

    @ViewBuilder
    private var emptyState: some View {
        VStack(spacing: 12) {
            Spacer()

            Text("Prism")
                .font(.system(size: 34, weight: .light))
                .foregroundColor(.primary)

            Text("Your private AI assistant")
                .font(.system(size: 12))
                .foregroundColor(.secondary)

            VStack(spacing: 10) {
                HStack(spacing: 10) {
                    CapabilityChip(icon: "doc.text", label: "Summarize this", prompt: "Summarize this") { text in
                        inputText = text
                    }
                    CapabilityChip(icon: "globe", label: "Research a topic", prompt: "Research a topic") { text in
                        inputText = text
                    }
                }
                HStack(spacing: 10) {
                    CapabilityChip(icon: "pencil", label: "Help me write", prompt: "Help me write") { text in
                        inputText = text
                    }
                    CapabilityChip(icon: "camera", label: "Analyze an image", prompt: "Analyze an image") { text in
                        inputText = text
                    }
                }
            }

            Spacer()
        }
        .padding(.horizontal, 20)
    }

    // MARK: - Chat Area

    @ViewBuilder
    private var chatArea: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 8) {
                    Color.clear
                        .frame(height: 1)
                        .onAppear {
                            Task { try? await store.loadEarlier() }
                        }

                    ForEach(Array(zip(store.visibleMessages.indices, store.visibleMessages)), id: \.1.id) { index, msg in
                        let showTimestamp = index > 0 && msg.timestamp.timeIntervalSince(store.visibleMessages[index - 1].timestamp) > 300
                        MessageBubble(message: msg, showTimestamp: showTimestamp)
                            .id(msg.id)
                    }

                    if isStreaming {
                        StreamingDot()
                            .id("streaming-indicator")
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal)
                    }
                }
                .padding()
            }
            .overlay(alignment: .topTrailing) {
                if let speed = inferenceSpeed {
                    let speedText = String(format: "%.0f tok/s", speed)
                    Text(speedText)
                        .font(.caption2)
                        .monospacedDigit()
                        .foregroundStyle(.tertiary)
                        .padding(6)
                }
            }
            .onChange(of: store.visibleMessages.count) { _, _ in
                if let targetID = store.lastScrollTarget {
                    withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                        proxy.scrollTo(targetID, anchor: .bottom)
                    }
                }
            }
        }
    }

    // MARK: - Submission

    func submitMessage() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        let image = attachedImage
        guard !text.isEmpty || image != nil else { return }

        inputText = ""
        attachedImage = nil

        Task {
            try? await store.append(role: .user, text: text, image: image)
        }
    }
    // MARK: - Voice Input

    func toggleVoiceInput() {
        switch voiceState {
        case .idle: startRecording()
        case .listening: stopRecording()
        case .transcribing, .responding: break
        }
    }

    private func startRecording() {
        guard let recognizer = speechRecognizer, recognizer.isAvailable else { return }
        SFSpeechRecognizer.requestAuthorization { status in
            guard status == .authorized else { return }
            Task { @MainActor in
                self.voiceState = .listening
                try self.audioCapture.start()
                self.recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
                guard let request = self.recognitionRequest else { return }
                request.shouldReportPartialResults = true
                self.recognitionTask = recognizer.recognitionTask(with: request) { result, error in
                    Task { @MainActor in
                        if let result = result {
                            self.inputText = result.bestTranscription.formattedString
                        }
                        if result?.isFinal == true || error != nil {
                            self.voiceState = .transcribing
                            self.audioCapture.stop()
                            self.submitVoiceText(self.inputText)
                        }
                    }
                }
            }
        }
    }

    private func stopRecording() {
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        audioCapture.stop()
        voiceState = .idle
    }

    private func submitVoiceText(_ text: String) {
        guard !text.trimmingCharacters(in: .whitespaces).isEmpty else {
            voiceState = .idle; return
        }
        Task {
            try? await store.append(role: .user, text: text, image: nil)
            voiceState = .listening
            let handler = StreamHandler()
            handler.attach(textUpdate: { token in }, complete: { fullText, didReceiveAudio in
                Task { @MainActor in
                    try? await store.append(role: .assistant, text: fullText, image: nil)
                    if didReceiveAudio {
                        voiceState = .idle
                    } else {
                        voiceState = .responding
                        let utterance = AVSpeechUtterance(string: fullText)
                        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
                        utterance.rate = 0.5
                        speechSynth.speak(utterance)
                        while speechSynth.isSpeaking {
                            try? await Task.sleep(nanoseconds: 100_000_000)
                        }
                        voiceState = .idle
                    }
                }
            }, error: { error in
                Task { @MainActor in voiceState = .idle }
            })
            let adapter = StreamCallbackAdapter(handler: handler)
            let engine = PrismEngineController.shared
            guard FileManager.default.fileExists(atPath: engine.activeCImagePath) else {
                Task { @MainActor in
                    voiceState = .idle
                    try? await store.append(role: .assistant, text: "No compiled Prism model is loaded yet. Open Model Management, compile a model, and try again.", image: nil)
                }
                return
            }
            prismInferMultimodalStream(
                cimagePath: engine.activeCImagePath,
                modelDir: engine.activeModelDirectory,
                prompt: text,
                callback: adapter
            )
        }
    }
}

// MARK: - MessageBubble

struct MessageBubble: View {
    let message: StoredMessage
    var showTimestamp: Bool = false

    @State private var displayedImage: NSImage? = nil
    @State private var isHovering = false
    @State private var isAppearing = false

    var body: some View {
        VStack(spacing: 4) {
            // Timestamp divider
            if showTimestamp {
                timestampDivider
            }

            HStack(alignment: .bottom, spacing: 8) {
                if message.role == .assistant {
                    Spacer(minLength: 60)
                }

                bubbleContent
                    .background(backgroundView)
                    .clipShape(RoundedRectangle(cornerRadius: 14))
                    .overlay(alignment: message.role == .user ? .trailing : .leading) {
                        BubbleTail(leading: message.role == .assistant)
                            .fill(message.role == .user
                                  ? AnyShapeStyle(LinearGradient(
                                    colors: [Color.accentColor.opacity(0.2), Color.accentColor.opacity(0.1)],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing))
                                  : AnyShapeStyle(.regularMaterial))
                            .frame(width: 10, height: 16)
                            .offset(x: message.role == .user ? 5 : -5)
                            .shadow(color: .black.opacity(0.06), radius: 4, x: message.role == .user ? 2 : -2, y: 2)
                    }
                    .shadow(color: .black.opacity(0.06), radius: isHovering ? 8 : 4, x: 0, y: isHovering ? 4 : 2)
                    .scaleEffect(message.role == .assistant ? (isHovering ? 1.02 : 1.0) : 1.0)
                    .animation(.easeOut(duration: 0.2), value: isHovering)
                    .onHover { hovering in
                        isHovering = hovering
                    }
                    .opacity(isAppearing ? 1 : 0)
                    .offset(y: isAppearing ? 0 : 10)
                    .onAppear {
                        withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) {
                            isAppearing = true
                        }
                    }

                if message.role == .user {
                    Spacer(minLength: 60)
                }
            }
        }
        .task {
            if let attachment = message.imageAttachment {
                displayedImage = await ConversationStore.shared.retrieveImage(attachment)
            }
        }
    }

    // MARK: - Background

    @ViewBuilder
    private var backgroundView: some View {
        if message.role == .user {
            LinearGradient(
                colors: [Color.accentColor.opacity(0.2), Color.accentColor.opacity(0.1)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        } else {
            Rectangle().fill(.regularMaterial)
        }
    }

    // MARK: - Bubble Content

    @ViewBuilder
    private var bubbleContent: some View {
        if message.role == .tool {
            ToolMessageView(text: message.text, role: "tool")
                .frame(maxWidth: 420, alignment: .leading)
                .padding(4)
        } else {
            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 4) {
                if let img = displayedImage {
                    Image(nsImage: img)
                        .resizable()
                        .scaledToFit()
                        .frame(maxHeight: 200)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                }
                if !message.text.isEmpty {
                    if message.role == .assistant && (message.text.contains("```tool_call") || message.text.contains("read_file") || message.text.contains("web_")) {
                        ToolMessageView(text: message.text, role: "assistant")
        } else {
                    Text(message.text)
                        .textSelection(.enabled)
                        .foregroundColor(message.role == .user ? .primary : .primary)
                }
                }
            }
            .padding(12)
        }
    }

    // MARK: - Timestamp Divider

    private var timestampDivider: some View {
        HStack(spacing: 8) {
            Rectangle()
                .fill(.separator.opacity(0.3))
                .frame(height: 1)
            Text(message.timestamp.formatted(date: .abbreviated, time: .shortened))
                .font(.caption2)
                .foregroundColor(.secondary)
            Rectangle()
                .fill(.separator.opacity(0.3))
                .frame(height: 1)
        }
        .padding(.horizontal, 40)
        .padding(.vertical, 4)
    }
}

// MARK: - Bubble Tail Shape

struct BubbleTail: Shape {
    let leading: Bool

    func path(in rect: CGRect) -> Path {
        var path = Path()
        let midY = rect.midY
        let tail: CGFloat = 4
        if leading {
            // Left-pointing triangle
            path.move(to: CGPoint(x: rect.maxX, y: midY - tail))
            path.addLine(to: CGPoint(x: rect.minX, y: midY))
            path.addLine(to: CGPoint(x: rect.maxX, y: midY + tail))
        } else {
            // Right-pointing triangle
            path.move(to: CGPoint(x: rect.minX, y: midY - tail))
            path.addLine(to: CGPoint(x: rect.maxX, y: midY))
            path.addLine(to: CGPoint(x: rect.minX, y: midY + tail))
        }
        path.closeSubpath()
        return path
    }
}

// MARK: - Streaming Dot

struct StreamingDot: View {
    @State private var opacity: Double = 0.3

    var body: some View {
        Circle()
            .fill(Color.secondary)
            .frame(width: 4, height: 4)
            .opacity(opacity)
            .onAppear {
                withAnimation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true)) {
                    opacity = 1.0
                }
            }
    }
}

// MARK: - Capability Chip

struct CapabilityChip: View {
    let icon: String
    let label: String
    let prompt: String
    let onFill: (String) -> Void

    var body: some View {
        Button(action: { onFill(prompt) }) {
            Label(label, systemImage: icon)
                .font(.subheadline)
                .foregroundColor(.white)
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
        }
        .buttonStyle(.borderedProminent)
        .tint(Color.accentColor.opacity(0.15))
        .buttonBorderShape(.capsule)
    }
}
