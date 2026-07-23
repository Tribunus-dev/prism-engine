import SwiftUI
import PrismCore

// MARK: - OrbPopover

/// Compact popover replacing the old 400px sidebar.
/// Shows agent status, micro-telemetry, recent threads, and quick settings access.
/// Triggered by tapping the orb in the nav bar.
struct OrbPopover: View {
    let session: AgentSession
    @State private var store = ConversationStore.shared
    @State private var voice = VoiceCaptureService.shared
    @StateObject private var engineController = PrismEngineController.shared
    @State private var hardware = PanelHardware()
    @State private var activeModel = "Gemma 4 12B"
    @State private var status = PanelStatus()
    var onSettingsTap: () -> Void
    @State private var contextService = ContextCaptureService.shared

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // ── Agent Status ─────────────────────────────────
            agentStatusSection
            Divider().padding(.vertical, 8)

            // ── Model Selector ────────────────────────────────
            modelSelectorSection
            Divider().padding(.vertical, 8)

            // ── Performance ──────────────────────────────────
            performanceSection
            Divider().padding(.vertical, 8)

            // ── Threads ──────────────────────────────────────
            threadSection
            Divider().padding(.vertical, 8)

            // ── Settings ─────────────────────────────────────
            Button(action: onSettingsTap) {
                Label("Settings", systemImage: "gearshape")
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.plain)
            .padding(.vertical, 4)
        }
        .padding(16)
        .frame(width: 300)
        .task {
            await contextService.captureContext()
        }
    }

    // MARK: - Agent Status

    private var agentStatusSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                OrbView(state: orbState, size: 36)

                VStack(alignment: .leading, spacing: 2) {
                    Text("Prism Agent")
                        .font(.system(.headline, design: .rounded))

                    Text(phaseLabel)
                        .font(.caption)
                        .foregroundColor(phaseColor)
                }

                Spacer()
            }

            if let modelRepo = UserDefaults.standard.string(forKey: "activeModelRepo") {
                LabeledContent("Model", value: modelRepo)
                    .font(.caption)
            }

            HStack(spacing: 12) {
                LabeledContent("Voice", value: voiceVoiceLabel)
                LabeledContent("Tone", value: "Neutral")
                LabeledContent("Speed", value: "1.0x")
            }
            .font(.caption2)
            .foregroundColor(.secondary)
        }
    }

    // MARK: - Model Selector

    private var modelSelectorSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Models").font(.caption).foregroundColor(.secondary)
            Gemma4ModelCard(
                modelName: "Gemma 4 12B",
                parameterCount: "12 billion parameters",
                isActive: activeModel == "Gemma 4 12B",
                onSelect: { activeModel = "Gemma 4 12B" }
            )
            Gemma4ModelCard(
                modelName: "Gemma 4 12B Unified",
                parameterCount: "12B params · multimodal",
                isActive: activeModel == "Gemma 4 12B Unified",
                onSelect: { activeModel = "Gemma 4 12B Unified" }
            )
        }
    }

    // MARK: - Performance

    private var performanceSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Hardware").font(.caption).foregroundColor(.secondary)
            HStack {
                VStack(alignment: .leading) {
                    Text("SLC").font(.caption2)
                    Text(hardware.slcUtilization.map { String(format: "%.0f%%", $0 * 100) } ?? "—")
                }
                Spacer()
                VStack(alignment: .trailing) {
                    Text("ANE").font(.caption2)
                    Text(hardware.aneUtilization.map { String(format: "%.0f%%", $0 * 100) } ?? "—")
                }
            }
        }
    }

    // MARK: - Threads

    private var threadSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Recent Threads")
                    .font(.caption.weight(.semibold))
                    .foregroundColor(.secondary)

                Spacer()

                Button {
                    Task { await store.createThread() }
                } label: {
                    Image(systemName: "plus")
                        .font(.caption)
                }
                .buttonStyle(.plain)
                .help("Create a new thread")
            }

            ScrollView {
                VStack(spacing: 4) {
                    ForEach(store.threads) { thread in
                        Button {
                            Task { try? await store.selectThread(id: thread.id) }
                        } label: {
                            HStack {
                                Text(thread.title)
                                    .font(.caption)
                                    .lineLimit(1)
                                Spacer()

                                if thread.id == store.selectedThreadID {
                                    Image(systemName: "checkmark")
                                        .font(.caption2)
                                        .foregroundColor(.accentColor)
                                }
                            }
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(
                                thread.id == store.selectedThreadID
                                    ? Color.accentColor.opacity(0.1)
                                    : Color.clear
                            )
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                        }
                        .buttonStyle(.plain)
                        .accessibilityIdentifier("popover-thread-\(thread.id)")
                        .contextMenu {
                            Button("Rename") {
                                // Rename handled via sheet in PrismConversationView
                            }

                            Button(role: .destructive) {
                                Task { await store.deleteThread(id: thread.id) }
                            } label: {
                                Label("Delete", systemImage: "trash")
                            }
                        }
                    }
                }
            }
            .frame(maxHeight: 160)
        }
    }

    // MARK: - Derived Values

    private var orbState: OrbView.OrbState {
        switch session.phase {
        case .idle: return .idle
        case .listening: return .listening(session.acousticPower)
        case .transcribing: return .processing(0.5)
        case .thinking: return .loading
        case .executing: return .processing(0.7)
        case .responding: return .speaking(session.acousticPower)
        case .showingContent: return .idle
        case .error: return .error
        }
    }

    private var phaseLabel: String {
        switch session.phase {
        case .idle: return "Ready"
        case .listening: return "Listening..."
        case .transcribing: return "Transcribing..."
        case .thinking: return "Thinking..."
        case .executing(let tool): return "Using \(tool.displayName)..."
        case .responding: return "Speaking"
        case .showingContent: return "Showing result"
        case .error(let msg): return msg
        }
    }

    private var phaseColor: Color {
        switch session.phase {
        case .idle: return .secondary
        case .listening: return .accentColor
        case .transcribing: return .orange
        case .thinking: return .blue
        case .executing: return .purple
        case .responding: return .green
        case .showingContent: return .secondary
        case .error: return .red
        }
    }

    private var voiceVoiceLabel: String {
        switch voice.state {
        case .idle: return "Ready"
        case .listening: return "Listening"
        case .transcribing: return "Active"
        case .responding: return "Speaking"
        }
    }
}
