import SwiftUI
import PrismCore
import CryptoKit

// MARK: - Conversation Settings Sheet

struct ConversationSettingsSheet: View {
    @Binding var activeModel: String
    let engineController: PrismEngineController
    let downloader: ModelDownloader
    let voice: VoiceCaptureService
    let audioMatrix: PrismAudioMatrix
    let accessibilityEngine: AccessibilityEngine
    let screenSlayer: PrismScreenSlayer
    let homeKit: PrismHomeKitController
    let status: PanelStatus
    let auth: PanelAuth
    let hardware: PanelHardware

    private var models: [String] {
        engineController.daemonModels.isEmpty
            ? ["Gemma 4 12B", "Gemma 4 12B Unified"]
            : engineController.daemonModels
    }

    private var orbState: OrbView.OrbState {
        switch voice.state {
        case .idle, .listening: break
        case .transcribing: return .processing(0.5)
        case .responding: return .speaking(voice.acousticPower)
        }
        if engineController.isRunning { return .processing(0.5) }
        else if engineController.isCompiling { return .loading }
        else { return .idle }
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    // Header with Orb
                    HStack {
                        OrbView(state: orbState, size: 32)
                        Text("Prism Agent")
                            .font(.system(.title3, design: .rounded))
                        Spacer()
                    }
                    .padding(.horizontal)

                    // Auth
                    HuggingFaceLoginView(hfToken: Binding(
                        get: { auth.hfToken },
                        set: { auth.hfToken = $0 }
                    ), isAuthenticated: Binding(
                        get: { auth.isAuthenticated },
                        set: { auth.isAuthenticated = $0 }
                    ))

                    if auth.isAuthenticated {
                        // Model picker
                        modelPickerSection

                        // Download
                        if downloader.status != "Idle" {
                            downloadProgressSection
                        } else {
                            downloadButtonSection
                        }

                        // Runtime stats
                        runtimeStatsSection

                        // Hardware utilization bars
                        hardwareUtilSection
                    }

                    // Feature toggles
                    settingsSection

                    progressiveModelSection

                    voiceSection

                    // Action buttons
                    actionButtonsSection
                }
                .padding(.horizontal)
                .padding(.vertical)
            }
            .navigationDestination(for: TTSDestination.self) { _ in
                TTSSettingsView()
            }
            .task { await engineController.refreshDaemonModels() }
        }
    }

    // MARK: - Model Picker

    @ViewBuilder
    private var modelPickerSection: some View {
        HStack {
            Text("Model").foregroundColor(.secondary)
            Spacer()
            Picker("", selection: $activeModel) {
                ForEach(models, id: \.self) { model in Text(model).tag(model) }
            }
            .labelsHidden().frame(width: 180)
        }
        .padding(.horizontal)
.background(PrismTheme.prismGlass(cornerRadius: 12))
.clipShape(RoundedRectangle(cornerRadius: 12))
    }

    // MARK: - Download Progress

    @ViewBuilder
    private var downloadProgressSection: some View {
        GroupBox(label: Text("Download").font(.caption)) {
            VStack(spacing: 4) {
                ProgressView(value: downloader.progress)
                HStack {
                    Text(downloader.status).font(.caption)
                    Spacer()
                    Text("\(Int(downloader.progress * 100))%").font(.caption.monospaced())
                }
            }
            .padding(4)
        }
.background(PrismTheme.prismGlass(cornerRadius: 12))
.clipShape(RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal)
    }

    // MARK: - Download Button

    @ViewBuilder
    private var downloadButtonSection: some View {
        HStack {
            Button("Download 12B Weights") {
                downloader.setToken(auth.hfToken)
                let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
                let destination = appSupport
                    .appendingPathComponent("downloads/google--gemma-4-12b-unified/model.safetensors")
                downloader.downloadModel(
                    repo: "google/gemma-4-12b-unified",
                    filename: "model.safetensors",
                    to: destination
                )
            }
            .buttonStyle(.borderedProminent)
            .disabled(!auth.isAuthenticated)
        }
        .padding(.horizontal)
.background(PrismTheme.prismGlass(cornerRadius: 12))
.clipShape(RoundedRectangle(cornerRadius: 12))
    }

    // MARK: - Runtime Stats

    @ViewBuilder
    private var runtimeStatsSection: some View {
        GroupBox(label: Text("Runtime").font(.caption)) {
            VStack(spacing: 6) {
                StatRow(label: "Status", value: status.statusText)
                StatRow(label: "Tokens", value: "\(status.tokensProcessed)")
                StatRow(label: "Agents", value: "\(status.activeAgents)/32")
            }
            .padding(4)
        }
.background(PrismTheme.prismGlass(cornerRadius: 12))
.clipShape(RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal)
    }

    // MARK: - Hardware Utilization

    @ViewBuilder
    private var hardwareUtilSection: some View {
        GroupBox(label: Text("Hardware").font(.caption)) {
            VStack(spacing: 6) {
                UtilRow(label: "SLC Cache", value: hardware.slcUtilization)
                UtilRow(label: "ANE", value: hardware.aneUtilization)
                StatRow(label: "CPU", value: "\(Int(hardware.cpuUtilization))%")
                StatRow(label: "Thermal", value: hardware.thermalState)
            }
            .padding(4)
        }
.background(PrismTheme.prismGlass(cornerRadius: 12))
.clipShape(RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal)
    }

    // MARK: - Feature Toggles

    @ViewBuilder
    private var settingsSection: some View {
        GroupBox(label: Text("Settings").font(.caption)) {
            VStack(spacing: 6) {
                Toggle("HomeKit Automation", isOn: Binding(
                    get: { homeKit.isReady },
                    set: { _ in }
                ))
                .toggleStyle(.switch)
                .font(.caption)
.prismBorder(cornerRadius: 8)

                Toggle("Screen Monitor", isOn: Binding(
                    get: { screenSlayer.isCapturing },
                    set: { enabled in
                        Task {
                            if enabled {
                                try? await screenSlayer.startCapture()
                            } else {
                                await screenSlayer.stopCapture()
                            }
                        }
                    }
                ))
                .toggleStyle(.switch)
                .font(.caption)
.prismBorder(cornerRadius: 8)

                Toggle("Accessibility Scan", isOn: Binding(
                    get: { accessibilityEngine.elements.count > 0 },
                    set: { enabled in
                        if enabled { accessibilityEngine.scanActiveApplication() }
                    }
                ))
                .toggleStyle(.switch)
                .font(.caption)
.prismBorder(cornerRadius: 8)

                Toggle("Spatial Audio", isOn: Binding(
                    get: { audioMatrix.isActive },
                    set: { enabled in
                        if enabled {
                            try? audioMatrix.start()
                        } else {
                            audioMatrix.stop()
                        }
                    }
                ))
                .toggleStyle(.switch)
                .font(.caption)
.prismBorder(cornerRadius: 8)
            }
            .padding(4)
        }
.background(PrismTheme.prismGlass(cornerRadius: 12))
.clipShape(RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal)
    }

    // MARK: - Voice

    @ViewBuilder
    private var voiceSection: some View {
        GroupBox(label: Text("Voice").font(.caption)) {
            VStack(spacing: 8) {
                HStack {
                    OrbView(state: orbState, size: 40)
                    VStack(alignment: .leading) {
                        Text("Voice Input")
                            .font(.caption)
                        Text(voiceStateLabel)
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                    Spacer()
                    Toggle(isOn: Binding(
                        get: { voice.isListening },
                        set: { enabled in
                            if enabled { voice.startListening() }
                            else { voice.stopListening() }
                        }
                    )) { EmptyView() }
                    .toggleStyle(.switch)
                }

                if voice.isListening || voice.state == .transcribing {
                    VoiceWaveformView(waveform: voice.waveformBuffer)
                        .frame(height: 32)
                }

                Divider()
                NavigationLink(value: TTSDestination()) {
                    HStack {
                        Image(systemName: "speaker.wave.2")
                        Text("Voice & Speech")
                        Spacer()
                    }
                    .font(.caption)
                }
                .buttonStyle(.plain)
.accessibilityHint("Opens voice settings")
            }
            .padding(4)
        }
.background(PrismTheme.prismGlass(cornerRadius: 12))
.clipShape(RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal)
    }

    private var voiceStateLabel: String {
        switch voice.state {
        case .idle: return "Tap mic to start"
        case .listening: return "Listening..."
        case .transcribing: return "Transcribing..."
        case .responding: return "Speaking"
        }
    }

    // MARK: - Progressive Model

    @ViewBuilder
    private var progressiveModelSection: some View {
        GroupBox(label: Text("Progressive Model").font(.caption)) {
            VStack(spacing: 8) {
                NavigationLink(destination: ContributionSettingsView()) {
                    HStack {
                        Image(systemName: "sparkles.rectangle.stack")
                        Text("Contribution Settings")
                        Spacer()
                    }
                    .font(.caption)
                }
                .buttonStyle(.plain)
                .accessibilityHint("Opens contribution settings")

                Divider()

                NavigationLink(destination: ContributionActivityView()) {
                    HStack {
                        Image(systemName: "chart.bar.fill")
                        Text("Contribution Activity")
                        Spacer()
                    }
                    .font(.caption)
                }
                .buttonStyle(.plain)
                .accessibilityHint("Opens contribution activity")
            }
            .padding(4)
        }
.background(PrismTheme.prismGlass(cornerRadius: 12))
.clipShape(RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal)
    }

    // MARK: - Action Buttons

    @ViewBuilder
    private var actionButtonsSection: some View {
        HStack {
            if auth.isAuthenticated {
                Button("Compile") {
                    Task { try? await engineController.compileDownloadedWeights() }
                    status.statusText = "Compiling\u{2026}"
                }
                .buttonStyle(.borderedProminent)
                .disabled(engineController.isCompiling)

                Button("Boot") {
                    do {
                        try engineController.bootEngineRuntime()
                        status.statusText = "Running"
                    } catch {
                        status.statusText = "Boot failed"
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(engineController.isRunning)

                Button("Stop") {
                    status.statusText = "Idle"
                }
                .buttonStyle(.bordered)
            }
            Spacer()
            Button("Quit") { NSApp.terminate(nil) }
                .foregroundColor(.red)
        }
        .padding(.horizontal)
        .padding(.bottom, 8)
    }
}

// MARK: - Navigation Destination

struct TTSDestination: Hashable {}
