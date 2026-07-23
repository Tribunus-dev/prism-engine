import ScreenCaptureKit
import PrismCore
import SwiftUI
import PHASE

// MARK: - OrbState alias

typealias OrbState = OrbView.OrbState

// MARK: - Decomposed Observable State

@Observable
final class PanelStatus {
    var statusText: String = "Idle"
    var tokensProcessed: Int = 0
    var activeAgents: Int = 0
}

@Observable
final class PanelHardware {
    var cpuUtilization: Double = 0
    var slcUtilization: Double? = nil
    var aneUtilization: Double? = nil
    var thermalState: String = "Unknown"
}

@Observable
final class PanelAuth {
    var isAuthenticated: Bool = false
    var hfToken: String = ""
}

// MARK: - Main Panel View

struct PrismControlPanelView: View {
    @State private var activeModel: String = "Gemma 4 12B"

    @State private var status = PanelStatus()
    @State private var hardware = PanelHardware()
    @State private var resourceTracker = ResourceTracker.shared
    @State private var auth = PanelAuth()

    @StateObject private var engineController = PrismEngineController()
    @StateObject private var downloader = ModelDownloader()
    @State private var voice = VoiceCaptureService()
    @StateObject private var audioMatrix = PrismAudioMatrix()
    @StateObject private var accessibilityEngine = AccessibilityEngine()
    @StateObject private var screenSlayer = PrismScreenSlayer()
    @StateObject private var homeKit = PrismHomeKitController.shared

    @State private var store = ConversationStore.shared
    @State private var showSettings = false
    @State private var showWebPanel = false
    @State private var showHardware = false
    @State private var showCompilerLab = false

    let models = ["Gemma 4 12B", "Gemma 4 12B Unified"]

    private var appDelegate: AppDelegate? {
        NSApplication.shared.delegate as? AppDelegate
    }

    private var orbState: OrbState {
        switch voice.state {
        case .idle, .listening: break
        case .transcribing: return .processing(0.5)
        case .responding: return .speaking(voice.acousticPower)
        }
        if engineController.isRunning { return .processing(0.5) }
        else if engineController.isCompiling { return .loading }
        else { return .idle }
    }

    private var isAnimating: Bool {
        voice.state == .listening
            || voice.state == .transcribing
            || voice.state == .responding
            || engineController.isRunning
            || engineController.isCompiling
    }

    var body: some View {
        VStack(spacing: 0) {
            chromeBar

            if voice.state == .listening || voice.state == .transcribing {
                VoiceWaveformView(waveform: voice.waveformBuffer)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 4)
                    .transition(.move(edge: .top).combined(with: .opacity))
            }

            if showCompilerLab {
                CompilerLabView(engineController: engineController)
            } else if showSettings {
                ConversationSettingsSheet(
                    activeModel: $activeModel,
                    engineController: engineController,
                    downloader: downloader,
                    voice: voice,
                    audioMatrix: audioMatrix,
                    accessibilityEngine: accessibilityEngine,
                    screenSlayer: screenSlayer,
                    homeKit: homeKit,
                    status: status,
                    auth: auth,
                    hardware: hardware
                )
            } else {
                PrismChatView(showWebPanel: $showWebPanel, webController: PrismWebController())
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }

            if showHardware {
                HardwarePulseStrip(
                    slcUtilization: hardware.slcUtilization ?? 0,
                    aneUtilization: hardware.aneUtilization ?? 0,
                    tokensPerSecond: Double(status.tokensProcessed)
                )
                .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .frame(width: 400)
        .frame(minHeight: 400, idealHeight: 600)
        .onChange(of: orbState) { _, newValue in
            switch newValue {
            case .idle, .error:
                appDelegate?.setProcessing(false)
            case .loading, .listening, .processing, .speaking, .remoteConnected, .multiDevice:
                appDelegate?.setProcessing(true)
            }
        }
        .task {
            resourceTracker.start(pollInterval: 2)
            while !Task.isCancelled {
                hardware.cpuUtilization = resourceTracker.cpuPercent / 100
                hardware.aneUtilization = resourceTracker.aneUtilization
                hardware.slcUtilization = resourceTracker.slcUtilization
                hardware.thermalState = String(describing: resourceTracker.thermalState)
                try? await Task.sleep(nanoseconds: 2_000_000_000)
            }
            resourceTracker.stop()
        }
    }

    // MARK: - Chrome Bar

    private var chromeBar: some View {
        HStack(spacing: 8) {
            OrbView(state: orbState, size: 18)

            Text("Prism Agent")
                .font(.system(.headline, design: .rounded))

            Spacer()

            Circle()
                .fill(statusColor)
                .frame(width: 6, height: 6)

            Button(action: { withAnimation { showHardware.toggle() } }) {
                Image(systemName: "chart.bar.xaxis")
                    .foregroundColor(showHardware ? .accentColor : .secondary)
            }
            .buttonStyle(.plain)
            .help("Show hardware utilization")

            Button(action: { withAnimation { showCompilerLab.toggle() } }) {
                Image(systemName: "chart.xyaxis.line")
                    .foregroundColor(showCompilerLab ? .accentColor : .secondary)
            }
            .buttonStyle(.plain)
            .help("Open Compiler Lab")

            Button(action: { showSettings.toggle() }) {
                Image(systemName: "gearshape")
                    .foregroundColor(.secondary)
            }
            .buttonStyle(.plain)
            .help("Settings")

            VoiceButton(state: voice.state) {
                if voice.isListening {
                    voice.stopListening()
                } else {
                    voice.startListening()
                }
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial)
    }

    private var statusColor: Color {
        if engineController.isRunning { .green }
        else if engineController.isCompiling { .orange }
        else { .gray }
    }
}

// MARK: - Sub-views for SettingsSheet (StatRow, UtilRow)

struct StatRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label).foregroundColor(.secondary).font(.caption)
            Spacer()
            Text(value).font(.caption.monospaced())
        }
    }
}

struct UtilRow: View {
    let label: String
    let value: Double?

    var body: some View {
        VStack(spacing: 2) {
            HStack {
                Text(label).foregroundColor(.secondary).font(.caption)
                Spacer()
                Text(value.map { "\(Int($0 * 100))%" } ?? "Unavailable").font(.caption.monospaced())
            }
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 2)
                        .fill(Color.gray.opacity(0.2))
                        .frame(height: 6)
                    RoundedRectangle(cornerRadius: 2)
                        .fill(value == nil ? Color.gray : value! > 0.8 ? Color.green : value! > 0.5 ? Color.yellow : Color.gray)
                        .frame(width: geo.size.width * (value ?? 0), height: 6)
                }
            }
            .frame(height: 6)
        }
    }
}
