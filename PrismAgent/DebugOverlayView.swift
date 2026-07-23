import SwiftUI
import PrismCore

// MARK: - DebugOverlayView

/// Rich diagnostic HUD triggered by ⌘D.
/// Shows full hardware utilization, conversation stats, and tool dispatch history.
/// The micro-telemetry in the orb popover shows a subset; this shows everything.
struct DebugOverlayView: View {
    @State private var hardware = PanelHardware()
    @State private var status = PanelStatus()

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Header
            HStack {
                Label("Debug Overlay", systemImage: "ladybug")
                    .font(.headline)
                Spacer()
                Text("⌘D to dismiss")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Divider()

            // Hardware
            GroupBox(label: Label("Hardware", systemImage: "cpu").font(.caption)) {
                VStack(spacing: 8) {
                    UtilRow(label: "SLC", value: hardware.slcUtilization)
                    UtilRow(label: "ANE", value: hardware.aneUtilization)
                    UtilRow(label: "GPU", value: nil)
                    UtilRow(label: "CPU", value: hardware.cpuUtilization)

                    Divider()

                    HStack {
                        StatRow(label: "Tokens/sec", value: "\(status.tokensProcessed)")
                        StatRow(label: "Active Agents", value: "\(status.activeAgents)")
                        StatRow(label: "Memory", value: "Unavailable")
                    }
                    .font(.caption2)
                }
                .padding(4)
            }

            // Agent State
            GroupBox(label: Label("Agent State", systemImage: "brain").font(.caption)) {
                VStack(spacing: 6) {
                    StatRow(label: "Phase", value: String(describing: type(of: AgentSession().phase)))
                    StatRow(label: "Thread", value: ConversationStore.shared.currentThreadTitle)
                    StatRow(label: "Messages", value: "\(ConversationStore.shared.visibleMessages.count)")
                }
                .padding(4)
            }

            // Voice Pipeline
            GroupBox(label: Label("Voice Pipeline", systemImage: "waveform").font(.caption)) {
                VStack(spacing: 6) {
                    let voice = VoiceCaptureService.shared
                    StatRow(label: "Voice State", value: String(describing: voice.state))
                    StatRow(label: "Acoustic Power", value: String(format: "%.3f", voice.acousticPower))
                    StatRow(label: "Buffer Size", value: "\(voice.waveformBuffer.count)")
                }
                .padding(4)
            }
        }
        .padding(20)
        .frame(width: 360)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .stroke(Color.white.opacity(0.08), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.2), radius: 20, x: 0, y: 10)
    }
}
