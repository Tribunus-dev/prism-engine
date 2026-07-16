import SwiftUI

/// Remote device view with glass header, agent reasoning overlay,
/// live video layer, and touch forwarding.
struct RemoteView: View {
    // MARK: - Agent State
    @State private var reasoning: String = ""
    @State private var agentPhase: String = ""
    @State private var isAgentActive: Bool = false

    // MARK: - Bridge state
    @State private var bridge = RemoteViewBridge.shared

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VStack(spacing: 0) {
                // ── Glass Header ──
                glassHeader

                // ── Agent Reasoning Overlay ──
                if isAgentActive {
                    agentPanel
                        .transition(.move(edge: .top).combined(with: .opacity))
                }

                // ── Video Layer ──
                videoLayer
                    .frame(maxWidth: .infinity, maxHeight: .infinity)

                // ── Touch Forwarding Layer ──
                touchForwardingLayer
            }
        }
        .onAppear(perform: subscribeToEvents)
    }

    // MARK: - Glass Header

    private var glassHeader: some View {
        HStack {
            Text("Remote Agent")
                .font(.system(size: 14, weight: .semibold))
                .foregroundColor(.white)

            Spacer()

            Button(action: { isAgentActive.toggle() }) {
                Image(systemName: isAgentActive ? "antenna.radiowaves.left.and.right" : "antenna.radiowaves.left.and.right.slash")
                    .font(.system(size: 12))
                    .foregroundColor(isAgentActive ? .green : .white.opacity(0.5))
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial)
    }

    // MARK: - Agent Reasoning Panel

    private var agentPanel: some View {
        VStack(spacing: 2) {
            // Phase indicator
            HStack(spacing: 6) {
                Circle()
                    .fill(phaseColor())
                    .frame(width: 4, height: 4)
                Text(agentPhase)
                    .font(.system(size: 9, weight: .medium))
                    .foregroundColor(.white.opacity(0.7))
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            // Reasoning text (matching the glass dark theme)
            if !reasoning.isEmpty {
                Text(reasoning)
                    .font(.system(size: 8))
                    .foregroundColor(.white.opacity(0.5))
                    .lineLimit(3)
                    .truncationMode(.tail)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 4)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial)
    }

    // MARK: - Video Layer

    private var videoLayer: some View {
        ZStack {
            Color.black

            VStack(spacing: 8) {
                Image(systemName: "video.fill")
                    .font(.system(size: 32))
                    .foregroundColor(.white.opacity(0.15))
                Text("Remote stream")
                    .font(.system(size: 11))
                    .foregroundColor(.white.opacity(0.25))
            }
        }
    }

    // MARK: - Touch Forwarding Layer

    private var touchForwardingLayer: some View {
        Color.clear
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onEnded { value in
                        forwardTouch(at: value.location)
                    }
            )
    }

    // MARK: - Color Helper

    private func phaseColor() -> Color {
        switch agentPhase {
        case "Thinking":   return .blue
        case "Using tool": return .orange
        case "Observing":  return .purple
        case "Done":       return .green
        default:           return .gray
        }
    }

    // MARK: - P2P Events

    private func subscribeToEvents() {
        bridge.onEvent("agent/reasoning") { _, payload in
            if let payload, let text = String(data: payload, encoding: .utf8) {
                withAnimation(.easeInOut(duration: 0.15)) {
                    reasoning = text
                    isAgentActive = true
                }
            }
        }
        bridge.onEvent("agent/tool_call") { _, _ in
            withAnimation(.easeInOut(duration: 0.15)) {
                agentPhase = "Using tool"
                isAgentActive = true
            }
        }
        bridge.onEvent("agent/complete") { _, _ in
            withAnimation(.easeInOut(duration: 0.25)) {
                agentPhase = "Done"
                isAgentActive = false
            }
        }
    }

    private func forwardTouch(at point: CGPoint) {
        let payload = try? JSONEncoder().encode([
            "x": point.x,
            "y": point.y
        ])
        RemoteViewBridge.sendAgentEvent("touch", data: payload)
    }
}

// MARK: - Preview

#Preview {
    RemoteView()
        .frame(width: 375, height: 812)
        .preferredColorScheme(.dark)
}
