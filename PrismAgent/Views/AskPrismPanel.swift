import SwiftUI
import AppKit

/// Floating panel that sends queries to the Prism agent and types
/// responses into the frontmost app via Accessibility (AXUIElement).
struct AskPrismPanel: View {
    @State private var queryText = ""
    @State private var response: String?
    @State private var isThinking = false
    @State private var statusText = ""

    var onClose: () -> Void

    var body: some View {
        VStack(spacing: 12) {
            // Header
            HStack {
                Text("Ask Prism")
                    .font(.system(.headline, design: .rounded))
                    .foregroundColor(.primary)
                Spacer()
                Button(action: onClose) {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundColor(.secondary)
                }
                .buttonStyle(.plain)
            }

            // Query input
            HStack {
                TextField("Ask anything...", text: $queryText)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 13))
                    .disabled(isThinking)
                    .onSubmit(submitQuery)

                Button(action: submitQuery) {
                    if isThinking {
                        ProgressView()
                            .scaleEffect(0.6)
                            .frame(width: 16, height: 16)
                    } else {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.system(size: 20))
                    }
                }
                .buttonStyle(.plain)
                .disabled(queryText.trimmingCharacters(in: .whitespaces).isEmpty || isThinking)
            }

            // Response area
            if let response {
                VStack(alignment: .leading, spacing: 8) {
                    Divider()
                    Text(response)
                        .font(.system(size: 12))
                        .foregroundColor(.secondary)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)

                    Button { typeResponseIntoFrontmostApp(response) } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "keyboard")
                                .font(.system(size: 10))
                            Text("Type into frontmost app")
                                .font(.system(size: 12, weight: .medium))
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(Capsule().fill(Color.accentColor.opacity(0.12)))
                        .overlay(Capsule().stroke(Color.accentColor.opacity(0.3), lineWidth: 0.5))
                    }
                    .buttonStyle(.plain)
                }
            }

            // Status
            if !statusText.isEmpty {
                Text(statusText)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .padding(16)
        .frame(width: 360)
        .background(PrismTheme.prismGlass(cornerRadius: 14))
    }

    private func submitQuery() {
        let query = queryText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return }
        queryText = ""
        response = nil
        isThinking = true
        statusText = "Thinking..."

        Task {
            do {
                // Use existing inference pipeline
                let answer = try await AskPrismService.shared.processQuery(query)
                response = answer
                isThinking = false
                statusText = ""
            } catch {
                response = "Error: \(error.localizedDescription)"
                isThinking = false
                statusText = "Failed"
            }
        }
    }

    private func typeResponseIntoFrontmostApp(_ text: String) {
        AccessibilityTypist.typeText(text)
        statusText = "Typing..."
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            statusText = ""
        }
    }
}

// MARK: - Ask Prism Service

@MainActor
final class AskPrismService {
    static let shared = AskPrismService()
    private init() {}

    func processQuery(_ query: String) async throws -> String {
        try await PrismDaemonClient.shared.generate(prompt: query)
    }
}

// MARK: - Accessibility Typist

enum AccessibilityTypist {
    /// Types the given text into the frontmost application using Accessibility API.
    static func typeText(_ text: String) {
        guard let app = NSWorkspace.shared.frontmostApplication else { return }

        Task { @MainActor in
            AgentActionOverlay.shared.showTyping(text)
        }

        let pid = app.processIdentifier
        let appRef = AXUIElementCreateApplication(pid)

        // Find the focused element
        var focused: CFTypeRef?
        AXUIElementCopyAttributeValue(appRef, kAXFocusedUIElementAttribute as CFString, &focused)
        guard let focused = focused else { return }
        // swiftlint:disable:next force_cast
        let focusedElement = focused as! AXUIElement

        // Check if it supports setting value (text fields, text views)
        var role: CFTypeRef?
        AXUIElementCopyAttributeValue(focusedElement, kAXRoleAttribute as CFString, &role)

        // Try to set the selected text range and insert
        // First check if we can get the selected text range
        var selectedRange: CFTypeRef?
        AXUIElementCopyAttributeValue(focusedElement, kAXSelectedTextRangeAttribute as CFString, &selectedRange)

        if let rangeValue = selectedRange {
            // Replace selected text or append at cursor position
            AXUIElementSetAttributeValue(focusedElement, kAXSelectedTextRangeAttribute as CFString, rangeValue)
            AXUIElementSetAttributeValue(focusedElement, kAXValueAttribute as CFString, text as CFTypeRef)
        }

        // Fallback: use CGEvent keyboard events
        // This is the more reliable path for most apps
        typeViaCGEvent(text)
    }

    private static func typeViaCGEvent(_ text: String) {
        for char in text {
            guard let cgChar = char.unicodeScalars.first else { continue }
            let keyCode = CGKeyCode(cgChar.value)

            guard let source = CGEventSource(stateID: .combinedSessionState) else { continue }

            let eventDown = CGEvent(keyboardEventSource: source, virtualKey: keyCode, keyDown: true)
            let eventUp = CGEvent(keyboardEventSource: source, virtualKey: keyCode, keyDown: false)

            eventDown?.post(tap: .cghidEventTap)
            eventUp?.post(tap: .cghidEventTap)

            // Small delay between characters for reliability
            usleep(10_000) // 10ms
        }
    }
}
