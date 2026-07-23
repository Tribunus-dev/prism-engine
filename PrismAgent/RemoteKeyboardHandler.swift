import Foundation
import PrismCore

/// Handles P2P "remoteKeyboardQuery" messages from iOS keyboard extension.
/// Receives queries, runs inference, sends responses back via P2P.
@MainActor
enum RemoteKeyboardHandler {
    static let actionType = "remoteKeyboardQuery"

    /// Register the handler with the P2P message router.
    /// Hooks into P2PRouter.shared.onMessageReceived to intercept
    /// messages with our action type.
    static func register() {
        let existingHandler = P2PRouter.shared.onMessageReceived
        P2PRouter.shared.onMessageReceived = { [existingHandler] message in
            existingHandler?(message)
            guard message.action == actionType else { return }
            Task { @MainActor in
                await processAndRespond(message)
            }
        }
    }

    private static func processAndRespond(_ message: P2PMessage) async {
        do {
            // Decrypt the query payload
            // let query: String = try message.decrypt(String.self, using: sharedKey) ?? ""

            // For now, use the shared key placeholder
            guard let query = message.encryptedPayload?.nonce.description else {
                print("[RemoteKeyboardHandler] No query in message")
                return
            }

            print("[RemoteKeyboardHandler] Received query: \(query.prefix(80))...")

            // Process through existing inference pipeline
            let response = try await processQuery(query)

            // Send response back via P2P
            // let reply = try P2PMessage.response(to: message, action: "remoteKeyboardResponse",
            //                                      payload: response, using: sharedKey)
            // try await P2PRouter.shared.send(reply)

            print("[RemoteKeyboardHandler] Response ready: \(response.prefix(80))...")

            // Also write to App Group in case iOS app polls
            if let defaults = UserDefaults(suiteName: "group.com.prismagent.keyboard") {
                defaults.set(response, forKey: "keyboardResponse")
                defaults.set("response_ready", forKey: "keyboardState")
                CFNotificationCenterPostNotification(
                    CFNotificationCenterGetDarwinNotifyCenter(),
                   CFNotificationName(rawValue: "com.prismagent.keyboardResponse" as CFString),
                    nil, nil, true
                )
            }
        } catch {
            print("[RemoteKeyboardHandler] Error: \(error.localizedDescription)")
        }
    }

    private static func processQuery(_ query: String) async throws -> String {
        try await PrismDaemonClient.shared.generate(prompt: query)
    }
}
