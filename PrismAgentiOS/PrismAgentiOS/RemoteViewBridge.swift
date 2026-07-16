import Foundation
import Network

/// Bridges P2P messages between iOS RemoteView and the Mac agent via NWConnection.
@MainActor
public final class RemoteViewBridge {
    public static let shared = RemoteViewBridge()

    private var connection: NWConnection?
    private let receiverQueue = DispatchQueue(label: "com.prismagent.remoteviewbridge")

    // MARK: - P2P Event Callbacks

    public typealias EventHandler = (String, Data?) -> Void
    private var eventHandlers: [String: EventHandler] = [:]

    public func onEvent(_ action: String, handler: @escaping EventHandler) {
        eventHandlers[action] = handler
    }

    // MARK: - Connection

    public func start(host: String, port: UInt16) {
        let params = NWParameters.tcp
        let endpoint = NWEndpoint.hostPort(
            host: NWEndpoint.Host(host),
            port: NWEndpoint.Port(rawValue: port)!
        )
        connection = NWConnection(to: endpoint, using: params)
        connection?.stateUpdateHandler = { [weak self] state in
            switch state {
            case .ready:
                self?.receiveLoop()
            case .failed(let error):
                print("[RemoteViewBridge] Connection failed: \(error)")
            default:
                break
            }
        }
        connection?.start(queue: receiverQueue)
    }

    private func receiveLoop() {
        connection?.receiveMessage { [weak self] data, _, _, error in
            guard let self else { return }
            if let error {
                print("[RemoteViewBridge] Receive error: \(error)")
                return
            }
            if let data, let message = try? JSONDecoder().decode(P2PMessage.self, from: data) {
                Task { @MainActor in
                    self.dispatch(message)
                }
            }
            self.receiveLoop()
        }
    }

    // MARK: - Dispatch

    private func dispatch(_ message: P2PMessage) {
        if let handler = eventHandlers[message.action] {
            handler(message.action, message.payload)
        }
    }

    // MARK: - Send

    public static func sendEvent(_ action: String, data: Data? = nil) {
        let message = P2PMessage(action: action, payload: data)
        guard let encoded = try? JSONEncoder().encode(message) else { return }
        Task {
            await Self.shared._send(encoded)
        }
    }

    /// Send an agent event (reasoning, tool_call, complete) from RemoteView to the P2P bridge.
    public static func sendAgentEvent(_ event: String, data: Data? = nil) {
        sendEvent(event, data: data)
    }

    private func _send(_ data: Data) {
        connection?.send(content: data, completion: .contentProcessed { error in
            if let error {
                print("[RemoteViewBridge] Send error: \(error)")
            }
        })
    }

    // MARK: - Cleanup

    public func stop() {
        connection?.cancel()
        connection = nil
    }
}

// MARK: - P2PMessage

struct P2PMessage: Codable {
    let action: String
    let payload: Data?
}
