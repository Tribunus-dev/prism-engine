//
//  AgentP2PBridge.swift
//  PrismAgent
//
//  Created by PrismAgent on 7/16/26.
//

import Foundation
import CryptoKit

/// Maintains a P2P encrypted channel to a paired iOS device and routes
/// incoming messages to the local agent session.
final class AgentP2PBridge {
    static let shared = AgentP2PBridge()

    /// The local agent session that processes transcribed text etc.
    weak var agentSession: AgentSession?

    private let key = SymmetricKey(size: .bits256)

    // MARK: - Message Handling

    /// Inbound P2P message from the remote device.
    struct P2PMessage {
        let action: String
        let payload: Data
    }

    /// Route an incoming encrypted message to the appropriate handler.
    func handleMessage(_ message: P2PMessage) {
        switch message.action {
        case "audio_chunk":
            // Audio PCM data from iOS – decrypt, transcribe, inject into session
            guard let audioData = try? decrypt(message.payload, using: key) else {
                return
            }
            Task {
                let text = await ASRService.shared.transcribe(audioData)
                await agentSession?.injectAmbient(event: "Remote user said: \(text)")
            }

        case "ping":
            // Keep-alive – respond with pong
            Task { await sendPong() }

        default:
            break
        }
    }

    // MARK: - Encryption

    private func decrypt(_ data: Data, using key: SymmetricKey) throws -> Data {
        let box = try AES.GCM.SealedBox(combined: data)
        return try AES.GCM.open(box, using: key)
    }

    // MARK: - Outbound

    func send(_ text: String) async {
        guard let encrypted = try? AES.GCM.seal(Data(text.utf8), using: key).combined else {
            return
        }
        // Transmit `encrypted` over the P2P channel (platform-specific)
        await transmit(encrypted)
    }

    private func sendPong() async {
        guard let encrypted = try? AES.GCM.seal(Data("pong".utf8), using: key).combined else {
            return
        }
        await transmit(encrypted)
    }

    private func transmit(_ data: Data) async {
        // Platform-specific transport (MultipeerConnectivity / NWConnection)
    }
}

/// Protocol that a local agent session conforms to, allowing the bridge to
/// inject ambient events.
@MainActor
protocol AgentSession: AnyObject {
    /// Injects an ambient event string into the agent's conversational context.
    func injectAmbient(event: String) async
}
