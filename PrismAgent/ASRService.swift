//
//  ASRService.swift
//  PrismAgent
//
//  Created by PrismAgent on 7/16/26.
//

import Speech
import AVFAudio

/// macOS speech-to-text service using `SFSpeechRecognizer`.
/// Transcribes 16-bit 16 kHz mono PCM audio data into text.
@available(macOS 15, *)
@MainActor
final class ASRService {
    static let shared = ASRService()

    private let recognizer: SFSpeechRecognizer?

    private init() {
        self.recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    }

    /// Transcribe 16 kHz 16-bit mono PCM audio to text.
    /// - Parameter pcmData: Raw PCM samples (no headers).
    /// - Returns: The best transcription, or empty string on failure.
    func transcribe(_ pcmData: Data) async -> String {
        let url = writeWaveFile(pcmData)

        let request = SFSpeechURLRecognitionRequest(url: url)

        return await withCheckedContinuation { continuation in
            guard let recognizer, recognizer.isAvailable else {
                continuation.resume(returning: "")
                return
            }

            recognizer.recognitionTask(with: request) { result, error in
                if let result {
                    continuation.resume(returning: result.bestTranscription.formattedString)
                } else {
                    continuation.resume(returning: "")
                }
            }
        }
    }

    /// Write a proper WAV file wrapping the raw PCM data.
    private func writeWaveFile(_ pcmData: Data) -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("audio_" + UUID().uuidString + ".wav")

        let sampleRate: UInt32 = 16000
        let channels: UInt16 = 1
        let bitsPerSample: UInt16 = 16
        let bytesPerFrame: UInt16 = channels * (bitsPerSample / 8)
        let byteRate: UInt32 = sampleRate * UInt32(bytesPerFrame)
        let dataSize = UInt32(pcmData.count)
        let fileSize = dataSize + 36

        var header = Data()
        // RIFF chunk
        header.append("RIFF".data(using: .utf8)!)
        header.append(Data(value: fileSize))
        header.append("WAVE".data(using: .utf8)!)

        // fmt sub-chunk
        header.append("fmt ".data(using: .utf8)!)
        header.append(Data(value: UInt32(16)))         // sub-chunk size
        header.append(Data(value: UInt16(1)))            // PCM format
        header.append(Data(value: channels))
        header.append(Data(value: sampleRate))
        header.append(Data(value: byteRate))
        header.append(Data(value: bytesPerFrame))
        header.append(Data(value: bitsPerSample))

        // data sub-chunk
        header.append("data".data(using: .utf8)!)
        header.append(Data(value: dataSize))

        try? (header + pcmData).write(to: url)
        return url
    }
}

// MARK: - Little-endian binary helpers

private extension Data {
    /// Append a value as little-endian raw bytes.
    init<T: FixedWidthInteger>(value: T) {
        var v = value.littleEndian
        self.init(bytes: &v, count: MemoryLayout<T>.size)
    }
}
