import Foundation
import SwiftUI

@MainActor
public class ModelDownloader: NSObject, ObservableObject, URLSessionDownloadDelegate {
    public static let shared = ModelDownloader()

    @Published public var progress: Double = 0.0
    @Published public var status: String = "Idle"
    @Published public var downloadBytes: Int64 = 0
    @Published public var totalBytes: Int64 = 0
    @Published public var isReady: Bool = false

    private var session: URLSession!
    private var token: String = ""
    private var destinations: [Int: URL] = [:]

    public override init() {
        super.init()
        let config = URLSessionConfiguration.background(withIdentifier: "dev.tribunus.prism.hf-downloader")
        config.isDiscretionary = false
        self.session = URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }

    public func setToken(_ t: String) { token = t }

    public func downloadModel(repo: String, filename: String, to destination: URL) {
        let url = URL(string: "https://huggingface.co/\(repo)/resolve/main/\(filename)")!
        var request = URLRequest(url: url)
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")

        self.status = "Downloading \(filename)..."
        let task = session.downloadTask(with: request)
        destinations[task.taskIdentifier] = destination
        task.resume()
    }

    nonisolated public func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        Task { @MainActor in
            downloadBytes = totalBytesWritten
            totalBytes = totalBytesExpectedToWrite
            progress = totalBytesExpectedToWrite > 0 ? Double(totalBytesWritten) / Double(totalBytesExpectedToWrite) : 0
        }
        }

    nonisolated public func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        Task { @MainActor in
            guard let destination = destinations.removeValue(forKey: downloadTask.taskIdentifier) else {
                status = "Download complete, but destination was not registered"
                return
            }
            do {
                try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
                try? FileManager.default.removeItem(at: destination)
                try FileManager.default.moveItem(at: location, to: destination)
                status = "Download complete"
            } catch {
                status = "Failed to store download: \(error.localizedDescription)"
            }
        }
    }

    nonisolated public func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error = error {
            Task { @MainActor in
                destinations.removeValue(forKey: task.taskIdentifier)
                status = "Failed: \(error.localizedDescription)"
            }
        }
        }
}
