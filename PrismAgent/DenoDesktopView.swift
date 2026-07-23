import SwiftUI
import WebKit

/// The native shell hosts the Deno Desktop dashboard. SwiftUI owns only the
/// macOS lifecycle and native integration points; product interaction lives in
/// the WebUI and talks to the local daemon over HTTP/WebSocket.
struct DenoDesktopView: NSViewRepresentable {
    let url: URL

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        let content = WKUserContentController()
        content.add(context.coordinator, name: "prismNative")
        content.addUserScript(WKUserScript(source: """
        window.prismNative = {
          close: () => window.webkit.messageHandlers.prismNative.postMessage({action:'close'}),
          openNative: () => window.webkit.messageHandlers.prismNative.postMessage({action:'openNative'})
        };
        """, injectionTime: .atDocumentStart, forMainFrameOnly: true))
        configuration.userContentController = content
        configuration.websiteDataStore = .default()
        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.allowsBackForwardNavigationGestures = true
        webView.navigationDelegate = context.coordinator
        webView.load(URLRequest(url: url))
        return webView
    }

    func updateNSView(_ webView: WKWebView, context: Context) {}

    @MainActor
    final class Coordinator: NSObject, WKScriptMessageHandler, WKNavigationDelegate {
        func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
            guard let body = message.body as? [String: Any], let action = body["action"] as? String else { return }
            if action == "close" { NSApp.keyWindow?.performClose(nil) }
            if action == "openNative" { NSApp.sendAction(#selector(AppDelegate.showSettings(_:)), to: NSApp.delegate, from: nil) }
        }

        func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction, decisionHandler: @escaping @MainActor @Sendable (WKNavigationActionPolicy) -> Void) {
            guard let scheme = navigationAction.request.url?.scheme?.lowercased(), ["http", "https", "file"].contains(scheme) else { decisionHandler(.cancel); return }
            decisionHandler(.allow)
        }
    }
}

struct PrismWorkspaceView: View {
    private let dashboardURL = URL(string: ProcessInfo.processInfo.environment["PRISM_WEBUI_URL"] ?? "http://127.0.0.1:8081")!

    var body: some View {
        DenoDesktopView(url: dashboardURL)
        .frame(minWidth: 760, minHeight: 620)
    }
}

final class DenoDesktopProcess {
    private var process: Process?

    func startIfAvailable() {
        guard ProcessInfo.processInfo.environment["PRISM_DENO_DESKTOP_EXTERNAL"] != "1" else { return }
        let root = dashboardRoot()
        let script = root.appendingPathComponent("main.ts")
        guard FileManager.default.fileExists(atPath: script.path), let deno = findDeno() else { return }
        let child = Process(); child.executableURL = deno; child.arguments = [
            "run", "--allow-read", "--allow-net", "--allow-env",
            "--allow-sys", "--allow-run", "--allow-write", script.path
        ]
        var environment = ProcessInfo.processInfo.environment
        environment["PRISM_DAEMON_HTTP"] = environment["PRISM_DAEMON_HTTP"] ?? "http://127.0.0.1:8080"
        environment["PRISM_DENO_PORT"] = environment["PRISM_DENO_PORT"] ?? "8081"
        child.environment = environment
        child.currentDirectoryURL = root
        try? child.run(); process = child
    }

    func stop() { process?.terminate(); process = nil }

    private func findDeno() -> URL? {
        let candidates = [
            ProcessInfo.processInfo.environment["PRISM_DENO_PATH"],
            "/opt/homebrew/bin/deno",
            "/usr/local/bin/deno",
            "/usr/bin/deno"
        ].compactMap { $0 }
        return candidates.lazy.map(URL.init(fileURLWithPath:)).first { FileManager.default.isExecutableFile(atPath: $0.path) }
    }

    private func dashboardRoot() -> URL {
        if let configured = ProcessInfo.processInfo.environment["PRISM_DENO_DESKTOP_DIR"] {
            return URL(fileURLWithPath: configured).standardizedFileURL
        }

        let fileManager = FileManager.default
        let candidates = [
            fileManager.currentDirectoryPath,
            Bundle.main.resourceURL?.path,
            Bundle.main.bundleURL.deletingLastPathComponent().path
        ].compactMap { $0 }.map { URL(fileURLWithPath: $0).appendingPathComponent("deno-dashboard") }

        return candidates.first { fileManager.fileExists(atPath: $0.appendingPathComponent("main.ts").path) }
            ?? URL(fileURLWithPath: "deno-dashboard").standardizedFileURL
    }
}
