import Cocoa
import Carbon
import SwiftUI
import PrismCore
import CloudKit
import PrismAgentSDK
import AVFoundation

@MainActor
final class ServicesProvider: NSObject {
    @objc func askPrism(_ pboard: NSPasteboard, userData: String, error: AutoreleasingUnsafeMutablePointer<NSString?>) {
        guard let text = pboard.string(forType: .string) else { return }
        // Injects selected text into current conversation
        Task { @MainActor in
            let store = ConversationStore.shared
            try? await store.append(role: .user, text: "\(text)", image: nil)
        }
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var instanceLock: InstanceLock!
    private var launchAtLoginManager: LaunchAtLoginManager!
    private var statusItem: NSStatusItem!
    private var contextMenu: NSMenu!
    private var respServer: EmbeddedRESPServer?
    var popover: NSPopover!
    var overlayPanel: PrismOverlayPanel!
    private var hotkeyManager: GlobalHotkeyManager!
    private let denoDesktop = DenoDesktopProcess()

    @available(macOS 15, *)
    lazy var cloudRelay = CloudKitRelay()

    @Published var isProcessing: Bool = false
    private var iconAnimationTimer: Timer?
    
    // MARK: - Application Lifecycle

    func applicationWillFinishLaunching(_ notification: Notification) {
        NSAppleEventManager.shared().setEventHandler(
            self,
            andSelector: #selector(handleAppleEvent(_:withReplyEvent:)),
            forEventClass: AEEventClass(kASAppleScriptSuite),
            andEventID: AEEventID(kASSubroutineEvent)
        )
    }

    @objc func handleAppleEvent(_ event: NSAppleEventDescriptor, withReplyEvent reply: NSAppleEventDescriptor) {
        guard let command = event.forKeyword(AEKeyword(keyASSubroutineName))?.stringValue else { return }
        Task { @MainActor in
            let store = ConversationStore.shared
            try? await store.append(role: .user, text: "(AppleScript) \(command)", image: nil)
        }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        denoDesktop.startIfAvailable()
        // 1. Single-instance lock (exits if another instance is running).
        instanceLock = InstanceLock()

        // 2. Launch-at-login state.
        launchAtLoginManager = LaunchAtLoginManager()

        // 3. Start embedded RESP server (replaces external valkey-server daemon).
        Task {
            if #available(macOS 15, *) {
                do {
                    let server = EmbeddedRESPServer(config: .default)
                    try await server.start()
                    self.respServer = server
                    let port = await server.assignedPort
                    print("[RESPServer] Started on port \(port)")
                } catch {
                    print("[RESPServer] Failed to start: \(error.localizedDescription)")
                }
            }
        }

        // 3. Act as a background agent (no dock icon, no menu bar).
        NSApp.setActivationPolicy(.accessory)

        // 4. Create the menu-bar status item.
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        statusItem.button?.image = NSImage(named: "MenuBarIcon")

        // 5. Create the popover that holds the control panel.
        popover = NSPopover()
        popover.behavior = .transient
        popover.contentViewController = NSHostingController(
            rootView: OrbPopover(
                session: AgentSession(),
                onSettingsTap: { [weak self] in
                    self?.popover.performClose(nil)
                    (NSApp.delegate as? AppDelegate)?.showSettings(nil)
                }
            )
        )

        // 6b. Register as services provider for system-wide Services menu.
#if !APP_STORE
        NSApp.servicesProvider = ServicesProvider()
        NSUpdateDynamicServices()
#endif

        // 6. Build the right-click context menu.
        buildMenu()

        // 7. Wire left-click to toggle popover, right-click to show menu.
        statusItem.button?.action = #selector(togglePopover)
        statusItem.button?.target = self
        statusItem.button?.sendAction(on: [.leftMouseUp, .rightMouseUp])

        // 8. Start icon animation timer.
        iconAnimationTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.updateMenuBarIcon()
            }
        }

        // 9. Wire CloudKit peer-discovery callback.
        if #available(macOS 15, *) {
            cloudRelay.onPeerDiscovered = { peer in
                Task { @MainActor in
                    print("[CloudKitRelay] Discovered peer: \(peer.displayName)")
                }
            }
        }
// 10. Register browser tools for agent discovery.
#if !APP_STORE
registerBrowserTools()
#endif
        // 12. Create the overlay panel (collapsed pill, floats above all).
        overlayPanel = PrismOverlayPanel()
        overlayPanel.showOverlay(expand: false)

        // 13. Create and start the global hotkey manager.
        hotkeyManager = GlobalHotkeyManager { [weak self] in
            self?.overlayPanel?.toggleOverlay()
        }
        if !hotkeyManager.start() {
            print("[GlobalHotkeyManager] Accessibility permission not granted; entering retry loop.")
            hotkeyManager.startRetryLoop()
        }

        // 11. Start LMDB memory store + CloudKit sync + KV cache.
        Task {
            await PrismMemoryInitializer.shared.start()
        }

        // 14. Request microphone permission on launch (non-blocking).
        Task {
            await requestMicrophonePermissionIfNeeded()
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        denoDesktop.stop()
        instanceLock?.terminate()
        iconAnimationTimer?.invalidate()
        Task {
            await PrismMemoryInitializer.shared.stop()
        }
    }

    // MARK: - Menu Bar Icon Animation

    func setProcessing(_ processing: Bool) {
        isProcessing = processing
        updateMenuBarIcon()
    }

    private func registerBrowserTools() {
        ToolRegistry.shared.register(NavigateInAppTool.self)
        ToolRegistry.shared.register(ExtractPageContentTool.self)
        ToolRegistry.shared.register(ExecuteJSTool.self)
        ToolRegistry.shared.register(SafariNavigateTool.self)
        ToolRegistry.shared.register(SafariExtractTool.self)
        ToolRegistry.shared.register(SafariExecuteJSTool.self)
        ToolRegistry.shared.register(SafariScreenshotTool.self)
        ToolRegistry.shared.register(SafariStructuredExtractTool.self)
        ToolRegistry.shared.register(SafariStructuredViewTool.self)
        ToolRegistry.shared.register(SafariClickRegionTool.self)
        ToolRegistry.shared.register(SafariTypeAtTool.self)
    }

    private func updateMenuBarIcon() {
        guard let button = statusItem.button else { return }
        if isProcessing {
            // Subtle pulse by toggling alpha
            let alpha: CGFloat = (Date().timeIntervalSince1970 * 2).truncatingRemainder(dividingBy: 1) > 0.5 ? 1.0 : 0.7
            button.alphaValue = alpha
        } else {
            button.alphaValue = 1.0
        }
    }

    // MARK: - Popover

    @objc func togglePopover(_ sender: Any?) {
        guard let event = NSApp.currentEvent else { return }

        switch event.type {
        case .rightMouseUp, .otherMouseUp:
            showContextMenu()
        default:
            // Left-click → toggle popover.
            if popover.isShown {
                popover.performClose(sender)
            } else {
                showPopover()
            }
        }
    }

    private func showPopover() {
        guard let button = statusItem.button else { return }
        popover.show(
            relativeTo: button.bounds,
            of: button,
            preferredEdge: .minY
        )
    }

    // MARK: - Context Menu

    private func buildMenu() {
        contextMenu = NSMenu()

        let settingsItem = NSMenuItem(
            title: "Settings\u{2026}",
            action: #selector(showSettings),
            keyEquivalent: ","
        )
        settingsItem.target = self
        contextMenu.addItem(settingsItem)

        contextMenu.addItem(.separator())

        let loginItem = NSMenuItem(
            title: "Launch at Login",
            action: #selector(toggleLaunchAtLogin),
            keyEquivalent: ""
        )
        loginItem.target = self
        loginItem.state = launchAtLoginManager.isEnabled ? .on : .off
        contextMenu.addItem(loginItem)

        contextMenu.addItem(.separator())

        let quitItem = NSMenuItem(
            title: "Quit",
            action: #selector(NSApplication.terminate(_:)),
            keyEquivalent: "q"
        )
        quitItem.target = NSApp
        contextMenu.addItem(quitItem)
    }

    private func showContextMenu() {
        guard let button = statusItem.button else { return }
        contextMenu.popUp(positioning: nil, at: .zero, in: button)
    }

    @objc func showSettings(_ sender: Any?) {
        NSApp.activate(ignoringOtherApps: true)
        if let window = NSApp.windows.first(where: { $0.contentViewController is NSHostingController<PrismWorkspaceView> }) {
            window.makeKeyAndOrderFront(sender)
            return
        }
        if let window = NSApp.windows.first(where: { $0.canBecomeKey }) {
            window.makeKeyAndOrderFront(sender)
        }
    }

    @objc func toggleLaunchAtLogin(_ sender: NSMenuItem) {
        launchAtLoginManager.toggle()
        sender.state = launchAtLoginManager.isEnabled ? .on : .off
    }

    // MARK: - CloudKit Push Notifications
    // MARK: - Microphone Permission

    /// Request microphone permission if not already granted.
    /// On first voice use, prompts the user; if previously denied, deep-links
    /// to System Settings > Privacy & Security > Microphone.
    private func requestMicrophonePermission() async -> Bool {
        let status = AVCaptureDevice.authorizationStatus(for: .audio)
        switch status {
        case .authorized:
            return true
        case .denied, .restricted:
            // Deep-link to System Settings
            if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone") {
                NSWorkspace.shared.open(url)
            }
            return false
        case .notDetermined:
            return await AVCaptureDevice.requestAccess(for: .audio)
        @unknown default:
            return false
        }
    }

    /// Check microphone permission on first voice use; log the result.
    private func requestMicrophonePermissionIfNeeded() async {
        let granted = await requestMicrophonePermission()
        if granted {
            print("[Microphone] Permission granted")
        } else {
            print("[Microphone] Permission denied or restricted")
        }
    }

    // MARK: - CloudKit Push Notifications


    @available(macOS 15, *)
    func application(_ application: NSApplication, didReceiveRemoteNotification userInfo: [String: Any]) {
        _ = CKNotification(fromRemoteNotificationDictionary: userInfo)
        Task { [weak self] in
            try? await self?.cloudRelay.refresh()
        }
    }
}
