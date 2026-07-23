import Foundation

class PrismXPCListener: NSObject, NSXPCListenerDelegate, PrismEngineXPCProtocol {
    private var listener: NSXPCListener

    override init() {
        self.listener = NSXPCListener(machServiceName: "dev.tribunus.prism.xpc")
        super.init()
        self.listener.delegate = self
        self.listener.resume()
    }

    func listener(_ listener: NSXPCListener, shouldAcceptNewConnection newConnection: NSXPCConnection) -> Bool {
        newConnection.exportedInterface = NSXPCInterface(with: PrismEngineXPCProtocol.self)
        newConnection.exportedObject = self
        newConnection.resume()
        return true
    }

    func generateCompletion(for codeContext: String, withReply reply: @escaping (String?, (any Error)?) -> Void) {
        PrismDaemonClient.shared.generate(prompt: "Complete this code:\n\(codeContext)") { result in
            switch result {
            case .success(let text): reply(text, nil)
            case .failure(let error): reply(nil, error)
            }
        }
    }
}
