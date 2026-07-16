import Foundation

/// Discovers available agents by scanning directories for agent metadata.
public struct AgentDiscoveryEngine {
    private let fileManager: FileManager

    public init(fileManager: FileManager = .default) {
        self.fileManager = fileManager
    }

    /// Discover agents in the given directory path.
    /// Returns an array of discovered agent names.
    public func discoverAgents(in directory: String) -> [String] {
        var isDir: ObjCBool = false
        let exists = fileManager.fileExists(atPath: directory, isDirectory: &isDir)
        guard exists, isDir.boolValue else {
            return []
        }

        guard let contents = try? fileManager.contentsOfDirectory(atPath: directory) else {
            return []
        }

        return contents.filter { item in
            let fullPath = (directory as NSString).appendingPathComponent(item)
            var itemIsDir: ObjCBool = false
            let itemExists = fileManager.fileExists(atPath: fullPath, isDirectory: &itemIsDir)
            return itemExists && !itemIsDir.boolValue && item.hasSuffix(".agent.json")
        }.map { $0.replacingOccurrences(of: ".agent.json", with: "") }
    }

    /// Check whether a specific agent configuration file exists.
    public func agentConfigExists(named name: String, in directory: String) -> Bool {
        let path = (directory as NSString).appendingPathComponent("\(name).agent.json")
        return fileManager.fileExists(atPath: path)
    }

    /// List subdirectories of the given path (each is a potential agent bundle).
    public func discoverAgentBundles(in directory: String) -> [String] {
        var isDir: ObjCBool = false
        let exists = fileManager.fileExists(atPath: directory, isDirectory: &isDir)
        guard exists, isDir.boolValue else {
            return []
        }

        guard let contents = try? fileManager.contentsOfDirectory(atPath: directory) else {
            return []
        }

        return contents.filter { item in
            let fullPath = (directory as NSString).appendingPathComponent(item)
            var itemIsDir: ObjCBool = false
            let itemExists = fileManager.fileExists(atPath: fullPath, isDirectory: &itemIsDir)
            return itemExists && itemIsDir.boolValue
        }
    }
}
