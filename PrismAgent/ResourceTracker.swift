import Foundation
import Observation

#if canImport(IOKit)
import IOKit
import IOKit.ps
#endif

/// Monitors system resource usage for the progressive contribution scheduler.
/// Polls at configurable intervals. All measurements are approximate and
/// used only for local policy decisions.
@MainActor
@Observable
final class ResourceTracker {
    static let shared = ResourceTracker()

    /// Current readings
    private(set) var cpuPercent: Double = 0
    private(set) var thermalState: ProcessInfo.ThermalState = .nominal
    private(set) var batteryLevel: Float = 1.0
    private(set) var isOnBattery: Bool = false
    private(set) var temperatureCelsius: Float = 0
    private(set) var dailyEnergyWh: Float = 0
    private(set) var dailyNetworkBytes: UInt64 = 0
    /// Best-effort diagnostic readings. Apple does not guarantee these counters.
    private(set) var aneUtilization: Double?
    private(set) var slcUtilization: Double?

    // Accumulators
    private var pollingTask: Task<Void, Never>?
    private var lastCpuTicks: host_cpu_load_info?
    private var lastPollTime: Date?
    private var lastBatteryCapacity: Float?
    private var powermetricsTask: Task<(ane: Double?, slc: Double?), Never>?

    private init() {}

    func start(pollInterval: TimeInterval = 5.0) {
        pollingTask?.cancel()
        pollingTask = Task { [weak self] in
            while !Task.isCancelled {
                await self?.poll()
                try? await Task.sleep(nanoseconds: UInt64(pollInterval * 1_000_000_000))
            }
        }
    }

    func stop() {
        pollingTask?.cancel()
        pollingTask = nil
        powermetricsTask?.cancel()
        powermetricsTask = nil
    }

    func recordNetworkDownload(bytes: UInt64) {
        dailyNetworkBytes += bytes
    }

    func recordNetworkUpload(bytes: UInt64) {
        dailyNetworkBytes += bytes
    }

    func resetDailyAccounting() {
        dailyEnergyWh = 0
        dailyNetworkBytes = 0
    }

    // MARK: - Poll

    private func poll() async {
        let now = Date()
        await readThermalState()
        await readPowerState(pollTime: now)
        await readCpuUsage()
        await readBestEffortDiagnostics()
        lastPollTime = now
    }

    private func readBestEffortDiagnostics() async {
        if let task = powermetricsTask, !task.isCancelled {
            let result = await task.value
            aneUtilization = result.ane
            slcUtilization = result.slc
            powermetricsTask = nil
            return
        }

        powermetricsTask = Task.detached(priority: .utility) {
            guard FileManager.default.isExecutableFile(atPath: "/usr/bin/powermetrics") else {
                return (ane: nil, slc: nil)
            }

            let process = Process()
            let output = Pipe()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/powermetrics")
            process.arguments = ["-n", "1", "-i", "1000", "--show-process-energy"]
            process.standardOutput = output
            process.standardError = output

            do {
                try process.run()
                process.waitUntilExit()
                let data = output.fileHandleForReading.readDataToEndOfFile()
                let text = String(decoding: data, as: UTF8.self)
                return (ane: Self.parsePercentage(in: text, labels: ["ane", "neural engine"]),
                        slc: Self.parsePercentage(in: text, labels: ["slc", "system level cache"]))
            } catch {
                return (ane: nil, slc: nil)
            }
        }
    }

    nonisolated private static func parsePercentage(in text: String, labels: [String]) -> Double? {
        let lines = text.split(whereSeparator: \.isNewline)
        for line in lines {
            let lower = line.lowercased()
            guard labels.contains(where: { lower.contains($0) }) else { continue }
            let values = lower.split(whereSeparator: { !$0.isNumber && $0 != "." })
                .compactMap { Double($0) }
            if let value = values.first(where: { $0 >= 0 && $0 <= 100 }) {
                return value / 100
            }
        }
        return nil
    }

    private func readThermalState() async {
        thermalState = ProcessInfo.processInfo.thermalState
        // Map thermal state to approximate Celsius surface temperature
        switch thermalState {
        case .nominal: temperatureCelsius = 35
        case .fair: temperatureCelsius = 50
        case .serious: temperatureCelsius = 70
        case .critical: temperatureCelsius = 90
        @unknown default: temperatureCelsius = 40
        }
    }

    private func readPowerState(pollTime: Date) async {
        #if canImport(IOKit)
        guard
            let powerSourcesInfo = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
            let powerSources = IOPSCopyPowerSourcesList(powerSourcesInfo)?
                .takeRetainedValue() as? [CFTypeRef],
            let firstSource = powerSources.first,
            let desc = IOPSGetPowerSourceDescription(powerSourcesInfo, firstSource)?
                .takeUnretainedValue() as? [String: Any]
        else {
            isOnBattery = false
            batteryLevel = 1.0
            return
        }

        isOnBattery = desc["Power Source State"] as? String == "Battery Power"
        let currentCapacity = desc["Current Capacity"] as? Float ?? 100
        let maxCapacity = desc["Max Capacity"] as? Float ?? 100
        batteryLevel = maxCapacity > 0 ? currentCapacity / maxCapacity : 1.0

        // Track energy discharge (either/or to avoid double-counting)
        if isOnBattery {
            // Prefer amperage-based estimation (more accurate)
            if let amperage = desc["Amperage"] as? Float, amperage < 0,
               let voltage = desc["Voltage"] as? Float,
               let lastPoll = lastPollTime {
                let intervalHours = Float(pollTime.timeIntervalSince(lastPoll)) / 3600.0
                // Power = |amperage| × voltage (mW), energy = power × time (mWh)
                let powerMw = -amperage * voltage
                let deltaWh = powerMw / 1000.0 * intervalHours
                if deltaWh > 0 { dailyEnergyWh += deltaWh }
            } else if let lastCap = lastBatteryCapacity {
                // Fallback: capacity delta (mAh difference × voltage)
                let deltaCap = lastCap - currentCapacity
                let voltage = desc["Voltage"] as? Float ?? 11.4
                let deltaWh = (deltaCap / 1000.0) * voltage
                if deltaWh > 0 { dailyEnergyWh += deltaWh }
            }
        }
        lastBatteryCapacity = currentCapacity
        #else
        // Non-macOS: assume always connected to AC
        isOnBattery = false
        batteryLevel = 1.0
        #endif
    }

    private func readCpuUsage() async {
        var cpuInfo = host_cpu_load_info()
        var count = mach_msg_type_number_t(
            MemoryLayout<host_cpu_load_info>.size / MemoryLayout<integer_t>.size
        )
        let result = withUnsafeMutablePointer(to: &cpuInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return }

        let totalTicks = cpuInfo.cpu_ticks.0 + cpuInfo.cpu_ticks.1
            + cpuInfo.cpu_ticks.2 + cpuInfo.cpu_ticks.3

        guard let prev = lastCpuTicks else {
            // First read — store baseline, report 0 until next poll
            lastCpuTicks = cpuInfo
            cpuPercent = 0
            return
        }

        let prevTotal = prev.cpu_ticks.0 + prev.cpu_ticks.1
            + prev.cpu_ticks.2 + prev.cpu_ticks.3
        let prevBusy = prev.cpu_ticks.0 + prev.cpu_ticks.1 + prev.cpu_ticks.3

        let deltaTotal = totalTicks - prevTotal
        let deltaBusy = (cpuInfo.cpu_ticks.0 + cpuInfo.cpu_ticks.1 + cpuInfo.cpu_ticks.3) - prevBusy

        if deltaTotal > 0 {
            cpuPercent = (Double(deltaBusy) / Double(deltaTotal)) * 100.0
        }

        lastCpuTicks = cpuInfo
    }
}
