import SwiftUI

struct CompilerLabEvent: Identifiable {
    let id = UUID()
    let phase: String
    let detail: String
    let progress: Double
}

/// A researcher-friendly view of compilation and evolutionary search.
/// Detailed values remain unavailable until the daemon publishes structured events.
struct CompilerLabView: View {
    @ObservedObject var engineController: PrismEngineController
    @State private var selectedPhase = "Pipeline"
    @State private var searchStarting = false
    @State private var searchError: String?

    private let phases = [
        ("Ingest", "arrow.down.doc", "Model source and metadata"),
        ("Graph", "point.3.connected.trianglepath.dotted", "IR construction and lowering"),
        ("Search", "wand.and.stars", "Format and kernel exploration"),
        ("Compile", "hammer", "CImage and Metal packaging"),
        ("Verify", "checkmark.shield", "Runtime and hardware gates")
    ]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                phaseTimeline
                liveEvents
                searchPanel
                technicalPanel
            }
            .padding(20)
        }
        .frame(minWidth: 620, minHeight: 560)
        .background(.regularMaterial)
        .task {
            do {
                for try await event in PrismDaemonClient.shared.compilerLabEvents() {
                    await MainActor.run { engineController.recordDaemonEvent(event) }
                }
            } catch {
                // The lab remains useful for local compile events when the daemon is offline.
            }
        }
    }

    private var liveEvents: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack { Text("Live execution").font(.headline); Spacer(); Text("\(Int(engineController.compileProgress * 100))%") .font(.caption.monospaced()).foregroundStyle(.secondary) }
            ProgressView(value: engineController.compileProgress)
                .tint(engineController.compilePhase == "Failed" ? .red : .accentColor)
            if engineController.compileEvents.isEmpty {
                Text("No compiler events yet.").font(.caption).foregroundStyle(.secondary)
            } else {
                EventProgressChart(events: engineController.compileEvents)
                    .frame(height: 72)
                ForEach(engineController.compileEvents.reversed()) { event in
                    HStack(alignment: .top, spacing: 9) {
                        Image(systemName: event.phase == "Failed" ? "xmark.circle.fill" : "checkmark.circle.fill")
                            .foregroundStyle(event.phase == "Failed" ? .red : .accentColor)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(event.phase).font(.caption.weight(.semibold))
                            Text(event.detail).font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        Text("\(Int(event.progress * 100))%").font(.caption2.monospaced()).foregroundStyle(.tertiary)
                    }
                }
            }
        }
        .padding(14).background(.background.opacity(0.65), in: RoundedRectangle(cornerRadius: 14))
    }

    private var header: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Compiler Lab").font(.system(.title2, design: .rounded).weight(.semibold))
                Text("Follow how Prism turns a model into a measured execution artifact.")
                    .font(.callout).foregroundStyle(.secondary)
            }
            Spacer()
            Label(engineController.isCompiling ? "Compiling" : "Idle",
                  systemImage: engineController.isCompiling ? "waveform.path.ecg" : "pause.circle")
                .font(.caption.weight(.medium))
                .foregroundStyle(engineController.isCompiling ? .orange : .secondary)
                .padding(.horizontal, 10).padding(.vertical, 6)
                .background(.quaternary, in: Capsule())
        }
    }

    private var phaseTimeline: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Pipeline").font(.headline)
            HStack(spacing: 0) {
                ForEach(Array(phases.enumerated()), id: \.offset) { index, phase in
                    VStack(spacing: 7) {
                        Image(systemName: phase.1)
                            .font(.system(size: 15, weight: .semibold))
                            .frame(width: 34, height: 34)
                            .foregroundStyle(index == 0 || (engineController.isCompiling && index == 3) ? .white : .secondary)
                            .background(index == 0 || (engineController.isCompiling && index == 3) ? Color.accentColor : Color.secondary.opacity(0.12), in: Circle())
                        Text(phase.0).font(.caption.weight(.medium))
                    }
                    .frame(maxWidth: .infinity)
                    if index < phases.count - 1 { Rectangle().fill(.quaternary).frame(height: 2).padding(.bottom, 24) }
                }
            }
            Text(engineController.isCompiling ? "The compiler is running. Detailed phase events will appear as the daemon stream is connected." : "Ready to compile. Start a model build to populate live phase evidence.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(14).background(.background.opacity(0.65), in: RoundedRectangle(cornerRadius: 14))
    }

    private var searchPanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Evolutionary search").font(.headline)
                Spacer()
                Button(searchStarting ? "Starting…" : "Run search") {
                    searchStarting = true
                    searchError = nil
                    Task {
                        do { try await PrismDaemonClient.shared.startEvolutionSearch(model: engineController.activeCImagePath); searchStarting = false }
                        catch { searchError = error.localizedDescription; searchStarting = false }
                    }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .disabled(searchStarting)
            }
            HStack(spacing: 12) {
                metric("Generation", engineController.searchGenerations == 0 ? "—" : "\(engineController.searchGeneration)/\(engineController.searchGenerations)")
                metric("Population", engineController.searchPopulation == 0 ? "—" : "\(engineController.searchPopulation)")
                metric("Best fitness", engineController.searchBestFitness.map { String(format: "%.3f", $0) } ?? "—")
                metric("Candidates", engineController.searchCandidates.isEmpty ? "—" : "\(engineController.searchCandidates.count)")
            }
            if engineController.searchCandidates.isEmpty {
                ContentUnavailableView("No search run yet", systemImage: "chart.xyaxis.line", description: Text("Start a daemon-backed search to populate candidate scores and the frontier."))
                    .frame(maxWidth: .infinity)
            } else {
                VStack(spacing: 5) {
                    ForEach(Array(engineController.searchCandidates.prefix(8)), id: \.id) { candidate in
                        HStack {
                            Text("Candidate \(candidate.id)").font(.caption.monospaced())
                            Text(candidate.representation).font(.caption).foregroundStyle(.secondary)
                            Spacer()
                            Text(String(format: "%.4f", candidate.fitness)).font(.caption.monospaced()).foregroundStyle(.accentColor)
                        }
                    }
                }
            }
            if let searchError { Text(searchError).font(.caption).foregroundStyle(.red) }
        }
        .padding(14).background(.background.opacity(0.65), in: RoundedRectangle(cornerRadius: 14))
    }

    private func metric(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 3) { Text(title).font(.caption).foregroundStyle(.secondary); Text(value).font(.system(.title3, design: .rounded).weight(.semibold)) }.frame(maxWidth: .infinity, alignment: .leading)
    }

    private var technicalPanel: some View {
        DisclosureGroup("Technical detail", isExpanded: Binding(get: { selectedPhase == "Technical" }, set: { selectedPhase = $0 ? "Technical" : "Pipeline" })) {
            VStack(alignment: .leading, spacing: 6) {
                detail("Artifact", engineController.activeCImagePath)
                detail("Backend", "Unavailable until compiler event stream connects")
                detail("SLC / ANE telemetry", "Best-effort diagnostics only")
            }.padding(.top, 8)
        }
        .font(.headline)
        .padding(14).background(.background.opacity(0.65), in: RoundedRectangle(cornerRadius: 14))
    }

    private func detail(_ label: String, _ value: String) -> some View { HStack { Text(label).foregroundStyle(.secondary); Spacer(); Text(value).font(.caption.monospaced()).lineLimit(1) } }
}

private struct EventProgressChart: View {
    let events: [CompilerLabEvent]

    var body: some View {
        GeometryReader { proxy in
            Canvas { context, size in
                guard events.count > 1 else { return }
                let step = size.width / CGFloat(max(events.count - 1, 1))
                var path = Path()
                for (index, event) in events.enumerated() {
                    let point = CGPoint(x: CGFloat(index) * step, y: size.height * (1 - event.progress))
                    if index == 0 { path.move(to: point) } else { path.addLine(to: point) }
                }
                context.stroke(path, with: .color(.accentColor), lineWidth: 2)
            }
            .overlay(alignment: .bottomLeading) { Text("compiler progress").font(.caption2).foregroundStyle(.tertiary) }
        }
    }
}
