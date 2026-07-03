//
//  ConversationSettingsSheet.swift
//  PrismAgent
//
//  Created by PrismAgent on 7/1/26.
//

import SwiftUI

struct ConversationSettingsSheet: View {
    @Binding var selectedModel: String
    @Binding var systemPrompt: String
    @Binding var temperature: Double
    @Binding var topP: Double
    @Binding var maxTokens: Int
    @Binding var streamingEnabled: Bool
    @Binding var conversationMemoryEnabled: Bool
    @Binding var toolUseEnabled: Bool
    @Binding var showAuth: Bool
    @Binding var apiEndpoint: String
    @Binding var apiKey: String
    @Environment(\.dismiss) var dismiss
    
    var modelOptions: [String]
    var onDownloadModel: (() -> Void)?
    var onClearConversation: (() -> Void)?
    var onResetSettings: (() -> Void)?
    var memoryUsage: String?
    var inferenceSpeed: String?
    var deviceName: String?
    var computeUnits: String?
    var engineVersion: String?
    var modelLoaded: Bool = false
    
    var body: some View {
        NavigationStack {
            Form {
                GroupBox {
                    Toggle(isOn: $showAuth) {
                        Text("Require Authentication")
                    }
                    
                    if showAuth {
                        TextField("API Endpoint", text: $apiEndpoint)
                            .textFieldStyle(.roundedBorder)
                        SecureField("API Key", text: $apiKey)
                            .textFieldStyle(.roundedBorder)
                    }
                } label: {
                    Label("Account", systemImage: "person.crop.circle")
                }
                
                GroupBox {
                    Picker("Model", selection: $selectedModel) {
                        ForEach(modelOptions, id: \.self) { model in
                            Text(model).tag(model)
                        }
                    }
                    
                    if modelLoaded {
                        Label("Model loaded", systemImage: "checkmark.circle.fill")
                            .font(.caption)
                            .foregroundColor(.green)
                    }
                    
                    if let onDownloadModel {
                        Button(action: onDownloadModel) {
                            Label("Download Model", systemImage: "icloud.and.arrow.down")
                        }
                    }
                } label: {
                    Label("Model", systemImage: "cpu")
                }
                
                GroupBox {
                    if let memoryUsage {
                        HStack {
                            Text("Memory")
                            Spacer()
                            Text(memoryUsage)
                                .foregroundStyle(.secondary)
                        }
                    }
                    
                    if let inferenceSpeed {
                        HStack {
                            Text("Inference")
                            Spacer()
                            Text(inferenceSpeed)
                                .foregroundStyle(.secondary)
                        }
                    }
                    
                    TextField("System Prompt", text: $systemPrompt, axis: .vertical)
                        .lineLimit(3...6)
                        .textFieldStyle(.roundedBorder)
                    
                    VStack(spacing: 8) {
                        HStack {
                            Text("Temperature")
                            Spacer()
                            Text(String(format: "%.2f", temperature))
                                .foregroundStyle(.secondary)
                                .monospacedDigit()
                        }
                        Slider(value: $temperature, in: 0...2, step: 0.05)
                    }
                    
                    VStack(spacing: 8) {
                        HStack {
                            Text("Top-P")
                            Spacer()
                            Text(String(format: "%.2f", topP))
                                .foregroundStyle(.secondary)
                                .monospacedDigit()
                        }
                        Slider(value: $topP, in: 0...1, step: 0.05)
                    }
                    
                    VStack(spacing: 8) {
                        HStack {
                            Text("Max Tokens")
                            Spacer()
                            Text("\(maxTokens)")
                                .foregroundStyle(.secondary)
                                .monospacedDigit()
                        }
                        Slider(
                            value: Binding(
                                get: { Double(maxTokens) },
                                set: { maxTokens = Int($0) }
                            ),
                            in: 64...4096,
                            step: 64
                        )
                    }
                } label: {
                    Label("Runtime", systemImage: "chart.bar.xaxis")
                }
                
                GroupBox {
                    if let deviceName {
                        HStack {
                            Text("Device")
                            Spacer()
                            Text(deviceName)
                                .foregroundStyle(.secondary)
                        }
                    }
                    
                    if let computeUnits {
                        HStack {
                            Text("Compute Units")
                            Spacer()
                            Text(computeUnits)
                                .foregroundStyle(.secondary)
                        }
                    }
                } label: {
                    Label("Hardware", systemImage: "memorychip")
                }
                
                GroupBox {
                    Toggle(isOn: $streamingEnabled) {
                        Text("Streaming Output")
                    }
                    
                    Toggle(isOn: $conversationMemoryEnabled) {
                        Text("Conversation Memory")
                    }
                    
                    Toggle(isOn: $toolUseEnabled) {
                        Text("Tool Use")
                    }
                } label: {
                    Label("Features", systemImage: "switch.2")
                }
                
                GroupBox {
                    if let engineVersion {
                        HStack {
                            Text("Version")
                            Spacer()
                            Text(engineVersion)
                                .foregroundStyle(.secondary)
                        }
                    }
                } label: {
                    Label("Engine", systemImage: "gearshape.2")
                }
                
                GroupBox {
                    if let onClearConversation {
                        Button(role: .destructive, action: onClearConversation) {
                            Label("Clear Conversation", systemImage: "trash")
                        }
                    }
                    
                    if let onResetSettings {
                        Button(role: .destructive, action: onResetSettings) {
                            Label("Reset Settings", systemImage: "arrow.counterclockwise")
                        }
                    }
                } label: {
                    Label("Actions", systemImage: "bolt")
                }
            }
            .formStyle(.grouped)
            .navigationTitle("Settings")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

#Preview {
    ConversationSettingsSheet(
        selectedModel: .constant("llama-3.2-3b"),
        systemPrompt: .constant("You are a helpful assistant."),
        temperature: .constant(0.7),
        topP: .constant(0.9),
        maxTokens: .constant(2048),
        streamingEnabled: .constant(true),
        conversationMemoryEnabled: .constant(true),
        toolUseEnabled: .constant(true),
        showAuth: .constant(false),
        apiEndpoint: .constant("http://localhost:8080"),
        apiKey: .constant(""),
        modelOptions: ["llama-3.2-3b", "llama-3.1-8b", "mistral-7b"],
        onDownloadModel: {},
        onClearConversation: {},
        onResetSettings: {},
        memoryUsage: "2.4 GB / 8.0 GB",
        inferenceSpeed: "24.5 tok/s",
        deviceName: "Apple M1",
        computeUnits: "8 GPU / 16 Neural",
        engineVersion: "0.1.0",
        modelLoaded: true
    )
}
