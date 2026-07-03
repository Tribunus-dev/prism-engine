import SwiftUI

/// A smooth bezier waveform view that renders 64 float samples (0–1) as a
/// Catmull-Rom spline with a gradient fill fading to transparent at the top.
struct VoiceWaveformView: View {
    let waveform: [Float]
    
    init(waveform: [Float] = Array(repeating: 0.5, count: 64)) {
        self.waveform = waveform
    }
    
    var body: some View {
        WaveformShape(waveform: waveform)
            .fill(
                LinearGradient(
                    colors: [
                        Color.accentColor.opacity(0.0),
                        Color.accentColor.opacity(0.3),
                        Color.accentColor
                    ],
                    startPoint: .top,
                    endPoint: .bottom
                )
            )
            .overlay(
                WaveformShape(waveform: waveform)
                    .stroke(Color.accentColor.opacity(0.8), lineWidth: 1.5)
            )
            .frame(height: 48)
            .shadow(color: Color.accentColor.opacity(0.2), radius: 4)
    }
}

// MARK: - Waveform Shape

private struct WaveformShape: Shape {
    let waveform: [Float]

    func path(in rect: CGRect) -> Path {
        Path { path in
            let count = waveform.count
            guard count > 1 else { return }

            let stepX = rect.width / CGFloat(count - 1)
            let bottomY = rect.height
            let midY = rect.height / 2
            let amplitude = rect.height / 2

            // Convert waveform values to points (centered vertically)
            let points: [CGPoint] = (0..<count).map { i in
                CGPoint(
                    x: CGFloat(i) * stepX,
                    y: midY - CGFloat(waveform[i]) * amplitude
                )
            }

            // Start from the left edge
            path.move(to: CGPoint(x: 0, y: bottomY))
            path.addLine(to: points[0])

            // Catmull-Rom spline through all points
            // For segment i → i+1:
            //   cp1 = points[i] + (points[i+1] - points[i-1]) / 6
            //   cp2 = points[i+1] - (points[i+2] - points[i]) / 6
            for i in 0..<(count - 1) {
                let p0 = i > 0 ? points[i - 1] : points[i]
                let p1 = points[i]
                let p2 = points[i + 1]
                let p3 = i < count - 2 ? points[i + 2] : points[i + 1]

                let cp1 = CGPoint(
                    x: p1.x + (p2.x - p0.x) / 6,
                    y: p1.y + (p2.y - p0.y) / 6
                )
                let cp2 = CGPoint(
                    x: p2.x - (p3.x - p1.x) / 6,
                    y: p2.y - (p3.y - p1.y) / 6
                )

                path.addCurve(to: p2, control1: cp1, control2: cp2)
            }

            // Complete the filled shape down to the bottom
            if let last = points.last {
                path.addLine(to: CGPoint(x: last.x, y: bottomY))
            }
            path.closeSubpath()
        }
    }
}

// MARK: - Preview

#Preview {
    VoiceWaveformView(waveform: (0..<64).map {
        sin(Float($0) * 0.2) * 0.5 + 0.5
    })
    .padding()
}
