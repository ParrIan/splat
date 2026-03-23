import SwiftUI

struct ContentView: View {
    @StateObject private var coordinator = SensorCoordinator()
    @State private var exportURL: URL?
    @State private var previewVideoURL: URL?

    var body: some View {
        ZStack {
            CameraPreviewView(session: coordinator.captureManager.session)
                .ignoresSafeArea()

            VStack {
                StatusBar(coordinator: coordinator, exporter: coordinator.exporter)
                    .padding(.top, 60)

                Spacer()

                ControlBar(coordinator: coordinator, onPreview: { previewVideoURL = coordinator.previewVideoURL })
                    .padding(.bottom, 50)
            }
        }
        .sheet(item: $exportURL) { url in
            ShareSheet(url: url)
        }
        .sheet(item: $previewVideoURL) { url in
            VideoPreviewView(url: url)
        }
        .onReceive(coordinator.$exportURL) { url in
            exportURL = url
        }
    }
}

struct StatusBar: View {
    @ObservedObject var coordinator: SensorCoordinator
    @ObservedObject var exporter: SessionExporter

    var body: some View {
        HStack(spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text(coordinator.captureManager.statusMessage)
                    .font(.subheadline)
                if coordinator.isCapturing {
                    Text("\(coordinator.captureManager.frameCount) frames")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            Spacer()

            if exporter.isExporting {
                ProgressView(value: Double(exporter.progress))
                    .frame(width: 80)
            }
        }
        .foregroundColor(.white)
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(Color.black.opacity(0.6))
        .cornerRadius(10)
        .padding(.horizontal, 16)
    }
}

struct ControlBar: View {
    @ObservedObject var coordinator: SensorCoordinator
    let onPreview: () -> Void

    var body: some View {
        HStack(spacing: 24) {
            if coordinator.readyToExport && !coordinator.isCapturing {
                Button("Preview") {
                    onPreview()
                }
                .buttonStyle(CaptureButtonStyle(color: .orange))

                Button("Export") {
                    coordinator.exportSession()
                }
                .buttonStyle(CaptureButtonStyle(color: .blue))
            }

            Button(coordinator.isCapturing ? "Stop" : "Record") {
                coordinator.isCapturing ? coordinator.stop() : coordinator.start()
            }
            .buttonStyle(CaptureButtonStyle(color: coordinator.isCapturing ? .red : .white))
        }
    }
}

struct CaptureButtonStyle: ButtonStyle {
    let color: Color

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.headline)
            .foregroundColor(color == .white ? .black : .white)
            .frame(width: 120, height: 50)
            .background(color.opacity(configuration.isPressed ? 0.7 : 1))
            .cornerRadius(25)
    }
}


struct ShareSheet: UIViewControllerRepresentable {
    let url: URL

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: [url], applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

extension URL: @retroactive Identifiable {
    public var id: String { absoluteString }
}

#Preview {
    ContentView()
}
