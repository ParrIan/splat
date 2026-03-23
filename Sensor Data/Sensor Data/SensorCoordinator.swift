import Foundation
import Combine

class SensorCoordinator: ObservableObject {
    let captureManager = ARCaptureManager()
    let exporter = SessionExporter()

    @Published var isCapturing = false
    @Published var readyToExport = false
    @Published var capturedFrames: [CaptureFrame] = []
    @Published var exportURL: URL?
    @Published var previewVideoURL: URL?

    private var cancellables = Set<AnyCancellable>()

    init() {
        captureManager.startPreview()

        captureManager.$isRunning
            .assign(to: &$isCapturing)

        exporter.$exportURL
            .assign(to: &$exportURL)

        captureManager.$previewVideoURL
            .assign(to: &$previewVideoURL)

        captureManager.onSessionEnd = { [weak self] frames in
            guard let self else { return }
            self.capturedFrames = frames
            self.readyToExport = !frames.isEmpty
        }
    }

    func start() {
        readyToExport = false
        capturedFrames = []
        captureManager.start()
    }

    func stop() {
        captureManager.stop()
    }

    func exportSession() {
        exporter.export(frames: capturedFrames)
    }
}
