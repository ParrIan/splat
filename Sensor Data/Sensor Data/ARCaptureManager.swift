import ARKit
import AVFoundation
import Combine
import UIKit

// Lightweight record kept during capture — no pixel buffer retained
private struct PoseRecord {
    let index: Int
    let timestamp: TimeInterval
    let intrinsics: simd_float3x3
    let transform: simd_float4x4
}

class ARCaptureManager: NSObject, ObservableObject {
    @Published var isRunning = false
    @Published var frameCount = 0
    @Published var statusMessage = "Ready"
    @Published var previewVideoURL: URL?

    let session = ARSession()
    private var recording = false

    // capture path — only poses, no pixel buffers
    private var poseRecords: [PoseRecord] = []
    private var pendingPixelBuffers: [(Int, CVPixelBuffer)] = []
    private let captureQueue = DispatchQueue(label: "capture.poses", qos: .userInteractive)

    // jpeg encoding — concurrent, lower priority
    private let encodeQueue = DispatchQueue(label: "capture.encode", qos: .utility, attributes: .concurrent)
    private let encodeGroup = DispatchGroup()

    private var sessionDir: URL?
    private var frameIndex = 0
    private var captureFrameCounter = 0

    private var assetWriter: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var pixelBufferAdaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var videoURL: URL?
    private var firstFrameTime: CMTime?

    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    var onSessionEnd: (([CaptureFrame]) -> Void)?

    override init() {
        super.init()
        session.delegate = self
    }

    func startPreview() {
        let config = ARWorldTrackingConfiguration()
        config.frameSemantics = []
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
        DispatchQueue.main.async { self.statusMessage = "Ready" }
        prepareNextSession()
    }

    private func prepareNextSession() {
        captureQueue.async {
            let dir = FileManager.default.temporaryDirectory
                .appendingPathComponent("splat_\(Int(Date().timeIntervalSince1970))", isDirectory: true)
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            self.sessionDir = dir
            self.setupVideoWriter(in: dir)
        }
    }

    func start() {
        captureQueue.async {
            self.poseRecords.removeAll()
            self.frameIndex = 0
            self.captureFrameCounter = 0
            self.firstFrameTime = nil
        }

        previewVideoURL = nil
        recording = true

        DispatchQueue.main.async {
            self.isRunning = true
            self.frameCount = 0
            self.statusMessage = "Recording..."
        }
    }

    func stop() {
        recording = false
        session.pause()

        DispatchQueue.main.async { self.statusMessage = "Processing..." }

        captureQueue.async {
            let poses = self.poseRecords
            let dir = self.sessionDir

            // wait for all concurrent JPEG encodes to finish
            self.encodeGroup.notify(queue: self.captureQueue) {
                self.finalizeVideo { [weak self] videoURL in
                    guard let self, let dir else { return }

                    let frames: [CaptureFrame] = poses.map { pose in
                        CaptureFrame(
                            timestamp: pose.timestamp,
                            intrinsics: pose.intrinsics,
                            transform: pose.transform,
                            imageURL: dir.appendingPathComponent(String(format: "frame_%06d.jpg", pose.index))
                        )
                    }

                    self.prepareNextSession()
                    DispatchQueue.main.async {
                        self.isRunning = false
                        self.statusMessage = "Captured \(frames.count) frames"
                        self.previewVideoURL = videoURL
                        self.onSessionEnd?(frames)
                    }
                }
            }
        }
    }

    private func setupVideoWriter(in dir: URL) {
        let url = dir.appendingPathComponent("preview.mov")
        videoURL = url

        guard let writer = try? AVAssetWriter(outputURL: url, fileType: .mov) else { return }

        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: 960,
            AVVideoHeightKey: 720
        ]

        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = true
        input.transform = CGAffineTransform(rotationAngle: .pi / 2)

        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: nil
        )

        writer.add(input)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        assetWriter = writer
        videoInput = input
        pixelBufferAdaptor = adaptor
    }

    private func finalizeVideo(completion: @escaping (URL?) -> Void) {
        guard let writer = assetWriter, let input = videoInput else {
            completion(nil)
            return
        }
        input.markAsFinished()
        writer.finishWriting {
            completion(writer.status == .completed ? self.videoURL : nil)
        }
    }

    private func appendToVideo(_ pixelBuffer: CVPixelBuffer, at timestamp: TimeInterval) {
        guard let input = videoInput, let adaptor = pixelBufferAdaptor, input.isReadyForMoreMediaData else { return }

        let cmTime = CMTime(seconds: timestamp, preferredTimescale: 600)
        if firstFrameTime == nil { firstFrameTime = cmTime }
        guard let first = firstFrameTime else { return }

        adaptor.append(pixelBuffer, withPresentationTime: CMTimeSubtract(cmTime, first))
    }

    private func encodeJPEG(_ pixelBuffer: CVPixelBuffer, to url: URL) {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else { return }
        let original = UIImage(cgImage: cgImage)
        let target = CGSize(width: 960, height: 720)
        let renderer = UIGraphicsImageRenderer(size: target)
        let resized = renderer.image { _ in original.draw(in: CGRect(origin: .zero, size: target)) }
        guard let data = resized.jpegData(compressionQuality: 0.7) else { return }
        try? data.write(to: url)
    }
}

extension ARCaptureManager: ARSessionDelegate {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard recording else { return }

        // capture at ~10fps — ARKit delivers 60fps, skip 5 of every 6 frames
        captureFrameCounter += 1
        guard captureFrameCounter % 6 == 0 else { return }

        let pixelBuffer = frame.capturedImage
        let timestamp = frame.timestamp
        let intrinsics = frame.camera.intrinsics
        let transform = frame.camera.transform

        captureQueue.async {
            let index = self.frameIndex
            self.frameIndex += 1

            // record pose immediately — fast, no encoding
            self.poseRecords.append(PoseRecord(
                index: index,
                timestamp: timestamp,
                intrinsics: intrinsics,
                transform: transform
            ))

            // append raw buffer to video — fast
            self.appendToVideo(pixelBuffer, at: timestamp)

            // encode JPEG concurrently — does not block capture path
            guard let dir = self.sessionDir else { return }
            let url = dir.appendingPathComponent(String(format: "frame_%06d.jpg", index))

            self.encodeGroup.enter()
            self.encodeQueue.async {
                self.encodeJPEG(pixelBuffer, to: url)
                self.encodeGroup.leave()
            }

            let count = self.poseRecords.count
            DispatchQueue.main.async { self.frameCount = count }
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        DispatchQueue.main.async {
            self.statusMessage = "ARSession error: \(error.localizedDescription)"
            self.isRunning = false
        }
    }
}
