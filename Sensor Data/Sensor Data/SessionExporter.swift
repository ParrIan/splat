import Foundation
import UIKit
import Combine
import ZIPFoundation
import simd

class SessionExporter: ObservableObject {
    @Published var isExporting = false
    @Published var progress: Float = 0
    @Published var exportURL: URL?

    private let exportQueue = DispatchQueue(label: "session.export", qos: .userInitiated)

    func export(frames: [CaptureFrame]) {
        guard !frames.isEmpty else { return }

        DispatchQueue.main.async {
            self.isExporting = true
            self.progress = 0
            self.exportURL = nil
        }

        exportQueue.async {
            do {
                let url = try self.writeBundle(frames: frames)
                DispatchQueue.main.async {
                    self.isExporting = false
                    self.progress = 1
                    self.exportURL = url
                }
            } catch {
                DispatchQueue.main.async {
                    self.isExporting = false
                    print("Export failed: \(error)")
                }
            }
        }
    }

    private func writeBundle(frames: [CaptureFrame]) throws -> URL {
        let fm = FileManager.default
        let tmp = fm.temporaryDirectory.appendingPathComponent("splat_bundle_\(Int(Date().timeIntervalSince1970))", isDirectory: true)
        let imagesDir = tmp.appendingPathComponent("images", isDirectory: true)

        try fm.createDirectory(at: tmp, withIntermediateDirectories: true)
        try fm.createDirectory(at: imagesDir, withIntermediateDirectories: true)

        var records: [FrameRecord] = []

        for (i, frame) in frames.enumerated() {
            let filename = String(format: "frame_%06d.jpg", i)
            let dest = imagesDir.appendingPathComponent(filename)

            try fm.copyItem(at: frame.imageURL, to: dest)

            let m = frame.intrinsics
            let t = frame.transform

            // ARKit intrinsics are in full-resolution pixel coords.
            // Images are saved at half resolution (960x720), so scale by 0.5.
            let scale: Float = 0.5
            let record = FrameRecord(
                index: i,
                timestamp: frame.timestamp,
                filename: "images/\(filename)",
                intrinsics: IntrinsicsRecord(
                    fx: m[0][0] * scale,
                    fy: m[1][1] * scale,
                    cx: m[2][0] * scale,
                    cy: m[2][1] * scale
                ),
                transform: TransformRecord(values: flattenRowMajor(t))
            )
            records.append(record)

            let p = Float(i + 1) / Float(frames.count)
            DispatchQueue.main.async { self.progress = p * 0.8 }
        }

        guard let first = frames.first else { throw ExportError.noFrames }

        // image dimensions from first frame on disk
        let imageSize = imageDimensions(at: first.imageURL)

        let manifest = SessionManifest(
            frameCount: records.count,
            capturedAt: first.timestamp,
            imageWidth: imageSize.width,
            imageHeight: imageSize.height
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted

        try encoder.encode(manifest).write(to: tmp.appendingPathComponent("manifest.json"))
        try encoder.encode(records).write(to: tmp.appendingPathComponent("frames.json"))

        DispatchQueue.main.async { self.progress = 0.9 }

        let zipURL = fm.temporaryDirectory.appendingPathComponent("splat_session_\(Int(Date().timeIntervalSince1970)).zip")
        try fm.zipItem(at: tmp, to: zipURL)
        try fm.removeItem(at: tmp)

        return zipURL
    }

    private func imageDimensions(at url: URL) -> (width: Int, height: Int) {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
              let w = props[kCGImagePropertyPixelWidth] as? Int,
              let h = props[kCGImagePropertyPixelHeight] as? Int
        else { return (0, 0) }
        return (w, h)
    }

    private func flattenRowMajor(_ m: simd_float4x4) -> [Float] {
        [
            m.columns.0.x, m.columns.1.x, m.columns.2.x, m.columns.3.x,
            m.columns.0.y, m.columns.1.y, m.columns.2.y, m.columns.3.y,
            m.columns.0.z, m.columns.1.z, m.columns.2.z, m.columns.3.z,
            m.columns.0.w, m.columns.1.w, m.columns.2.w, m.columns.3.w
        ]
    }

    enum ExportError: Error {
        case noFrames
    }
}
