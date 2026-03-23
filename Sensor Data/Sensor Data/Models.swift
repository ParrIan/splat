import Foundation
import CoreVideo
import simd

struct CaptureFrame {
    let timestamp: TimeInterval
    let intrinsics: simd_float3x3
    let transform: simd_float4x4
    let imageURL: URL
}

struct SessionManifest: Codable {
    let frameCount: Int
    let capturedAt: TimeInterval
    let imageWidth: Int
    let imageHeight: Int
}

struct FrameRecord: Codable {
    let index: Int
    let timestamp: TimeInterval
    let filename: String
    let intrinsics: IntrinsicsRecord
    let transform: TransformRecord
}

struct IntrinsicsRecord: Codable {
    let fx: Float
    let fy: Float
    let cx: Float
    let cy: Float
}

struct TransformRecord: Codable {
    // row-major flattened 4x4
    let values: [Float]
}
