import SwiftUI
import ARKit
import RealityKit

struct CameraPreviewView: UIViewRepresentable {
    let session: ARSession

    func makeUIView(context: Context) -> ARView {
        let view = ARView(frame: .zero)
        view.session = session
        view.renderOptions = [
            .disableDepthOfField,
            .disableMotionBlur,
            .disableHDR,
            .disableFaceOcclusions,
            .disablePersonOcclusion,
            .disableGroundingShadows,
            .disableCameraGrain
        ]
        view.environment.background = .cameraFeed()
        return view
    }

    func updateUIView(_ uiView: ARView, context: Context) {}
}
