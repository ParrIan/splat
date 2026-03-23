import SwiftUI
import AVKit

struct VideoPreviewView: View {
    let url: URL
    @Environment(\.dismiss) private var dismiss

    private let player: AVPlayer

    init(url: URL) {
        self.url = url
        self.player = AVPlayer(url: url)
    }

    var body: some View {
        NavigationStack {
            VideoPlayer(player: player)
                .ignoresSafeArea()
                .navigationTitle("Preview")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .confirmationAction) {
                        Button("Done") { dismiss() }
                    }
                }
        }
        .onAppear { player.play() }
        .onDisappear { player.pause() }
    }
}
