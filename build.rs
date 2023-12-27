fn main() {
    tonic_build::configure()
        .compile(&["proto/messages.proto"], &["proto"])
        .unwrap();
}