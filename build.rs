fn main() {
    tonic_build::configure()
        .type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]")
        .compile(&["proto/messages.proto", "proto/service.proto"], &["proto"])
        .unwrap();
}
