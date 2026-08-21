use std::path::Path;
use std::process::Command;

#[test]
fn libm_gate_passes_newt_src() {
    let source_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let binary = std::env::current_exe()
        .expect("smoke test path should be available")
        .parent()
        .expect("smoke test should have a parent")
        .parent()
        .expect("cargo test binary should be under target/debug/deps")
        .join("libm-gate");
    let output = Command::new(binary)
        .arg(source_dir)
        .output()
        .expect("libm-gate should start");

    assert!(
        output.status.success(),
        "libm-gate failed:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(
        output.stdout.is_empty(),
        "libm-gate reported matches:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
}
