use std::process::Command;

#[test]
fn help_lists_public_commands() {
    let output = Command::new(env!("CARGO_BIN_EXE_crabsaxs"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(output.status.success());
    let text = String::from_utf8_lossy(&output.stdout);
    for command in [
        "calc", "fit", "score", "ensemble", "batch", "inspect", "compare", "info",
    ] {
        assert!(text.contains(command), "missing {command} in help");
    }
}

#[test]
fn calc_and_score_machine_output_are_valid() {
    let directory = tempfile::tempdir().unwrap();
    let curve = directory.path().join("curve.dat");
    let status = Command::new(env!("CARGO_BIN_EXE_crabsaxs"))
        .args([
            "calc",
            "examples/data/small.pdb",
            "--q-min",
            "0.01",
            "--q-max",
            "0.05",
            "--q-step",
            "0.01",
            "--output",
            curve.to_str().unwrap(),
            "--force",
        ])
        .status()
        .unwrap();
    assert!(status.success());
    assert!(curve.exists());

    let output = Command::new(env!("CARGO_BIN_EXE_crabsaxs"))
        .args([
            "--quiet",
            "--format",
            "json",
            "score",
            "examples/data/small.pdb",
            "examples/data/test_experimental.dat",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(json.get("reduced_chi2").is_some());
    assert!(output.stderr.is_empty(), "quiet mode wrote to stderr");
}

#[test]
fn ensemble_writes_population_report() {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("ensemble");
    let status = Command::new(env!("CARGO_BIN_EXE_crabsaxs"))
        .args([
            "ensemble",
            "examples/data/small.pdb",
            "examples/data/test_structure.pdb",
            "--experiment",
            "examples/data/test_experimental.dat",
            "--max-conformers",
            "2",
            "--output",
            output.to_str().unwrap(),
            "--force",
        ])
        .status()
        .unwrap();
    assert!(status.success());
    assert!(output.join("ensemble.json").exists());
    assert!(output.join("populations.csv").exists());
}
