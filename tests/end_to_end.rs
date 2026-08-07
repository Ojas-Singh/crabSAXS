//! Offline end-to-end calculation test using the committed two-atom fixture.

#[test]
fn compute_curve_for_example_structure_is_well_formed() {
    let structure = crabsaxs::Structure::from_pdb_file("examples/data/small.pdb").unwrap();
    let q_values: Vec<f64> = (0..100).map(|i| i as f64 * 0.005).collect();
    let curve = crabsaxs::scattering::debye_intensity(&structure, &q_values);
    assert_eq!(curve.len(), q_values.len());
    assert!(curve.iter().all(|v| v.is_finite() && *v >= 0.0));
}

#[test]
fn compute_cli_writes_solution_scattering_curve() {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("solution.dat");
    let status = std::process::Command::new(env!("CARGO_BIN_EXE_crabsaxs"))
        .args([
            "calc",
            "examples/data/small.pdb",
            "--qmin",
            "0.01",
            "--qmax",
            "0.05",
            "--qstep",
            "0.01",
            "--method",
            "auto",
            "--hydration",
            "adaptive-grid",
            "--c1",
            "1.0",
            "--c2",
            "0.02",
            "--output",
            output.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success());
    let curve = std::fs::read_to_string(output).unwrap();
    let rows = curve
        .lines()
        .filter(|line| !line.starts_with('#'))
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 5);
    assert!(rows.iter().all(|row| {
        row.split_whitespace()
            .all(|value| value.parse::<f64>().is_ok_and(f64::is_finite))
    }));
}
