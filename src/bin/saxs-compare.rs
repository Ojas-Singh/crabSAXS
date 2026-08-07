//! Opt-in, black-box comparison against a locally installed Pepsi-SAXS.
//!
//! Both programs are timed as end-to-end subprocesses with matched atom,
//! background, smearing, hydration-resolution, and q-range policies. The
//! executable is never inspected internally; only documented command-line
//! options and public curve output are used.

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use crabsaxs::fit::{fit_to_experimental_with_options, BackgroundMode, FitOptions, FitParams};
use crabsaxs::io::ExperimentalCurve;
use crabsaxs::scattering::{compute_curve, ComputeOptions, ScatteringMethod};
use crabsaxs::structure::{OccupancyPolicy, ParseOptions, Structure};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread::sleep;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum Background {
    /// Fit a q-independent offset in both programs (`-cst` for Pepsi-SAXS).
    Fit,
    /// Constrain the offset to zero in both programs.
    Zero,
}

#[derive(Debug, Serialize)]
struct CurveParity {
    fitted_scale: f64,
    normalized_nrmse: f64,
    relative_l2_rmse: f64,
}

#[derive(Debug, Serialize)]
struct ComponentParity {
    atomic: CurveParity,
    excluded_volume: CurveParity,
    hydration: CurveParity,
    atomic_excluded_cross: CurveParity,
    atomic_hydration_cross: CurveParity,
    excluded_hydration_cross: CurveParity,
}

impl From<Background> for BackgroundMode {
    fn from(value: Background) -> Self {
        match value {
            Background::Fit => Self::Fit,
            Background::Zero => Self::FixedZero,
        }
    }
}

#[derive(Parser)]
struct Args {
    #[arg(long)]
    pepsi_bin: Option<PathBuf>,
    /// `crabsaxs` executable to time; defaults to the sibling of this binary.
    #[arg(long)]
    crab_bin: Option<PathBuf>,
    #[arg(long)]
    pdb: PathBuf,
    #[arg(long)]
    exp: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = Background::Fit)]
    background: Background,
    /// Include non-water HETATM records in crabSAXS, matching Pepsi-SAXS.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    include_hetatm: bool,
    /// Multipole-basis q stride used by crabSAXS fitting (1 is full accuracy).
    #[arg(long, default_value_t = 20)]
    q_sampling_stride: usize,
    /// crabSAXS adaptive-grid spacing matched to Pepsi's `-fast` mode.
    #[arg(long, default_value_t = 8.0)]
    hydration_spacing: f64,
    #[arg(long, default_value_t = 10)]
    repetitions: usize,
    #[arg(long, default_value = "compare-report.json")]
    output: PathBuf,
    #[arg(long, default_value_t = 120)]
    timeout_seconds: u64,
}

#[derive(Debug, Serialize)]
struct Report {
    report_schema: String,
    host_arch: String,
    host_os: String,
    rust_version: String,
    build_profile: String,
    thread_count: usize,
    pepsi_bin: String,
    pepsi_sha256: String,
    pepsi_file_description: String,
    pepsi_execution_mode: String,
    crab_bin: String,
    crab_sha256: String,
    crab_file_description: String,
    pdb: String,
    pdb_sha256: String,
    experimental_sha256: Option<String>,
    atom_count: usize,
    include_non_water_hetatm: bool,
    occupancy_policy: String,
    background: Background,
    q_sampling_stride: usize,
    hydration_spacing_angstrom: Option<f64>,
    pepsi_hydration_points: Option<usize>,
    crab_hydration_points: Option<usize>,
    pepsi_fit_radius_angstrom: Option<f64>,
    pepsi_hydration_contrast_e_per_a3: Option<f64>,
    crab_fit_radius_angstrom: Option<f64>,
    crab_hydration_contrast_e_per_a3: Option<f64>,
    component_parity: Option<ComponentParity>,
    component_skip_reason: Option<String>,
    pepsi_arguments: Vec<String>,
    crab_arguments: Vec<String>,
    q_points: usize,
    q_min: f64,
    q_max: f64,
    comparison_scale: f64,
    normalized_nrmse: f64,
    relative_l2_rmse: f64,
    mean_absolute_relative_error: f64,
    max_relative_error: f64,
    max_relative_error_q: f64,
    pepsi_median_ms: f64,
    pepsi_mad_ms: f64,
    crab_median_ms: f64,
    crab_mad_ms: f64,
    pepsi_chi2: Option<f64>,
    crab_chi2: Option<f64>,
    crab_fit_params: Option<FitParams>,
    pepsi_over_crab_speedup: f64,
    accuracy_gate_passed: bool,
    fit_quality_gate_passed: bool,
    speed_gate_passed: bool,
    component_gate_passed: bool,
    target_gate_passed: bool,
    repetitions: usize,
    warmups: usize,
    timing_methodology: String,
}

fn run_with_timeout(mut child: Child, timeout: Duration) -> Result<ExitStatus> {
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(status);
        }
        if start.elapsed() > timeout {
            child.kill()?;
            bail!("subprocess exceeded {} seconds", timeout.as_secs());
        }
        sleep(Duration::from_millis(1));
    }
}

fn parse_curve(path: &Path, fitted: bool) -> Result<(Vec<f64>, Vec<f64>)> {
    let mut q = Vec::new();
    let mut intensity = Vec::new();
    for line in BufReader::new(File::open(path)?).lines() {
        let line = line?;
        let parsed: std::result::Result<Vec<f64>, _> = line
            .replace(',', " ")
            .split_whitespace()
            .map(str::parse)
            .collect();
        let Ok(values) = parsed else {
            continue;
        };
        if values.len() >= 2
            && values[0].is_finite()
            && values[1..].iter().all(|value| value.is_finite())
        {
            // Fit files contain q, I_exp, sigma, I_fit. Calculation files
            // expose the atomic component as their third numeric column.
            q.push(values[0]);
            intensity.push(if fitted {
                *values.last().unwrap()
            } else {
                *values.get(2).unwrap_or(&values[1])
            });
        }
    }
    if q.is_empty() {
        bail!("no numeric q/I rows found in {}", path.display());
    }
    Ok((q, intensity))
}

fn validate_q_grid(reference: &[f64], candidate: &[f64]) -> Result<()> {
    if reference.len() != candidate.len() {
        bail!(
            "q-grid length mismatch: reference {}, candidate {}",
            reference.len(),
            candidate.len()
        );
    }
    let max_difference = reference
        .iter()
        .zip(candidate)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max);
    if max_difference > 5e-7 {
        bail!("q grids differ by as much as {max_difference:.3e} 1/A");
    }
    Ok(())
}

fn curve_parity(reference: &[f64], candidate: &[f64]) -> CurveParity {
    let dot = reference
        .iter()
        .zip(candidate)
        .map(|(left, right)| left * right)
        .sum::<f64>();
    let denominator = candidate.iter().map(|value| value * value).sum::<f64>();
    let scale = if denominator > 0.0 {
        dot / denominator
    } else {
        1.0
    };
    let squared_error = reference
        .iter()
        .zip(candidate)
        .map(|(left, right)| (scale * right - left).powi(2))
        .sum::<f64>();
    let rmse = (squared_error / reference.len().max(1) as f64).sqrt();
    let range = reference.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - reference.iter().copied().fold(f64::INFINITY, f64::min);
    let reference_l2 = reference.iter().map(|value| value * value).sum::<f64>();
    CurveParity {
        fitted_scale: scale,
        normalized_nrmse: rmse / range.abs().max(1e-30),
        relative_l2_rmse: (squared_error / reference_l2.max(1e-30)).sqrt(),
    }
}

fn pepsi_components(path: &Path) -> Result<Option<[Vec<f64>; 6]>> {
    let mut components: [Vec<f64>; 6] = std::array::from_fn(|_| Vec::new());
    for line in BufReader::new(File::open(path)?).lines() {
        let line = line?;
        let values = line
            .split_whitespace()
            .map(str::parse::<f64>)
            .collect::<std::result::Result<Vec<_>, _>>();
        let Ok(values) = values else { continue };
        if values.len() < 8 {
            continue;
        }
        for (target, source) in components.iter_mut().zip(&values[2..8]) {
            target.push(*source);
        }
    }
    Ok((!components[0].is_empty()).then_some(components))
}

fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn median_absolute_deviation(values: &[f64]) -> f64 {
    let centre = median(values);
    let deviations = values
        .iter()
        .map(|value| (value - centre).abs())
        .collect::<Vec<_>>();
    median(&deviations)
}

fn sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn file_description(path: &Path) -> String {
    Command::new("file")
        .arg("-b")
        .arg(path)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|description| description.trim().to_owned())
        .unwrap_or_else(|| "unknown".into())
}

fn execution_mode(description: &str) -> String {
    if cfg!(target_os = "macos")
        && std::env::consts::ARCH == "aarch64"
        && description.contains("x86_64")
    {
        "Rosetta 2 translation (x86_64 executable on aarch64 macOS)".into()
    } else if description.contains(std::env::consts::ARCH)
        || (std::env::consts::ARCH == "aarch64" && description.contains("arm64"))
    {
        "native".into()
    } else {
        "unknown or cross-architecture".into()
    }
}

fn rust_version() -> String {
    Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|version| version.trim().to_owned())
        .unwrap_or_else(|| "unknown".into())
}

fn pepsi_hydration_points(path: &Path) -> Result<Option<usize>> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    let log = String::from_utf8_lossy(&bytes);
    Ok(log.lines().find_map(|line| {
        line.strip_prefix("Number of points in the hydration shell")
            .and_then(|suffix| suffix.rsplit(':').next())
            .and_then(|value| value.trim().parse().ok())
    }))
}

fn pepsi_fit_parameters(path: &Path) -> Result<Option<(f64, f64)>> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    let log = String::from_utf8_lossy(&bytes);
    let value = |label: &str| {
        log.lines().find_map(|line| {
            line.strip_prefix(label)
                .and_then(|suffix| suffix.rsplit(':').next())
                .and_then(|value| value.split_whitespace().next())
                .and_then(|value| value.parse::<f64>().ok())
        })
    };
    Ok(value("Best r0 found").zip(value("Best d_rho found")))
}

fn sibling_crab_binary() -> Result<PathBuf> {
    let current = std::env::current_exe().context("locate comparison executable")?;
    let candidate = current
        .parent()
        .context("comparison executable has no parent directory")?
        .join("crabsaxs");
    if !candidate.exists() {
        bail!(
            "crab executable does not exist at {}; pass --crab-bin",
            candidate.display()
        );
    }
    Ok(candidate)
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.repetitions == 0
        || args.q_sampling_stride == 0
        || !args.hydration_spacing.is_finite()
        || args.hydration_spacing <= 0.0
    {
        bail!("repetitions, q-sampling-stride, and hydration-spacing must be positive");
    }
    let pepsi_bin = args
        .pepsi_bin
        .or_else(|| std::env::var_os("PEPSI_SAXS_BIN").map(PathBuf::from))
        .context("set --pepsi-bin or PEPSI_SAXS_BIN")?;
    if !pepsi_bin.exists() {
        bail!("Pepsi executable does not exist: {}", pepsi_bin.display());
    }
    let crab_bin = args.crab_bin.clone().map_or_else(sibling_crab_binary, Ok)?;
    if !crab_bin.exists() {
        bail!("crab executable does not exist: {}", crab_bin.display());
    }

    let temporary = tempfile::Builder::new().prefix("saxs-compare-").tempdir()?;
    let pepsi_out = temporary.path().join("pepsi.out");
    let pepsi_log = temporary.path().join("pepsi.log");
    let mut pepsi_arguments = vec![args.pdb.display().to_string()];
    if let Some(experimental) = &args.exp {
        pepsi_arguments.push(experimental.display().to_string());
    }
    pepsi_arguments.extend([
        "-o".into(),
        pepsi_out.display().to_string(),
        "--noSmearing".into(),
        "-fast".into(),
    ]);
    if args.exp.is_some() && matches!(args.background, Background::Fit) {
        pepsi_arguments.push("-cst".into());
    }

    let mut pepsi_times = Vec::new();
    for run in 0..(args.repetitions + 2) {
        let log = File::create(&pepsi_log)?;
        let start = Instant::now();
        let child = Command::new(&pepsi_bin)
            .args(&pepsi_arguments)
            .stdout(Stdio::from(log.try_clone()?))
            .stderr(Stdio::from(log))
            .spawn()
            .context("launch Pepsi-SAXS")?;
        let status = run_with_timeout(child, Duration::from_secs(args.timeout_seconds))?;
        if !status.success() {
            bail!("Pepsi-SAXS exited with {status}");
        }
        if run >= 2 {
            pepsi_times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
    }
    let (q, reference) = parse_curve(&pepsi_out, args.exp.is_some())?;
    let pepsi_fit_parameters = if args.exp.is_some() {
        pepsi_fit_parameters(&pepsi_log)?
    } else {
        None
    };
    let pepsi_hydration_points = pepsi_hydration_points(&pepsi_log)?;
    let pepsi_components = if args.exp.is_none() {
        pepsi_components(&pepsi_out)?
    } else {
        None
    };
    let q_min = *q.first().context("Pepsi output has no q values")?;
    let q_max = *q.last().context("Pepsi output has no q values")?;

    let crab_out = temporary.path().join("crab.out");
    let crab_log = temporary.path().join("crab.log");
    let mut crab_arguments = if let Some(experimental) = &args.exp {
        vec![
            "fit".into(),
            "--pdb".into(),
            args.pdb.display().to_string(),
            "--exp".into(),
            experimental.display().to_string(),
            "--method".into(),
            "multipole".into(),
            "--hydration".into(),
            "adaptive-grid".into(),
            "--background".into(),
            match args.background {
                Background::Fit => "fit".into(),
                Background::Zero => "zero".into(),
            },
            "--hydration-spacing".into(),
            args.hydration_spacing.to_string(),
            "--q-sampling-stride".into(),
            args.q_sampling_stride.to_string(),
            "--qmax".into(),
            q_max.to_string(),
            "--output".into(),
            crab_out.display().to_string(),
        ]
    } else {
        let q_step = q.get(1).map_or(0.005, |second| second - q[0]);
        vec![
            "compute".into(),
            "--pdb".into(),
            args.pdb.display().to_string(),
            "--method".into(),
            "multipole".into(),
            "--qmin".into(),
            q_min.to_string(),
            "--qmax".into(),
            q_max.to_string(),
            "--qstep".into(),
            q_step.to_string(),
            "--output".into(),
            crab_out.display().to_string(),
        ]
    };
    if args.include_hetatm {
        crab_arguments.push("--include-hetatm".into());
    }
    crab_arguments.extend(["--occupancy".into(), "full".into()]);

    let mut crab_times = Vec::new();
    for run in 0..(args.repetitions + 2) {
        let log = File::create(&crab_log)?;
        let start = Instant::now();
        let child = Command::new(&crab_bin)
            .args(&crab_arguments)
            .stdout(Stdio::null())
            .stderr(Stdio::from(log))
            .spawn()
            .context("launch crabSAXS")?;
        let status = run_with_timeout(child, Duration::from_secs(args.timeout_seconds))?;
        if !status.success() {
            bail!("crabSAXS exited with {status}");
        }
        if run >= 2 {
            crab_times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
    }

    let structure = Structure::from_pdb_file_with_options(
        &args.pdb,
        ParseOptions {
            include_hetatm: args.include_hetatm,
            occupancy_policy: OccupancyPolicy::SelectedConformerFull,
            ..Default::default()
        },
    )?;
    let experimental = args
        .exp
        .as_ref()
        .map(ExperimentalCurve::from_dat_file)
        .transpose()?
        .map(|curve| curve.through_qmax(q_max));
    let crab_hydration_points = Some(
        crabsaxs::solvent::build_adaptive_hydration_grid(
            &structure,
            &crabsaxs::AdaptiveHydrationOptions {
                spacing_angstrom: args.hydration_spacing,
                ..Default::default()
            },
        )?
        .sites
        .len(),
    );
    let (crab, crab_chi2, crab_fit_params, crab_q) = if let Some(data) = &experimental {
        validate_q_grid(&q, &data.q)?;
        let fit = fit_to_experimental_with_options(
            &structure,
            data,
            FitParams::default(),
            FitOptions {
                method: ScatteringMethod::Multipole,
                background_mode: args.background.into(),
                // Published hydration contrast range: -15..33.4 e/nm^3,
                // normalized by the 334 e/nm^3 bulk-water density.
                c2_bounds: (-0.045, 0.1),
                q_sampling_stride: args.q_sampling_stride,
                solvent: crabsaxs::solvent::SolventParams {
                    hydration_model: crabsaxs::solvent::HydrationModel::AdaptiveGrid,
                    hydration_grid_spacing_angstrom: args.hydration_spacing,
                    ..Default::default()
                },
                ..Default::default()
            },
        )?;
        (
            fit.fitted_curve,
            Some(fit.chi2),
            Some(fit.params),
            data.q.clone(),
        )
    } else {
        let curve = compute_curve(
            &structure,
            &q,
            &ComputeOptions {
                method: ScatteringMethod::Multipole,
                ..Default::default()
            },
        )?;
        (curve.intensity, None, None, curve.q)
    };
    validate_q_grid(&q, &crab_q)?;
    let component_skip_reason = (pepsi_components.is_some() && structure.atom_count() > 5_000)
        .then(|| "partial-profile audit skipped above 5000 atoms to bound audit memory".into());
    let component_parity = if structure.atom_count() <= 5_000 {
        if let Some(pepsi_basis) = pepsi_components {
            let centre = structure
                .atoms
                .iter()
                .fold(nalgebra::Vector3::zeros(), |sum, atom| sum + atom.pos)
                / structure.atom_count().max(1) as f64;
            let maximum_radius = structure
                .atoms
                .iter()
                .map(|atom| (atom.pos - centre).norm())
                .fold(0.0_f64, f64::max);
            let l_max = ((q_max * maximum_radius).ceil() as usize + 8).clamp(8, 64);
            let crab_basis = crabsaxs::spherical::multipole_solvent_basis(
                &structure,
                &q,
                &crabsaxs::spherical::MultipoleParams {
                    l_max,
                    hydrogen_mode: crabsaxs::HydrogenMode::Implicit,
                },
                &crabsaxs::SolventParams {
                    hydration_model: crabsaxs::HydrationModel::AdaptiveGrid,
                    hydration_grid_spacing_angstrom: args.hydration_spacing,
                    ..Default::default()
                },
            )?;
            Some(ComponentParity {
                atomic: curve_parity(&pepsi_basis[0], &crab_basis[0]),
                excluded_volume: curve_parity(&pepsi_basis[1], &crab_basis[1]),
                hydration: curve_parity(&pepsi_basis[2], &crab_basis[2]),
                atomic_excluded_cross: curve_parity(&pepsi_basis[3], &crab_basis[3]),
                atomic_hydration_cross: curve_parity(&pepsi_basis[4], &crab_basis[4]),
                excluded_hydration_cross: curve_parity(&pepsi_basis[5], &crab_basis[5]),
            })
        } else {
            None
        }
    } else {
        None
    };

    // Fitted curves already include their independently optimized scale and
    // must not receive an extra alignment. Calculation-only curves have an
    // arbitrary overall scale, so align them by one least-squares scalar.
    let comparison_scale = if experimental.is_some() {
        1.0
    } else {
        let dot = reference
            .iter()
            .zip(&crab)
            .map(|(left, right)| left * right)
            .sum::<f64>();
        let denominator = crab.iter().map(|value| value * value).sum::<f64>();
        if denominator > 0.0 {
            dot / denominator
        } else {
            1.0
        }
    };
    let squared_error = reference
        .iter()
        .zip(&crab)
        .map(|(left, right)| (comparison_scale * right - left).powi(2))
        .sum::<f64>();
    let rmse = (squared_error / reference.len() as f64).sqrt();
    let range = reference.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - reference.iter().copied().fold(f64::INFINITY, f64::min);
    let reference_l2 = reference.iter().map(|value| value * value).sum::<f64>();
    let relative_floor = (range * 1e-8).max(1e-15);
    let mut max_relative_error = 0.0_f64;
    let mut max_relative_error_q = q[0];
    let mut relative_sum = 0.0;
    for ((&q_value, &left), &right) in q.iter().zip(&reference).zip(&crab) {
        let relative = (comparison_scale * right - left).abs() / left.abs().max(relative_floor);
        relative_sum += relative;
        if relative > max_relative_error {
            max_relative_error = relative;
            max_relative_error_q = q_value;
        }
    }
    let normalized_nrmse = rmse / range.max(1e-15);
    let relative_l2_rmse = (squared_error / reference_l2.max(1e-30)).sqrt();
    let mean_absolute_relative_error = relative_sum / reference.len() as f64;
    let pepsi_chi2 = experimental.as_ref().map(|data| {
        let degrees_of_freedom = data.len().saturating_sub(1).max(1) as f64;
        data.intensity
            .iter()
            .zip(&data.sigma)
            .zip(&reference)
            .map(|((&observed, &sigma), &fitted)| ((fitted - observed) / sigma).powi(2))
            .sum::<f64>()
            / degrees_of_freedom
    });

    let pepsi_median_ms = median(&pepsi_times);
    let crab_median_ms = median(&crab_times);
    let speedup = pepsi_median_ms / crab_median_ms;
    let crab_fit_radius_angstrom =
        crab_fit_params.map(|params| params.c1 * crabsaxs::solvent::MEAN_ATOMIC_RADIUS_ANGSTROM);
    let crab_hydration_contrast_e_per_a3 = crab_fit_params.map(|params| {
        params.c2 * crabsaxs::solvent::SolventParams::default().bulk_density_e_per_a3
    });
    let accuracy_gate_passed = normalized_nrmse <= 0.01;
    let fit_quality_gate_passed = pepsi_chi2
        .zip(crab_chi2)
        .is_none_or(|(pepsi, crab)| crab <= pepsi * 1.1);
    let speed_gate_passed = crab_median_ms <= pepsi_median_ms;
    let component_gate_passed = component_parity.as_ref().is_none_or(|components| {
        [
            &components.atomic,
            &components.excluded_volume,
            &components.hydration,
            &components.atomic_excluded_cross,
            &components.atomic_hydration_cross,
            &components.excluded_hydration_cross,
        ]
        .into_iter()
        .all(|component| component.normalized_nrmse <= 0.01)
    });
    let target_gate_passed = accuracy_gate_passed
        && fit_quality_gate_passed
        && speed_gate_passed
        && component_gate_passed;

    let report = Report {
        report_schema: "crabsaxs-pepsi-parity-v3".into(),
        host_arch: std::env::consts::ARCH.into(),
        host_os: std::env::consts::OS.into(),
        rust_version: rust_version(),
        build_profile: if cfg!(debug_assertions) {
            "debug".into()
        } else {
            "release".into()
        },
        thread_count: rayon::current_num_threads(),
        pepsi_bin: pepsi_bin.display().to_string(),
        pepsi_sha256: sha256(&pepsi_bin)?,
        pepsi_file_description: file_description(&pepsi_bin),
        pepsi_execution_mode: execution_mode(&file_description(&pepsi_bin)),
        crab_bin: crab_bin.display().to_string(),
        crab_sha256: sha256(&crab_bin)?,
        crab_file_description: file_description(&crab_bin),
        pdb: args.pdb.display().to_string(),
        pdb_sha256: sha256(&args.pdb)?,
        experimental_sha256: args.exp.as_deref().map(sha256).transpose()?,
        atom_count: structure.atom_count(),
        include_non_water_hetatm: args.include_hetatm,
        occupancy_policy: "selected_conformer_full".into(),
        background: args.background,
        q_sampling_stride: args.q_sampling_stride,
        hydration_spacing_angstrom: Some(args.hydration_spacing),
        pepsi_hydration_points,
        crab_hydration_points,
        pepsi_fit_radius_angstrom: pepsi_fit_parameters.map(|parameters| parameters.0),
        pepsi_hydration_contrast_e_per_a3: pepsi_fit_parameters.map(|parameters| parameters.1),
        crab_fit_radius_angstrom,
        crab_hydration_contrast_e_per_a3,
        component_parity,
        component_skip_reason,
        pepsi_arguments,
        crab_arguments,
        q_points: q.len(),
        q_min,
        q_max,
        comparison_scale,
        normalized_nrmse,
        relative_l2_rmse,
        mean_absolute_relative_error,
        max_relative_error,
        max_relative_error_q,
        pepsi_median_ms,
        pepsi_mad_ms: median_absolute_deviation(&pepsi_times),
        crab_median_ms,
        crab_mad_ms: median_absolute_deviation(&crab_times),
        pepsi_chi2,
        crab_chi2,
        crab_fit_params,
        pepsi_over_crab_speedup: speedup,
        accuracy_gate_passed,
        fit_quality_gate_passed,
        speed_gate_passed,
        component_gate_passed,
        target_gate_passed,
        repetitions: args.repetitions,
        warmups: 2,
        timing_methodology: "Both executables timed as end-to-end subprocesses, including process startup, structure/data parsing, calculation or fit, and curve writing".into(),
    };
    serde_json::to_writer_pretty(File::create(&args.output)?, &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
