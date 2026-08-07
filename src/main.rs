use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use crabsaxs::analysis::{
    compare_curves, ProfileCalculator, Quality, SaxsFitter, SaxsScorer, ScoreOptions,
};
use crabsaxs::ensemble::{EnsembleFitter, EnsembleOptions};
use crabsaxs::fit::{BackgroundMode, FitOptions, ScaleMode};
use crabsaxs::io::{write_curve_csv, write_curve_dat, ExperimentalCurve};
use crabsaxs::scattering::{ComputeOptions, HydrogenMode, MultipoleOptions, ScatteringMethod};
use crabsaxs::solvent::{HydrationModel, SolventParams};
use crabsaxs::structure::{AltlocPolicy, OccupancyPolicy, ParseOptions, Structure};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "crabsaxs",
    version,
    about = "Fast, reproducible SAXS calculation and fitting"
)]
struct Cli {
    #[command(flatten)]
    global: GlobalArgs,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Args, Debug, Clone)]
struct GlobalArgs {
    #[arg(short = 'q', long, global = true)]
    quiet: bool,
    #[arg(short = 'v', long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,
    #[arg(short = 'j', long, default_value_t = 0, global = true)]
    threads: usize,
    #[arg(long, default_value_t = 0xC0FFEE, global = true)]
    seed: u64,
    #[arg(long, value_enum, default_value_t = OutputFormat::Text, global = true)]
    format: OutputFormat,
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[arg(long, global = true)]
    write_config: Option<PathBuf>,
    #[arg(long, global = true)]
    no_progress: bool,
    #[arg(long, value_enum, default_value_t = ColorMode::Auto, global = true)]
    color: ColorMode,
    #[arg(long, global = true)]
    force: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum OutputFormat {
    Text,
    Json,
    Csv,
    Dat,
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ColorMode {
    Auto,
    Always,
    Never,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Method {
    Auto,
    Debye,
    Multipole,
}
impl From<Method> for ScatteringMethod {
    fn from(value: Method) -> Self {
        match value {
            Method::Auto => Self::Auto,
            Method::Debye => Self::Debye,
            Method::Multipole => Self::Multipole,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Hydrogens {
    Implicit,
    Explicit,
}
impl From<Hydrogens> for HydrogenMode {
    fn from(value: Hydrogens) -> Self {
        match value {
            Hydrogens::Implicit => Self::Implicit,
            Hydrogens::Explicit => Self::Explicit,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum HydrationChoice {
    Auto,
    On,
    Off,
    Analytic,
    Grid,
    AdaptiveGrid,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum GlycanChoice {
    Include,
    Exclude,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum AltlocChoice {
    Highest,
    First,
    All,
}
impl From<AltlocChoice> for AltlocPolicy {
    fn from(value: AltlocChoice) -> Self {
        match value {
            AltlocChoice::Highest => Self::Highest,
            AltlocChoice::First => Self::First,
            AltlocChoice::All => Self::All,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum BackgroundChoice {
    Auto,
    Zero,
    Fixed(f64),
}
fn parse_background(value: &str) -> std::result::Result<BackgroundChoice, String> {
    if value.eq_ignore_ascii_case("auto") {
        Ok(BackgroundChoice::Auto)
    } else if value.eq_ignore_ascii_case("zero") {
        Ok(BackgroundChoice::Zero)
    } else {
        value
            .parse::<f64>()
            .map(BackgroundChoice::Fixed)
            .map_err(|_| "expected auto, zero, or a finite number".to_string())
    }
}

#[derive(Subcommand)]
enum Commands {
    Calc(CalcArgs),
    Fit(FitArgs),
    Score(ScoreArgs),
    Ensemble(EnsembleArgs),
    Batch(BatchArgs),
    Inspect(InspectArgs),
    Compare(CompareArgs),
    Info,
}

#[derive(Args)]
struct StructureArgs {
    /// PDB or mmCIF structure.
    structure: PathBuf,
    #[arg(long, default_value_t = 1)]
    model: usize,
    #[arg(long, value_delimiter = ',')]
    chain: Vec<String>,
    #[arg(long, value_enum, default_value_t = AltlocChoice::Highest)]
    altloc: AltlocChoice,
    #[arg(long, value_enum, default_value_t = GlycanChoice::Include)]
    glycans: GlycanChoice,
    #[arg(long)]
    include_hetatm: bool,
    #[arg(long)]
    include_waters: bool,
    #[arg(long, value_enum, default_value_t = Hydrogens::Implicit)]
    hydrogens: Hydrogens,
}

#[derive(Args)]
struct GridArgs {
    #[arg(long, default_value_t = 0.0, alias = "qmin")]
    q_min: f64,
    #[arg(long, default_value_t = 0.5, alias = "qmax")]
    q_max: f64,
    #[arg(long)]
    q_points: Option<usize>,
    #[arg(long, default_value_t = 0.005, alias = "qstep")]
    q_step: f64,
}

#[derive(Args)]
struct SolventArgs {
    #[arg(long, value_enum, default_value_t = HydrationChoice::Auto)]
    hydration: HydrationChoice,
    #[arg(long, default_value_t = 1.0)]
    c1: f64,
    #[arg(long, default_value_t = 0.0)]
    c2: f64,
    #[arg(long)]
    hydration_density: Option<f64>,
    #[arg(long)]
    hydration_thickness: Option<f64>,
}

#[derive(Args)]
struct CalcArgs {
    #[command(flatten)]
    structure: StructureArgs,
    #[command(flatten)]
    grid: GridArgs,
    #[command(flatten)]
    solvent: SolventArgs,
    #[arg(long, value_enum, default_value_t = Method::Auto)]
    method: Method,
    #[arg(long, value_enum, default_value_t = Quality::Balanced)]
    quality: Quality,
    #[arg(long)]
    l_max: Option<usize>,
    #[arg(short, long)]
    output: Option<PathBuf>,
}

#[derive(Args)]
struct FitArgs {
    #[command(flatten)]
    structure: StructureArgs,
    #[arg(value_name = "EXPERIMENTAL")]
    experimental: PathBuf,
    #[arg(long, value_enum, default_value_t = Method::Auto)]
    method: Method,
    #[arg(long, value_enum, default_value_t = Quality::Balanced)]
    quality: Quality,
    #[command(flatten)]
    solvent: SolventArgs,
    #[arg(long, default_value_t = 0.5)]
    q_max: f64,
    #[arg(long, default_value_t = 20)]
    q_sampling_stride: usize,
    #[arg(long, value_parser = parse_mode)]
    scale: Option<ScalarMode>,
    #[arg(long, value_parser = parse_background, default_value = "auto")]
    background: BackgroundChoice,
    #[arg(long)]
    fit_background: bool,
    #[arg(long)]
    constant_background: Option<f64>,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Clone, Copy)]
enum ScalarMode {
    Auto,
    Fixed(f64),
}
fn parse_mode(value: &str) -> std::result::Result<ScalarMode, String> {
    if value.eq_ignore_ascii_case("auto") {
        Ok(ScalarMode::Auto)
    } else {
        value
            .parse::<f64>()
            .map(ScalarMode::Fixed)
            .map_err(|_| "expected auto or a finite number".into())
    }
}

#[derive(Args)]
struct ScoreArgs {
    #[command(flatten)]
    structure: StructureArgs,
    experimental: PathBuf,
    #[arg(long, value_enum, default_value_t = Quality::Fast)]
    quality: Quality,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Args)]
struct EnsembleArgs {
    #[arg(required = true)]
    structures: Vec<String>,
    #[arg(long)]
    experiment: PathBuf,
    #[arg(long, value_enum, default_value_t = Method::Auto)]
    method: Method,
    #[arg(long, value_enum, default_value_t = Quality::Balanced)]
    quality: Quality,
    #[arg(long)]
    fit_weights: bool,
    #[arg(long)]
    uniform: bool,
    #[arg(long, default_value_t = 0.0)]
    regularization: f64,
    #[arg(long)]
    max_conformers: Option<usize>,
    #[arg(long, default_value_t = 0.0)]
    min_weight: f64,
    #[arg(long, default_value_t = 0)]
    bootstrap: usize,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Args)]
struct BatchArgs {
    #[arg(required = true)]
    structures: Vec<String>,
    #[arg(long)]
    experiment: Option<PathBuf>,
    #[arg(long)]
    recursive: bool,
    #[arg(long)]
    pattern: Option<String>,
    #[arg(long, value_enum, default_value_t = Quality::Fast)]
    quality: Quality,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Args)]
struct InspectArgs {
    input: PathBuf,
    #[arg(long)]
    data: bool,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Args)]
struct CompareArgs {
    calculated: PathBuf,
    experimental: PathBuf,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Serialize, Deserialize)]
struct FileConfig {
    threads: Option<usize>,
    seed: Option<u64>,
    quiet: Option<bool>,
    format: Option<OutputFormat>,
}

#[derive(Serialize)]
struct FitReport<'a> {
    version: &'static str,
    structure: String,
    experimental: String,
    q_unit: &'static str,
    has_errors: bool,
    quality: Quality,
    result: &'a crabsaxs::fit::FitResult,
}

#[derive(Serialize)]
struct CurveFile {
    version: &'static str,
    q: Vec<f64>,
    intensity: Vec<f64>,
}

fn main() -> Result<()> {
    let mut cli = Cli::parse();
    apply_config(&mut cli.global)?;
    if cli.global.threads > 0 {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(cli.global.threads)
            .build_global();
    }
    if let Some(path) = cli.global.write_config.clone() {
        write_config(&path, &cli.global)?;
    }
    if cli.global.verbose > 0 && !cli.global.quiet {
        eprintln!("CrabSAXS 0.1.0");
    }
    match cli.command {
        Commands::Calc(args) => run_calc(args, &cli.global),
        Commands::Fit(args) => run_fit(args, &cli.global),
        Commands::Score(args) => run_score(args, &cli.global),
        Commands::Ensemble(args) => run_ensemble(args, &cli.global),
        Commands::Batch(args) => run_batch(args, &cli.global),
        Commands::Inspect(args) => run_inspect(args, &cli.global),
        Commands::Compare(args) => run_compare(args, &cli.global),
        Commands::Info => run_info(&cli.global),
    }
}

fn apply_config(global: &mut GlobalArgs) -> Result<()> {
    let Some(path) = global.config.clone() else {
        return Ok(());
    };
    let file = fs::read_to_string(path)?;
    let config: FileConfig = toml::from_str(&file).context("parse TOML config")?;
    if global.threads == 0 {
        global.threads = config.threads.unwrap_or(0);
    }
    if global.seed == 0xC0FFEE {
        global.seed = config.seed.unwrap_or(global.seed);
    }
    if !global.quiet {
        global.quiet = config.quiet.unwrap_or(false);
    }
    if matches!(global.format, OutputFormat::Text) {
        global.format = config.format.unwrap_or(global.format);
    }
    Ok(())
}

fn write_config(path: &Path, global: &GlobalArgs) -> Result<()> {
    let config = FileConfig {
        threads: Some(global.threads),
        seed: Some(global.seed),
        quiet: Some(global.quiet),
        format: Some(global.format),
    };
    fs::write(path, toml::to_string_pretty(&config)?)?;
    Ok(())
}

fn parse_options(args: &StructureArgs) -> ParseOptions {
    ParseOptions {
        include_hetatm: args.include_hetatm,
        include_waters: args.include_waters,
        include_glycans: matches!(args.glycans, GlycanChoice::Include),
        model_index: args.model.saturating_sub(1),
        occupancy_policy: OccupancyPolicy::Reported,
        altloc_policy: args.altloc.into(),
        chains: (!args.chain.is_empty()).then_some(args.chain.clone()),
        ..Default::default()
    }
}

fn load_structure(args: &StructureArgs) -> Result<Structure> {
    Structure::from_pdb_file_with_options(&args.structure, parse_options(args))
        .context("parse structure")
}

fn q_grid(grid: &GridArgs) -> Result<Vec<f64>> {
    if !grid.q_min.is_finite() || !grid.q_max.is_finite() || grid.q_max < grid.q_min {
        anyhow::bail!("invalid q range");
    }
    if let Some(points) = grid.q_points {
        if points < 2 {
            anyhow::bail!("q-points must be at least two");
        }
        return Ok((0..points)
            .map(|index| {
                grid.q_min + (grid.q_max - grid.q_min) * index as f64 / (points - 1) as f64
            })
            .collect());
    }
    if !grid.q_step.is_finite() || grid.q_step <= 0.0 {
        anyhow::bail!("q-step must be positive");
    }
    let count = ((grid.q_max - grid.q_min) / grid.q_step).floor() as usize;
    let mut values = (0..=count)
        .map(|index| grid.q_min + index as f64 * grid.q_step)
        .collect::<Vec<_>>();
    if values.last().copied().unwrap_or(grid.q_min) < grid.q_max - 1e-12 {
        values.push(grid.q_max);
    }
    Ok(values)
}

fn hydration_model(choice: HydrationChoice) -> Option<HydrationModel> {
    match choice {
        HydrationChoice::Off => None,
        HydrationChoice::Analytic => Some(HydrationModel::Analytic),
        HydrationChoice::Grid => Some(HydrationModel::Grid),
        HydrationChoice::Auto | HydrationChoice::On | HydrationChoice::AdaptiveGrid => {
            Some(HydrationModel::AdaptiveGrid)
        }
    }
}

fn solvent(args: &SolventArgs) -> Option<SolventParams> {
    hydration_model(args.hydration).map(|model| SolventParams {
        c1: args.c1,
        c2: args.c2,
        hydration_model: model,
        hydration_grid_spacing_angstrom: match args.hydration {
            HydrationChoice::Auto => 4.0,
            _ => 4.0,
        },
        bulk_density_e_per_a3: args.hydration_density.unwrap_or(0.334),
        shell_thickness_angstrom: args.hydration_thickness.unwrap_or(3.0),
    })
}

fn compute_options(args: &CalcArgs) -> ComputeOptions {
    ComputeOptions {
        method: args.method.into(),
        hydrogen_mode: args.structure.hydrogens.into(),
        include_hetatm: args.structure.include_hetatm,
        multipole: MultipoleOptions { l_max: args.l_max },
        solvent: solvent(&args.solvent),
    }
}

fn run_calc(args: CalcArgs, global: &GlobalArgs) -> Result<()> {
    let structure = load_structure(&args.structure)?;
    let q = q_grid(&args.grid)?;
    let options = args.quality.compute_options(compute_options(&args));
    let calculator = ProfileCalculator::new(q, options)?;
    let curve = calculator.calculate(&structure)?;
    let output = args.output.unwrap_or_else(|| {
        let stem = args
            .structure
            .structure
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("structure");
        PathBuf::from(format!("{stem}_saxs.dat"))
    });
    write_curve_output(
        &output,
        global.format,
        &curve.q,
        &curve.intensity,
        global.force,
    )?;
    if !global.quiet {
        eprintln!("wrote {}", output.display());
    }
    Ok(())
}

fn fit_options(args: &FitArgs) -> FitOptions {
    let mut options = FitOptions {
        method: args.method.into(),
        hydrogen_mode: args.structure.hydrogens.into(),
        solvent: solvent(&args.solvent).unwrap_or_default(),
        background_mode: if let Some(value) = args.constant_background {
            BackgroundMode::Fixed(value)
        } else if args.fit_background || matches!(args.background, BackgroundChoice::Auto) {
            BackgroundMode::Fit
        } else {
            match args.background {
                BackgroundChoice::Zero => BackgroundMode::FixedZero,
                BackgroundChoice::Fixed(value) => BackgroundMode::Fixed(value),
                BackgroundChoice::Auto => BackgroundMode::Fit,
            }
        },
        scale_mode: match args.scale.unwrap_or(ScalarMode::Auto) {
            ScalarMode::Auto => ScaleMode::Fit,
            ScalarMode::Fixed(value) => ScaleMode::Fixed(value),
        },
        q_sampling_stride: args.q_sampling_stride,
        ..Default::default()
    };
    if matches!(
        args.solvent.hydration,
        HydrationChoice::AdaptiveGrid | HydrationChoice::Auto | HydrationChoice::On
    ) {
        options.c2_bounds = (-0.045, 0.1);
    }
    if matches!(args.solvent.hydration, HydrationChoice::Off) {
        options.c2_bounds = (-1e-12, 1e-12);
    }
    args.quality.fit_options(options)
}

fn run_fit(args: FitArgs, global: &GlobalArgs) -> Result<()> {
    let structure = load_structure(&args.structure)?;
    let data = ExperimentalCurve::from_dat_file(&args.experimental)?.through_qmax(args.q_max);
    if data.is_empty() {
        anyhow::bail!("experimental curve has no points at or below q-max");
    }
    let result = SaxsFitter::new(fit_options(&args)).fit(&structure, &data)?;
    let directory = args.output.unwrap_or_else(|| {
        let stem = args
            .structure
            .structure
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("structure");
        PathBuf::from(format!("{stem}-fit"))
    });
    let bundle = ReportBundle::new(&directory, global.force)?;
    let calculated = result.fitted_curve.clone();
    let residuals = data
        .intensity
        .iter()
        .zip(&calculated)
        .map(|(observed, model)| observed - model)
        .collect::<Vec<_>>();
    bundle.write_curve("fit.dat", &data.q, &calculated)?;
    bundle.write_curve_with_errors(
        "calculated.dat",
        &data.q,
        &data.intensity,
        &data.sigma,
        &calculated,
    )?;
    bundle.write_curve_with_errors(
        "residuals.dat",
        &data.q,
        &data.intensity,
        &data.sigma,
        &residuals,
    )?;
    bundle.write_json(
        "fit.json",
        &FitReport {
            version: "0.1.0",
            structure: args.structure.structure.display().to_string(),
            experimental: args.experimental.display().to_string(),
            q_unit: data.q_unit(),
            has_errors: data.has_errors,
            quality: args.quality,
            result: &result,
        },
    )?;
    bundle.write_text("report.txt", &format!("CrabSAXS 0.1.0\nchi2={:.6}\nreduced_chi2={:.6}\nscale={:.6e}\nbackground={:.6e}\nc1={:.6}\nc2={:.6}\niterations={}\nconverged={}\n", result.chi2 * data.len().saturating_sub(1).max(1) as f64, result.chi2, result.params.scale, result.params.background, result.params.c1, result.params.c2, result.iterations, result.converged))?;
    if !global.quiet {
        if matches!(global.format, OutputFormat::Json) {
            println!("{}", serde_json::to_string_pretty(&result)?);
        } else {
            println!(
                "chi2={:.6} reduced_chi2={:.6} scale={:.6e} background={:.6e}",
                result.chi2 * data.len().saturating_sub(1).max(1) as f64,
                result.chi2,
                result.params.scale,
                result.params.background
            );
        }
    }
    Ok(())
}

fn run_score(args: ScoreArgs, global: &GlobalArgs) -> Result<()> {
    let structure = load_structure(&args.structure)?;
    let data = ExperimentalCurve::from_dat_file(&args.experimental)?;
    let result = SaxsScorer::new(
        data,
        ScoreOptions {
            quality: args.quality,
            ..Default::default()
        },
    )?
    .score(&structure)?;
    if let Some(path) = args.output {
        write_json_or_text(&path, global.format, &result, global.force)?;
    }
    if matches!(global.format, OutputFormat::Json) || !global.quiet {
        if matches!(global.format, OutputFormat::Json) {
            println!("{}", serde_json::to_string_pretty(&result)?);
        } else {
            println!(
                "chi2={:.6} reduced_chi2={:.6}",
                result.chi2, result.reduced_chi2
            );
        }
    }
    Ok(())
}

fn expand_inputs(
    values: &[String],
    recursive: bool,
    pattern: Option<&str>,
) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for value in values {
        if value.contains('*') || value.contains('?') || value.contains('[') {
            for entry in glob::glob(value).context("invalid input glob")? {
                paths.push(entry?);
            }
        } else {
            paths.push(PathBuf::from(value));
        }
    }
    if recursive {
        let mut expanded = Vec::new();
        for root in paths {
            if root.is_dir() {
                collect_structures(&root, pattern.unwrap_or("*.pdb"), &mut expanded)?;
            } else {
                expanded.push(root);
            }
        }
        paths = expanded;
    }
    paths.sort();
    paths.dedup();
    if paths.is_empty() {
        anyhow::bail!("no input structures matched");
    }
    Ok(paths)
}

fn collect_structures(root: &Path, pattern: &str, output: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_structures(&path, pattern, output)?;
        } else if glob::Pattern::new(pattern)
            .map(|p| p.matches_path(&path))
            .unwrap_or(false)
        {
            output.push(path);
        }
    }
    Ok(())
}

fn run_ensemble(args: EnsembleArgs, global: &GlobalArgs) -> Result<()> {
    let paths = expand_inputs(&args.structures, false, None)?;
    let mut conformers = Vec::new();
    let mut names = Vec::new();
    for path in paths {
        let parse = ParseOptions {
            include_hetatm: true,
            include_waters: false,
            ..Default::default()
        };
        let models = Structure::from_pdb_file_models_with_options(&path, parse)?;
        if models.len() == 1 {
            conformers.push(models.into_iter().next().unwrap());
            names.push(path.display().to_string());
        } else {
            for (index, model) in models.into_iter().enumerate() {
                conformers.push(model);
                names.push(format!("{}#{}", path.display(), index + 1));
            }
        }
    }
    let data = ExperimentalCurve::from_dat_file(&args.experiment)?;
    let mut fit = FitOptions {
        method: args.method.into(),
        ..Default::default()
    };
    fit = args.quality.fit_options(fit);
    let options = EnsembleOptions {
        fit_weights: args.fit_weights || !args.uniform,
        uniform: args.uniform,
        regularization: args.regularization,
        max_conformers: args.max_conformers,
        min_weight: args.min_weight,
        bootstrap: args.bootstrap,
        seed: global.seed,
        fit,
    };
    let result = EnsembleFitter::new(conformers, names, data, options)?.fit()?;
    let output = args.output.unwrap_or_else(|| PathBuf::from("ensemble-fit"));
    let bundle = ReportBundle::new(&output, global.force)?;
    bundle.write_json("ensemble.json", &result)?;
    let mut populations = String::from("conformer,weight,lower,upper\n");
    for (name, interval) in result.conformers.iter().zip(&result.weight_intervals) {
        populations.push_str(&format!(
            "{name},{:.8},{:.8},{:.8}\n",
            interval.median, interval.lower, interval.upper
        ));
    }
    bundle.write_text("populations.csv", &populations)?;
    bundle.write_text(
        "report.txt",
        &format!(
            "CrabSAXS 0.1.0 ensemble\nreduced_chi2={:.6}\nconformers={}\n",
            result.reduced_chi2,
            result.conformers.len()
        ),
    )?;
    if !global.quiet {
        if matches!(global.format, OutputFormat::Json) {
            println!("{}", serde_json::to_string_pretty(&result)?);
        } else {
            println!("reduced_chi2={:.6}", result.reduced_chi2);
        }
    }
    Ok(())
}

fn run_batch(args: BatchArgs, global: &GlobalArgs) -> Result<()> {
    let paths = expand_inputs(&args.structures, args.recursive, args.pattern.as_deref())?;
    let mut rows = Vec::new();
    let experimental = args
        .experiment
        .as_ref()
        .map(ExperimentalCurve::from_dat_file)
        .transpose()?;
    for path in paths {
        let structure = Structure::from_pdb_file(&path)?;
        if let Some(data) = &experimental {
            let score = SaxsScorer::new(
                data.clone(),
                ScoreOptions {
                    quality: args.quality,
                    ..Default::default()
                },
            )?
            .score(&structure)?;
            rows.push((path.display().to_string(), score.chi2, score.reduced_chi2));
        } else {
            rows.push((path.display().to_string(), f64::NAN, f64::NAN));
        }
    }
    let output = args.output.unwrap_or_else(|| PathBuf::from("batch.csv"));
    if output.exists() && !global.force {
        anyhow::bail!("output exists: {} (use --force)", output.display());
    }
    let mut text = String::from("structure,chi2,reduced_chi2\n");
    for (path, chi2, reduced) in rows {
        text.push_str(&format!("{path},{chi2:.8},{reduced:.8}\n"));
    }
    fs::write(&output, text)?;
    if !global.quiet {
        eprintln!("wrote {}", output.display());
    }
    Ok(())
}

fn run_inspect(args: InspectArgs, global: &GlobalArgs) -> Result<()> {
    if args.data
        || args
            .input
            .extension()
            .is_some_and(|ext| ext == "dat" || ext == "txt" || ext == "csv")
    {
        let data = ExperimentalCurve::from_dat_file(&args.input)?;
        let report = serde_json::json!({"kind":"experimental","points":data.len(),"q_min":data.q[0],"q_max":data.q.last(),"errors":data.has_errors,"q_unit":data.q_unit()});
        emit_report(&report, &args.output, global)
    } else {
        let structure = Structure::from_pdb_file(&args.input)?;
        let atoms = structure.atoms.len();
        let chains = structure
            .atoms
            .iter()
            .map(|atom| atom.chain_id.clone())
            .collect::<std::collections::BTreeSet<_>>();
        let glycans = structure
            .atoms
            .iter()
            .filter(|atom| {
                crabsaxs::structure::residue_kind(&atom.residue_name)
                    == crabsaxs::structure::ResidueKind::Glycan
            })
            .count();
        let report = serde_json::json!({"kind":"structure","atoms":atoms,"chains":chains.len(),"glycan_atoms":glycans});
        emit_report(&report, &args.output, global)
    }
}

fn run_compare(args: CompareArgs, global: &GlobalArgs) -> Result<()> {
    let experimental = ExperimentalCurve::from_dat_file(&args.experimental)?;
    let calculated = read_curve(&args.calculated)?;
    let result = compare_curves(&experimental, &calculated)?;
    emit_serializable(&result, &args.output, global)
}

fn run_info(global: &GlobalArgs) -> Result<()> {
    let info = serde_json::json!({"name":"crabsaxs","version":"0.1.0","method":"Debye + spherical multipole","threads":rayon::current_num_threads(),"rust":"1.97"});
    if matches!(global.format, OutputFormat::Json) {
        println!("{}", serde_json::to_string_pretty(&info)?);
    } else {
        println!(
            "CrabSAXS 0.1.0\nthreads={}\nalgorithms=Debye,multipole",
            rayon::current_num_threads()
        );
    }
    Ok(())
}

fn read_curve(path: &Path) -> Result<Vec<f64>> {
    let mut values = Vec::new();
    for line in fs::read_to_string(path)?.lines() {
        let normalized = line.replace(',', " ");
        let fields = normalized.split_whitespace().collect::<Vec<_>>();
        if fields.len() >= 2 && fields[0].parse::<f64>().is_ok() {
            values.push(fields[1].parse::<f64>()?);
        }
    }
    Ok(values)
}

fn write_curve_output(
    path: &Path,
    format: OutputFormat,
    q: &[f64],
    intensity: &[f64],
    force: bool,
) -> Result<()> {
    if path.exists() && !force {
        anyhow::bail!("output exists: {} (use --force)", path.display());
    }
    match format {
        OutputFormat::Csv => write_curve_csv(path, q, intensity)?,
        OutputFormat::Json => fs::write(
            path,
            serde_json::to_string_pretty(&CurveFile {
                version: "0.1.0",
                q: q.to_vec(),
                intensity: intensity.to_vec(),
            })?,
        )?,
        _ => write_curve_dat(path, q, intensity)?,
    }
    Ok(())
}

fn write_json_or_text<T: Serialize>(
    path: &Path,
    format: OutputFormat,
    value: &T,
    force: bool,
) -> Result<()> {
    if path.exists() && !force {
        anyhow::bail!("output exists: {} (use --force)", path.display());
    }
    let _ = format;
    let content = serde_json::to_string_pretty(value)?;
    fs::write(path, content)?;
    Ok(())
}

fn emit_serializable<T: Serialize>(
    value: &T,
    output: &Option<PathBuf>,
    global: &GlobalArgs,
) -> Result<()> {
    let content = serde_json::to_string_pretty(value)?;
    if let Some(path) = output {
        if path.exists() && !global.force {
            anyhow::bail!("output exists: {} (use --force)", path.display());
        }
        fs::write(path, &content)?;
    }
    if !global.quiet {
        println!("{content}");
    }
    Ok(())
}

fn emit_report(
    value: &serde_json::Value,
    output: &Option<PathBuf>,
    global: &GlobalArgs,
) -> Result<()> {
    if let Some(path) = output {
        if path.exists() && !global.force {
            anyhow::bail!("output exists: {} (use --force)", path.display());
        }
        fs::write(path, serde_json::to_string_pretty(value)?)?;
    }
    if !global.quiet {
        if matches!(global.format, OutputFormat::Json) {
            println!("{}", serde_json::to_string_pretty(value)?);
        } else {
            println!("{}", value);
        }
    }
    Ok(())
}

struct ReportBundle {
    path: PathBuf,
}
impl ReportBundle {
    fn new(path: &Path, force: bool) -> Result<Self> {
        if path.exists() && !force {
            anyhow::bail!("output exists: {} (use --force)", path.display());
        }
        fs::create_dir_all(path)?;
        Ok(Self {
            path: path.to_path_buf(),
        })
    }
    fn write_text(&self, name: &str, content: &str) -> Result<()> {
        fs::write(self.path.join(name), content)?;
        Ok(())
    }
    fn write_json<T: Serialize>(&self, name: &str, value: &T) -> Result<()> {
        self.write_text(name, &serde_json::to_string_pretty(value)?)
    }
    fn write_curve(&self, name: &str, q: &[f64], intensity: &[f64]) -> Result<()> {
        Ok(write_curve_dat(self.path.join(name), q, intensity)?)
    }
    fn write_curve_with_errors(
        &self,
        name: &str,
        q: &[f64],
        observed: &[f64],
        sigma: &[f64],
        fitted: &[f64],
    ) -> Result<()> {
        let mut file = fs::File::create(self.path.join(name))?;
        writeln!(file, "# q I_exp sigma I_calc residual")?;
        for (((&q, &observed), &sigma), &fitted) in q.iter().zip(observed).zip(sigma).zip(fitted) {
            writeln!(
                file,
                "{q:.8e} {observed:.8e} {sigma:.8e} {fitted:.8e} {:.8e}",
                observed - fitted
            )?;
        }
        Ok(())
    }
}
