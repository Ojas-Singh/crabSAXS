//! CrabSAXS: theoretical SAXS profile computation and fitting from atomic structures.
//!
//! See `docs/MILESTONES.md` for the completed implementation checklist and
//! `README.md` for CLI, validation, and benchmark usage.

pub mod analysis;
pub mod ensemble;
pub mod error;
pub mod fit;
pub mod formfactor;
pub mod io;
pub mod metrics;
pub mod mixture;
pub mod plot;
pub mod reweight;
pub mod scattering;
pub mod solvent;
pub mod spherical;
pub mod structure;

pub use analysis::{
    compare_curves, ComparisonResult, ProfileCalculator, Quality, SaxsFitter, SaxsScorer,
    ScoreOptions, ScoreResult,
};
pub use ensemble::{EnsembleFitter, EnsembleOptions, EnsembleResult, WeightInterval};
pub use error::SaxsError;
pub use fit::{
    fit_to_experimental, fit_to_experimental_with_options, BackgroundMode, FitOptions, FitParams,
    FitResult, ScaleMode,
};
pub use io::ExperimentalCurve;
pub use metrics::{
    analyze_experimental, coordinate_features, features_from_fit, merge_external_analysis,
    parse_autorg_output, parse_gnom_output, read_autorg_output, read_gnom_output,
    CoordinateFeatures, ExperimentalAnalysis, ExternalSaxsAnalysis, PairDistribution, PrOptions,
    SaxsFeatures,
};
pub use mixture::{
    rank_and_marginalize, Assignment, CandidateCombination, CombinationAnalysis,
    CombinationPosterior, RankWeights,
};
pub use plot::{render_diagnostic_png, render_diagnostic_svg, DiagnosticPlot, PlotCurve};
pub use reweight::{
    fit_calculated_curve, reweight_curves, CurveFitResult, MaximumEntropyOptions,
    MaximumEntropyResult,
};
pub use scattering::{
    compute_curve, ComputeOptions, CurveMetadata, HydrogenMode, ScatteringCurve, ScatteringMethod,
};
pub use solvent::{AdaptiveHydrationOptions, HydrationModel, SolventGridOptions, SolventParams};
pub use structure::OccupancyPolicy;
pub use structure::{Atom, Element, Structure};
