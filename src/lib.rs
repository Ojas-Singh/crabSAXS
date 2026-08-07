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
pub use scattering::{
    compute_curve, ComputeOptions, CurveMetadata, HydrogenMode, ScatteringCurve, ScatteringMethod,
};
pub use solvent::{AdaptiveHydrationOptions, HydrationModel, SolventGridOptions, SolventParams};
pub use structure::OccupancyPolicy;
pub use structure::{Atom, Element, Structure};
