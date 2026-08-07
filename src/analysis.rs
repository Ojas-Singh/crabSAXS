//! Reusable high-level calculation, scoring, and comparison APIs.

use crate::error::{Result, SaxsError};
use crate::fit::{fit_to_experimental_with_options, FitOptions, FitParams, FitResult};
use crate::io::ExperimentalCurve;
use crate::scattering::{compute_curve, ComputeOptions, ScatteringCurve};
use crate::structure::Structure;
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

/// User-facing accuracy preset. Numerical details remain centralized here so
/// the CLI and Rust callers use identical behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum, Default)]
#[serde(rename_all = "lowercase")]
pub enum Quality {
    Fast,
    #[default]
    Balanced,
    Accurate,
}

impl Quality {
    pub fn fit_options(self, mut options: FitOptions) -> FitOptions {
        match self {
            Self::Fast => {
                options.q_sampling_stride = options.q_sampling_stride.max(20);
                options.max_iterations = options.max_iterations.min(32);
                options.solvent.hydration_grid_spacing_angstrom = 8.0;
            }
            Self::Balanced => {
                options.q_sampling_stride = 20;
                options.max_iterations = options.max_iterations.max(64);
                options.solvent.hydration_grid_spacing_angstrom = 4.0;
            }
            Self::Accurate => {
                options.q_sampling_stride = 1;
                options.max_iterations = options.max_iterations.max(128);
                options.solvent.hydration_grid_spacing_angstrom = 3.0;
            }
        }
        options
    }

    pub fn compute_options(self, mut options: ComputeOptions) -> ComputeOptions {
        if matches!(self, Self::Fast) {
            if let Some(solvent) = options.solvent.as_mut() {
                solvent.hydration_grid_spacing_angstrom = 8.0;
            }
        } else if matches!(self, Self::Accurate) {
            if let Some(solvent) = options.solvent.as_mut() {
                solvent.hydration_grid_spacing_angstrom = 3.0;
            }
        }
        options
    }
}

/// Reusable profile calculator for repeated q grids and structures.
#[derive(Debug, Clone)]
pub struct ProfileCalculator {
    pub q: Vec<f64>,
    pub options: ComputeOptions,
}

/// Reusable structure/profile fitter.
#[derive(Debug, Clone)]
pub struct SaxsFitter {
    pub options: FitOptions,
}

impl SaxsFitter {
    pub fn new(options: FitOptions) -> Self {
        Self { options }
    }

    pub fn fit(
        &self,
        structure: &Structure,
        experimental: &ExperimentalCurve,
    ) -> Result<FitResult> {
        fit_to_experimental_with_options(
            structure,
            experimental,
            FitParams::default(),
            self.options,
        )
    }
}

impl ProfileCalculator {
    pub fn new(q: Vec<f64>, options: ComputeOptions) -> Result<Self> {
        if q.iter().any(|value| !value.is_finite() || *value < 0.0)
            || q.windows(2).any(|pair| pair[1] < pair[0])
        {
            return Err(SaxsError::InvalidInput(
                "q grid must be finite, non-negative, and non-decreasing".into(),
            ));
        }
        Ok(Self { q, options })
    }

    pub fn calculate(&self, structure: &Structure) -> Result<ScatteringCurve> {
        compute_curve(structure, &self.q, &self.options)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreOptions {
    pub fit: FitOptions,
    pub quality: Quality,
}

impl Default for ScoreOptions {
    fn default() -> Self {
        Self {
            fit: FitOptions::default(),
            quality: Quality::Fast,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreResult {
    pub chi2: f64,
    pub reduced_chi2: f64,
    pub scale: f64,
    pub background: f64,
    pub q_min: f64,
    pub q_max: f64,
    pub points: usize,
    pub has_errors: bool,
    pub q_unit: String,
    pub fit: FitResult,
}

/// Reusable scorer that parses/validates the experimental curve once.
#[derive(Debug, Clone)]
pub struct SaxsScorer {
    pub experimental: ExperimentalCurve,
    pub options: ScoreOptions,
}

impl SaxsScorer {
    pub fn new(experimental: ExperimentalCurve, options: ScoreOptions) -> Result<Self> {
        if experimental.is_empty() {
            return Err(SaxsError::InvalidInput(
                "experimental curve is empty".into(),
            ));
        }
        Ok(Self {
            experimental,
            options,
        })
    }

    pub fn score(&self, structure: &Structure) -> Result<ScoreResult> {
        let fit_options = self.options.quality.fit_options(self.options.fit);
        let fit = fit_to_experimental_with_options(
            structure,
            &self.experimental,
            FitParams::default(),
            fit_options,
        )?;
        let dof = self.experimental.len().saturating_sub(1).max(1) as f64;
        Ok(ScoreResult {
            chi2: fit.chi2 * dof,
            reduced_chi2: fit.chi2,
            scale: fit.params.scale,
            background: fit.params.background,
            q_min: self.experimental.q[0],
            q_max: *self.experimental.q.last().unwrap_or(&0.0),
            points: self.experimental.len(),
            has_errors: self.experimental.has_errors,
            q_unit: self.experimental.q_unit().into(),
            fit,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub chi2: f64,
    pub reduced_chi2: f64,
    pub pearson_correlation: f64,
    pub r_factor: f64,
    pub weighted_rmse: f64,
    pub residuals: Vec<f64>,
}

pub fn compare_curves(
    experimental: &ExperimentalCurve,
    calculated: &[f64],
) -> Result<ComparisonResult> {
    if experimental.len() != calculated.len() || experimental.is_empty() {
        return Err(SaxsError::InvalidInput(
            "experimental and calculated curves must have equal non-zero lengths".into(),
        ));
    }
    let residuals = experimental
        .intensity
        .iter()
        .zip(calculated)
        .map(|(observed, model)| model - observed)
        .collect::<Vec<_>>();
    let dof = experimental.len().saturating_sub(1).max(1) as f64;
    let chi_sum = residuals
        .iter()
        .zip(&experimental.sigma)
        .map(|(residual, sigma)| (residual / sigma).powi(2))
        .sum::<f64>();
    let mean_observed = experimental.intensity.iter().sum::<f64>() / experimental.len() as f64;
    let mean_model = calculated.iter().sum::<f64>() / calculated.len() as f64;
    let covariance = experimental
        .intensity
        .iter()
        .zip(calculated)
        .map(|(observed, model)| (observed - mean_observed) * (model - mean_model))
        .sum::<f64>();
    let observed_var = experimental
        .intensity
        .iter()
        .map(|value| (value - mean_observed).powi(2))
        .sum::<f64>();
    let model_var = calculated
        .iter()
        .map(|value| (value - mean_model).powi(2))
        .sum::<f64>();
    let pearson = if observed_var > 0.0 && model_var > 0.0 {
        covariance / (observed_var * model_var).sqrt()
    } else {
        0.0
    };
    let denominator = experimental
        .intensity
        .iter()
        .map(|value| value.abs())
        .sum::<f64>();
    let r_factor = if denominator > 0.0 {
        residuals.iter().map(|value| value.abs()).sum::<f64>() / denominator
    } else {
        0.0
    };
    Ok(ComparisonResult {
        chi2: chi_sum,
        reduced_chi2: chi_sum / dof,
        pearson_correlation: pearson,
        r_factor,
        weighted_rmse: (chi_sum / experimental.len() as f64).sqrt(),
        residuals,
    })
}
