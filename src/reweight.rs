//! Curve-level fitting and maximum-entropy ensemble reweighting.

use crate::error::{Result, SaxsError};
use crate::fit::{BackgroundMode, FitOptions, FitParams, ScaleMode};
use crate::io::ExperimentalCurve;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurveFitResult {
    pub params: FitParams,
    pub chi2: f64,
    pub reduced_chi2: f64,
    pub fitted_curve: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaximumEntropyOptions {
    pub kl_strength: f64,
    pub max_iterations: usize,
    pub step_size: f64,
    pub tolerance: f64,
    pub min_weight: f64,
    pub prior_weights: Option<Vec<f64>>,
    pub fit: FitOptions,
}

impl Default for MaximumEntropyOptions {
    fn default() -> Self {
        Self {
            kl_strength: 1.0,
            max_iterations: 500,
            step_size: 0.5,
            tolerance: 1.0e-8,
            min_weight: 1.0e-10,
            prior_weights: None,
            fit: FitOptions::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaximumEntropyResult {
    pub names: Vec<String>,
    pub weights: Vec<f64>,
    pub prior_weights: Vec<f64>,
    pub params: FitParams,
    pub chi2: f64,
    pub reduced_chi2: f64,
    pub fitted_curve: Vec<f64>,
    pub kl_divergence: f64,
    pub effective_sample_size: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Fit scale/background to an already calculated theoretical curve.
pub fn fit_calculated_curve(
    theory: &[f64],
    experimental: &ExperimentalCurve,
    options: &FitOptions,
) -> Result<CurveFitResult> {
    validate_curve(theory, experimental)?;
    let (scale, background, reduced_chi2) = solve_scale_background(theory, experimental, options);
    let fitted_curve = theory
        .iter()
        .map(|value| scale * value + background)
        .collect::<Vec<_>>();
    let dof = experimental.len().saturating_sub(1).max(1) as f64;
    Ok(CurveFitResult {
        params: FitParams {
            scale,
            background,
            c1: options.solvent.c1,
            c2: options.solvent.c2,
        },
        chi2: reduced_chi2 * dof,
        reduced_chi2,
        fitted_curve,
    })
}

/// Reweight calculated conformer curves while staying close to a supplied
/// prior, uniform by default.
pub fn reweight_curves(
    names: Vec<String>,
    curves: &[Vec<f64>],
    experimental: &ExperimentalCurve,
    options: MaximumEntropyOptions,
) -> Result<MaximumEntropyResult> {
    validate_ensemble(&names, curves, experimental, &options)?;
    let prior = normalized_prior(options.prior_weights.as_deref(), curves.len())?;
    let mut weights = apply_floor(prior.clone(), options.min_weight);
    let mut converged = false;
    let mut iterations = 0;
    for iteration in 0..options.max_iterations {
        iterations = iteration + 1;
        let fit = fit_calculated_curve(&mix_curves(curves, &weights), experimental, &options.fit)?;
        let gradient = objective_gradient(
            curves,
            experimental,
            &weights,
            &prior,
            &fit,
            options.kl_strength,
        );
        let max_gradient = gradient
            .iter()
            .map(|value| value.abs())
            .fold(0.0, f64::max)
            .max(1.0e-12);
        let mut step = options.step_size / max_gradient;
        let objective = fit.reduced_chi2 + options.kl_strength * kl_divergence(&weights, &prior);
        let mut accepted = None;
        for _ in 0..14 {
            let proposal = mirror_step(&weights, &gradient, step, options.min_weight);
            let proposal_fit =
                fit_calculated_curve(&mix_curves(curves, &proposal), experimental, &options.fit)?;
            let proposal_objective =
                proposal_fit.reduced_chi2 + options.kl_strength * kl_divergence(&proposal, &prior);
            if proposal_objective <= objective + 1.0e-12 {
                accepted = Some(proposal);
                break;
            }
            step *= 0.5;
        }
        let Some(proposal) = accepted else {
            break;
        };
        let delta = proposal
            .iter()
            .zip(&weights)
            .map(|(left, right)| (left - right).abs())
            .sum::<f64>();
        weights = proposal;
        if delta <= options.tolerance {
            converged = true;
            break;
        }
    }
    let fit = fit_calculated_curve(&mix_curves(curves, &weights), experimental, &options.fit)?;
    Ok(MaximumEntropyResult {
        names,
        weights: weights.clone(),
        prior_weights: prior.clone(),
        params: fit.params,
        chi2: fit.chi2,
        reduced_chi2: fit.reduced_chi2,
        fitted_curve: fit.fitted_curve,
        kl_divergence: kl_divergence(&weights, &prior),
        effective_sample_size: 1.0 / weights.iter().map(|weight| weight * weight).sum::<f64>(),
        iterations,
        converged,
    })
}

fn validate_curve(theory: &[f64], experimental: &ExperimentalCurve) -> Result<()> {
    if theory.len() != experimental.len()
        || theory.is_empty()
        || theory.iter().any(|value| !value.is_finite())
    {
        return Err(SaxsError::InvalidInput(
            "calculated and experimental curves must have equal non-zero finite lengths".into(),
        ));
    }
    Ok(())
}

fn validate_ensemble(
    names: &[String],
    curves: &[Vec<f64>],
    experimental: &ExperimentalCurve,
    options: &MaximumEntropyOptions,
) -> Result<()> {
    if curves.is_empty() || names.len() != curves.len() {
        return Err(SaxsError::InvalidInput(
            "reweighting names and curves must have equal non-zero lengths".into(),
        ));
    }
    if curves.iter().any(|curve| curve.len() != experimental.len()) {
        return Err(SaxsError::InvalidInput(
            "all reweighting curves must match the experimental q grid".into(),
        ));
    }
    if !options.kl_strength.is_finite()
        || options.kl_strength < 0.0
        || options.max_iterations == 0
        || !options.step_size.is_finite()
        || options.step_size <= 0.0
        || !options.tolerance.is_finite()
        || options.tolerance <= 0.0
        || !options.min_weight.is_finite()
        || options.min_weight < 0.0
        || options.min_weight * curves.len() as f64 >= 1.0
    {
        return Err(SaxsError::InvalidInput(
            "maximum-entropy options are invalid".into(),
        ));
    }
    if curves
        .iter()
        .flat_map(|curve| curve.iter())
        .any(|value| !value.is_finite())
    {
        return Err(SaxsError::InvalidInput(
            "reweighting curves must contain finite values".into(),
        ));
    }
    Ok(())
}

fn normalized_prior(prior: Option<&[f64]>, count: usize) -> Result<Vec<f64>> {
    let mut values = prior
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| vec![1.0; count]);
    if values.len() != count
        || values
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(SaxsError::InvalidInput(
            "maximum-entropy prior weights must be positive and match the ensemble".into(),
        ));
    }
    let sum = values.iter().sum::<f64>();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(SaxsError::InvalidInput(
            "maximum-entropy prior weights have zero total mass".into(),
        ));
    }
    for value in &mut values {
        *value /= sum;
    }
    Ok(values)
}

fn solve_scale_background(
    theory: &[f64],
    data: &ExperimentalCurve,
    options: &FitOptions,
) -> (f64, f64, f64) {
    let mut sw = 0.0;
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for ((&model, &observed), &sigma) in theory.iter().zip(&data.intensity).zip(&data.sigma) {
        let weight = 1.0 / (sigma * sigma);
        sw += weight;
        sx += weight * model;
        sy += weight * observed;
        sxx += weight * model * model;
        sxy += weight * model * observed;
    }
    let fitted_scale = |background: f64| {
        if sxx > 0.0 {
            ((sxy - background * sx) / sxx).max(0.0)
        } else {
            0.0
        }
    };
    let fitted_background = |scale: f64| {
        if sw > 0.0 {
            (sy - scale * sx) / sw
        } else {
            0.0
        }
    };
    let (scale, background) = match (options.scale_mode, options.background_mode) {
        (ScaleMode::Fixed(scale), BackgroundMode::Fixed(background)) => {
            (scale.max(0.0), background)
        }
        (ScaleMode::Fixed(scale), BackgroundMode::FixedZero) => (scale.max(0.0), 0.0),
        (ScaleMode::Fixed(scale), BackgroundMode::Fit) => {
            let scale = scale.max(0.0);
            (scale, fitted_background(scale))
        }
        (ScaleMode::Fit, BackgroundMode::Fixed(background)) => {
            (fitted_scale(background), background)
        }
        (ScaleMode::Fit, BackgroundMode::FixedZero) => (fitted_scale(0.0), 0.0),
        (ScaleMode::Fit, BackgroundMode::Fit) => {
            let determinant = sxx * sw - sx * sx;
            let scale = if determinant.abs() > 1.0e-20 {
                ((sxy * sw - sx * sy) / determinant).max(0.0)
            } else {
                0.0
            };
            (scale, fitted_background(scale))
        }
    };
    let dof = data.len().saturating_sub(1).max(1) as f64;
    let chi2 = theory
        .iter()
        .zip(&data.intensity)
        .zip(&data.sigma)
        .map(|((&model, &observed), &sigma)| {
            ((scale * model + background - observed) / sigma).powi(2)
        })
        .sum::<f64>();
    (scale, background, chi2 / dof)
}

fn objective_gradient(
    curves: &[Vec<f64>],
    data: &ExperimentalCurve,
    weights: &[f64],
    prior: &[f64],
    fit: &CurveFitResult,
    kl_strength: f64,
) -> Vec<f64> {
    let dof = data.len().saturating_sub(1).max(1) as f64;
    curves
        .iter()
        .enumerate()
        .map(|(index, curve)| {
            let data_gradient = curve
                .iter()
                .zip(&data.intensity)
                .zip(&data.sigma)
                .zip(&fit.fitted_curve)
                .map(|(((&model, &observed), &sigma), &fitted)| {
                    2.0 * (fitted - observed) * fit.params.scale * model / (sigma * sigma) / dof
                })
                .sum::<f64>();
            data_gradient + kl_strength * ((weights[index] / prior[index]).ln() + 1.0)
        })
        .collect()
}

fn mirror_step(weights: &[f64], gradient: &[f64], step: f64, floor: f64) -> Vec<f64> {
    let log_weights = weights
        .iter()
        .zip(gradient)
        .map(|(&weight, &gradient)| weight.max(1.0e-300).ln() - step * gradient)
        .collect::<Vec<_>>();
    let maximum = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut proposal = log_weights
        .iter()
        .map(|value| (*value - maximum).exp())
        .collect::<Vec<_>>();
    let sum = proposal.iter().sum::<f64>().max(1.0e-300);
    for value in &mut proposal {
        *value /= sum;
    }
    apply_floor(proposal, floor)
}

fn apply_floor(mut weights: Vec<f64>, floor: f64) -> Vec<f64> {
    if floor <= 0.0 {
        let sum = weights.iter().sum::<f64>().max(1.0e-300);
        for value in &mut weights {
            *value /= sum;
        }
        return weights;
    }
    let count = weights.len() as f64;
    let residual = 1.0 - floor * count;
    let sum = weights.iter().sum::<f64>().max(1.0e-300);
    for value in &mut weights {
        *value = floor + residual * *value / sum;
    }
    weights
}

fn kl_divergence(weights: &[f64], prior: &[f64]) -> f64 {
    weights
        .iter()
        .zip(prior)
        .map(|(&weight, &prior)| weight * (weight / prior).ln())
        .sum()
}

fn mix_curves(curves: &[Vec<f64>], weights: &[f64]) -> Vec<f64> {
    let points = curves.first().map_or(0, Vec::len);
    (0..points)
        .map(|point| {
            curves
                .iter()
                .zip(weights)
                .map(|(curve, weight)| curve[point] * weight)
                .sum()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> ExperimentalCurve {
        ExperimentalCurve {
            q: vec![0.01, 0.02],
            intensity: vec![0.2, 0.8],
            sigma: vec![0.01, 0.01],
            has_errors: true,
        }
    }

    #[test]
    fn curve_fit_respects_fixed_scale_and_background() {
        let mut options = FitOptions::default();
        options.scale_mode = ScaleMode::Fixed(1.0);
        options.background_mode = BackgroundMode::FixedZero;
        let fit = fit_calculated_curve(&[0.2, 0.8], &data(), &options).unwrap();
        assert!(fit.chi2.abs() < 1.0e-12);
        assert_eq!(fit.params.scale, 1.0);
    }

    #[test]
    fn maximum_entropy_moves_toward_the_experimental_curve() {
        let mut options = MaximumEntropyOptions::default();
        options.kl_strength = 0.01;
        options.fit.scale_mode = ScaleMode::Fixed(1.0);
        options.fit.background_mode = BackgroundMode::FixedZero;
        let result = reweight_curves(
            vec!["left".into(), "right".into()],
            &[vec![1.0, 0.0], vec![0.0, 1.0]],
            &data(),
            options,
        )
        .unwrap();
        assert!((result.weights.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);
        assert!(result.weights[1] > result.weights[0]);
        assert!(
            (result.effective_sample_size
                - 1.0 / result.weights.iter().map(|w| w * w).sum::<f64>())
            .abs()
                < 1.0e-12
        );
    }
}
