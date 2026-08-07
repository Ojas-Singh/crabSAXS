//! Population fitting for incoherent SAXS ensembles.

use crate::error::{Result, SaxsError};
use crate::fit::{BackgroundMode, FitOptions, FitParams, ScaleMode};
use crate::io::ExperimentalCurve;
use crate::scattering::{compute_curve, ComputeOptions};
use crate::structure::Structure;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleOptions {
    pub fit_weights: bool,
    pub uniform: bool,
    pub regularization: f64,
    pub max_conformers: Option<usize>,
    pub min_weight: f64,
    pub bootstrap: usize,
    pub seed: u64,
    pub fit: FitOptions,
}

impl Default for EnsembleOptions {
    fn default() -> Self {
        Self {
            fit_weights: true,
            uniform: false,
            regularization: 0.0,
            max_conformers: None,
            min_weight: 0.0,
            bootstrap: 0,
            seed: 0xC0FFEE,
            fit: FitOptions::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightInterval {
    pub median: f64,
    pub lower: f64,
    pub upper: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleResult {
    pub conformers: Vec<String>,
    pub weights: Vec<f64>,
    pub weight_intervals: Vec<WeightInterval>,
    pub params: FitParams,
    pub chi2: f64,
    pub reduced_chi2: f64,
    pub fitted_curve: Vec<f64>,
    pub bootstrap_reduced_chi2: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
}

#[derive(Debug, Clone)]
pub struct EnsembleFitter {
    pub conformers: Vec<Structure>,
    pub names: Vec<String>,
    pub experimental: ExperimentalCurve,
    pub options: EnsembleOptions,
}

#[derive(Debug, Clone)]
struct CoreFit {
    weights: Vec<f64>,
    params: FitParams,
    chi2: f64,
    fitted: Vec<f64>,
    iterations: usize,
}

type CandidateFit = (f64, (f64, f64), Vec<f64>, Vec<f64>, f64, f64, usize);

impl EnsembleFitter {
    pub fn new(
        conformers: Vec<Structure>,
        names: Vec<String>,
        experimental: ExperimentalCurve,
        options: EnsembleOptions,
    ) -> Result<Self> {
        if conformers.is_empty()
            || conformers
                .iter()
                .any(|structure| structure.atoms.is_empty())
        {
            return Err(SaxsError::InvalidInput(
                "ensemble must contain non-empty conformers".into(),
            ));
        }
        if names.len() != conformers.len() {
            return Err(SaxsError::InvalidInput(
                "ensemble names and conformers differ in length".into(),
            ));
        }
        if experimental.is_empty()
            || experimental.q.len() != experimental.intensity.len()
            || experimental.q.len() != experimental.sigma.len()
        {
            return Err(SaxsError::InvalidInput(
                "ensemble requires a validated experimental curve".into(),
            ));
        }
        if !options.regularization.is_finite()
            || options.regularization < 0.0
            || !options.min_weight.is_finite()
            || !(0.0..=1.0).contains(&options.min_weight)
            || options.max_conformers == Some(0)
        {
            return Err(SaxsError::InvalidInput(
                "ensemble regularization and pruning options are invalid".into(),
            ));
        }
        Ok(Self {
            conformers,
            names,
            experimental,
            options,
        })
    }

    pub fn fit(&self) -> Result<EnsembleResult> {
        let mut selected = (0..self.conformers.len()).collect::<Vec<_>>();
        let mut core = self.fit_core(&selected, &self.experimental)?;
        let mut changed = true;
        while changed {
            changed = false;
            let mut keep = selected
                .iter()
                .enumerate()
                .filter_map(|(position, &index)| {
                    (core.weights[position] + 1e-12 >= self.options.min_weight).then_some(index)
                })
                .collect::<Vec<_>>();
            if keep.is_empty() {
                keep.push(
                    selected[core
                        .weights
                        .iter()
                        .enumerate()
                        .max_by(|left, right| left.1.total_cmp(right.1))
                        .map(|(i, _)| i)
                        .unwrap_or(0)],
                );
            }
            if let Some(maximum) = self.options.max_conformers {
                if keep.len() > maximum {
                    let mut ranked = selected
                        .iter()
                        .enumerate()
                        .map(|(position, &index)| (core.weights[position], index))
                        .collect::<Vec<_>>();
                    ranked.sort_by(|left, right| right.0.total_cmp(&left.0));
                    keep = ranked
                        .into_iter()
                        .take(maximum)
                        .map(|(_, index)| index)
                        .collect();
                }
            }
            keep.sort_unstable();
            if keep != selected {
                selected = keep;
                core = self.fit_core(&selected, &self.experimental)?;
                changed = true;
            }
        }

        let mut intervals = core
            .weights
            .iter()
            .map(|&weight| WeightInterval {
                median: weight,
                lower: weight,
                upper: weight,
            })
            .collect::<Vec<_>>();
        let mut bootstrap_chi2 = Vec::new();
        if self.options.bootstrap > 0 {
            let mut rng = ChaCha8Rng::seed_from_u64(self.options.seed);
            let residuals = self
                .experimental
                .intensity
                .iter()
                .zip(&core.fitted)
                .zip(&self.experimental.sigma)
                .map(|((&observed, &fitted), &sigma)| (observed - fitted) / sigma)
                .collect::<Vec<_>>();
            let mut samples = vec![Vec::with_capacity(self.options.bootstrap); selected.len()];
            for _ in 0..self.options.bootstrap {
                let mut synthetic = self.experimental.clone();
                for index in 0..synthetic.len() {
                    let residual = residuals[rng.gen_range(0..residuals.len())];
                    synthetic.intensity[index] =
                        core.fitted[index] + residual * synthetic.sigma[index];
                }
                let replicate = self.fit_core(&selected, &synthetic)?;
                for (position, weight) in replicate.weights.iter().enumerate() {
                    samples[position].push(*weight);
                }
                bootstrap_chi2.push(replicate.chi2);
            }
            for (position, sample) in samples.iter_mut().enumerate() {
                sample.sort_by(f64::total_cmp);
                intervals[position] = WeightInterval {
                    median: quantile(sample, 0.5),
                    lower: quantile(sample, 0.025),
                    upper: quantile(sample, 0.975),
                };
            }
        }
        Ok(EnsembleResult {
            conformers: selected
                .iter()
                .map(|&index| self.names[index].clone())
                .collect(),
            weights: core.weights,
            weight_intervals: intervals,
            params: core.params,
            chi2: core.chi2 * self.experimental.len().saturating_sub(1).max(1) as f64,
            reduced_chi2: core.chi2,
            fitted_curve: core.fitted,
            bootstrap_reduced_chi2: bootstrap_chi2,
            iterations: core.iterations,
            converged: true,
        })
    }

    fn fit_core(&self, selected: &[usize], data: &ExperimentalCurve) -> Result<CoreFit> {
        let fit_options = self.options.fit;
        let compute_options = ComputeOptions {
            method: fit_options.method,
            hydrogen_mode: fit_options.hydrogen_mode,
            include_hetatm: true,
            multipole: Default::default(),
            solvent: Some(fit_options.solvent),
        };
        let candidate_parameters = solvent_candidates(&fit_options);
        let curves_by_candidate = candidate_parameters
            .par_iter()
            .map(|&(c1, c2)| {
                let curves = selected
                    .iter()
                    .map(|&index| {
                        let mut options = compute_options.clone();
                        let mut solvent = fit_options.solvent;
                        solvent.c1 = c1;
                        solvent.c2 = c2;
                        options.solvent = Some(solvent);
                        compute_curve(&self.conformers[index], &data.q, &options)
                            .map(|curve| curve.intensity)
                    })
                    .collect::<Result<Vec<_>>>();
                curves.map(|curves| ((c1, c2), curves))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut best: Option<CandidateFit> = None;
        for ((c1, c2), curves) in curves_by_candidate {
            let (weights, scale, background, chi2, fitted, iterations) = optimize_weights(
                &curves,
                data,
                &fit_options,
                self.options.uniform || !self.options.fit_weights,
                self.options.regularization,
            );
            if best.as_ref().is_none_or(|current| chi2 < current.0) {
                best = Some((
                    chi2,
                    (c1, c2),
                    weights,
                    fitted,
                    scale,
                    background,
                    iterations,
                ));
            }
        }
        let (chi2, (c1, c2), weights, fitted, scale, background, iterations) =
            best.ok_or_else(|| {
                SaxsError::FitDidNotConverge("no ensemble candidate succeeded".into())
            })?;
        Ok(CoreFit {
            weights,
            params: FitParams {
                scale,
                background,
                c1,
                c2,
            },
            chi2,
            fitted,
            iterations,
        })
    }
}

fn solvent_candidates(options: &FitOptions) -> Vec<(f64, f64)> {
    let steps = 5usize;
    (0..steps)
        .flat_map(|i| {
            let c1 = options.c1_bounds.0
                + (options.c1_bounds.1 - options.c1_bounds.0) * i as f64 / (steps - 1) as f64;
            (0..steps).map(move |j| {
                let c2 = options.c2_bounds.0
                    + (options.c2_bounds.1 - options.c2_bounds.0) * j as f64 / (steps - 1) as f64;
                (c1, c2)
            })
        })
        .collect()
}

fn optimize_weights(
    curves: &[Vec<f64>],
    data: &ExperimentalCurve,
    options: &FitOptions,
    uniform: bool,
    regularization: f64,
) -> (Vec<f64>, f64, f64, f64, Vec<f64>, usize) {
    let count = curves.len();
    let mut weights = vec![1.0 / count as f64; count];
    if !uniform {
        for _ in 0..250 {
            let mixture = mix_curves(curves, &weights);
            let (scale, background, _) = solve_linear_fixed(&mixture, data, options);
            let residual = data
                .intensity
                .iter()
                .zip(&mixture)
                .zip(&data.sigma)
                .map(|((&observed, &model), &sigma)| {
                    (scale * model + background - observed) / sigma
                })
                .collect::<Vec<_>>();
            let mut gradient = vec![0.0; count];
            for (position, curve) in curves.iter().enumerate() {
                gradient[position] = 2.0
                    * curve
                        .iter()
                        .zip(&residual)
                        .zip(&data.sigma)
                        .map(|((&model, &residual), &sigma)| residual * scale * model / sigma)
                        .sum::<f64>()
                    + 2.0 * regularization * (weights[position] - 1.0 / count as f64);
            }
            let norm = gradient
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt()
                .max(1.0);
            let proposal = project_simplex(
                &weights
                    .iter()
                    .zip(&gradient)
                    .map(|(weight, gradient)| weight - 0.05 * gradient / norm)
                    .collect::<Vec<_>>(),
            );
            let delta = proposal
                .iter()
                .zip(&weights)
                .map(|(left, right)| (left - right).abs())
                .sum::<f64>();
            weights = proposal;
            if delta < 1e-8 {
                break;
            }
        }
    }
    let mixture = mix_curves(curves, &weights);
    let (scale, background, chi2) = solve_linear_fixed(&mixture, data, options);
    let fitted = mixture
        .iter()
        .map(|value| scale * value + background)
        .collect();
    (weights, scale, background, chi2, fitted, 250)
}

fn solve_linear_fixed(
    theory: &[f64],
    data: &ExperimentalCurve,
    options: &FitOptions,
) -> (f64, f64, f64) {
    let scale = match options.scale_mode {
        ScaleMode::Fit => {
            let mut numerator = 0.0;
            let mut denominator = 0.0;
            for ((&model, &observed), &sigma) in theory.iter().zip(&data.intensity).zip(&data.sigma)
            {
                numerator += model * observed / (sigma * sigma);
                denominator += model * model / (sigma * sigma);
            }
            if denominator > 0.0 {
                numerator / denominator
            } else {
                0.0
            }
        }
        ScaleMode::Fixed(value) => value,
    }
    .max(0.0);
    let background = match options.background_mode {
        BackgroundMode::Fit => {
            let mut numerator = 0.0;
            let mut denominator = 0.0;
            for ((&model, &observed), &sigma) in theory.iter().zip(&data.intensity).zip(&data.sigma)
            {
                numerator += (observed - scale * model) / (sigma * sigma);
                denominator += 1.0 / (sigma * sigma);
            }
            if denominator > 0.0 {
                numerator / denominator
            } else {
                0.0
            }
        }
        BackgroundMode::FixedZero => 0.0,
        BackgroundMode::Fixed(value) => value,
    };
    let chi2 = theory
        .iter()
        .zip(&data.intensity)
        .zip(&data.sigma)
        .map(|((&model, &observed), &sigma)| {
            ((scale * model + background - observed) / sigma).powi(2)
        })
        .sum::<f64>()
        / data.len().saturating_sub(1).max(1) as f64;
    (scale, background, chi2)
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

fn project_simplex(values: &[f64]) -> Vec<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| right.total_cmp(left));
    let mut cumulative = 0.0;
    let mut rho = 0usize;
    for (index, value) in sorted.iter().enumerate() {
        cumulative += value;
        if *value + (1.0 - cumulative) / (index + 1) as f64 > 0.0 {
            rho = index + 1;
        }
    }
    let theta = (sorted.iter().take(rho).sum::<f64>() - 1.0) / rho.max(1) as f64;
    values
        .iter()
        .map(|value| (value - theta).max(0.0))
        .collect()
}

fn quantile(values: &[f64], probability: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let position = probability.clamp(0.0, 1.0) * (values.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    values[lower] + (values[upper] - values[lower]) * (position - lower as f64)
}
