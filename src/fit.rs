//! Weighted least-squares fitting of scale, background, and solvent terms.

use crate::error::{Result, SaxsError};
use crate::io::ExperimentalCurve;
use crate::scattering::{HydrogenMode, ScatteringMethod};
use crate::solvent::{HydrationModel, SolventParams};
use crate::structure::Structure;
use argmin::core::{CostFunction, Error as ArgminError, Executor, State, TerminationReason};
use argmin::solver::neldermead::NelderMead;

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct FitParams {
    pub scale: f64,
    pub background: f64,
    pub c1: f64,
    pub c2: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FitResult {
    pub params: FitParams,
    pub chi2: f64,
    pub fitted_curve: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
    pub termination: String,
}

/// Whether the overall intensity scale is optimized or held at a supplied value.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ScaleMode {
    Fit,
    Fixed(f64),
}

/// Treatment of a q-independent experimental intensity offset.
///
/// `Fit` is the model described by Grudinin et al. (2017), Supporting
/// Information equations (3)-(6). `FixedZero` provides a like-for-like mode
/// for programs run without constant-background correction.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackgroundMode {
    Fit,
    FixedZero,
    Fixed(f64),
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct FitOptions {
    pub method: ScatteringMethod,
    pub hydrogen_mode: HydrogenMode,
    pub c1_bounds: (f64, f64),
    pub c2_bounds: (f64, f64),
    pub max_iterations: usize,
    /// Evaluate the expensive multipole basis at every Nth experimental
    /// point and use cubic Hermite interpolation between samples.
    pub q_sampling_stride: usize,
    pub solvent: SolventParams,
    pub background_mode: BackgroundMode,
    pub scale_mode: ScaleMode,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            method: ScatteringMethod::Auto,
            hydrogen_mode: HydrogenMode::Implicit,
            // Published Pepsi-SAXS global excluded-volume radius range.
            c1_bounds: (0.95, 1.05),
            // The legacy analytic shell uses an amplitude multiplier rather
            // than a normalized density contrast. Accurate adaptive-grid
            // callers should use the published (-0.045, 0.1) bounds.
            c2_bounds: (-2.0, 4.0),
            max_iterations: 64,
            q_sampling_stride: 20,
            solvent: SolventParams::default(),
            background_mode: BackgroundMode::Fit,
            scale_mode: ScaleMode::Fit,
        }
    }
}

fn sampled_indices(length: usize, stride: usize) -> Vec<usize> {
    let stride = if length < 64 { 1 } else { stride.max(1) };
    let mut indices = (0..length).step_by(stride).collect::<Vec<_>>();
    if indices.last().copied() != Some(length - 1) {
        indices.push(length - 1);
    }
    indices
}

fn cubic_hermite_interpolate(x: &[f64], y: &[f64], targets: &[f64]) -> Vec<f64> {
    debug_assert_eq!(x.len(), y.len());
    if x.len() == 1 {
        return vec![y[0]; targets.len()];
    }
    let mut slopes = vec![0.0; x.len()];
    slopes[0] = (y[1] - y[0]) / (x[1] - x[0]);
    let last = x.len() - 1;
    slopes[last] = (y[last] - y[last - 1]) / (x[last] - x[last - 1]);
    for i in 1..last {
        slopes[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1]);
    }

    let mut segment = 0;
    targets
        .iter()
        .map(|&target| {
            while segment + 1 < last && target > x[segment + 1] {
                segment += 1;
            }
            let h = x[segment + 1] - x[segment];
            let t = ((target - x[segment]) / h).clamp(0.0, 1.0);
            let t2 = t * t;
            let t3 = t2 * t;
            (2.0 * t3 - 3.0 * t2 + 1.0) * y[segment]
                + (t3 - 2.0 * t2 + t) * h * slopes[segment]
                + (-2.0 * t3 + 3.0 * t2) * y[segment + 1]
                + (t3 - t2) * h * slopes[segment + 1]
        })
        .collect()
}

impl Default for FitParams {
    fn default() -> Self {
        let p = SolventParams::default();
        Self {
            scale: 1.0,
            background: 0.0,
            c1: p.c1,
            c2: p.c2,
        }
    }
}

fn solve_linear(
    theory: &[f64],
    data: &ExperimentalCurve,
    background_mode: BackgroundMode,
) -> (f64, f64, f64) {
    let mut sw = 0.0;
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for ((&x, &y), &sigma) in theory.iter().zip(&data.intensity).zip(&data.sigma) {
        let w = 1.0 / (sigma * sigma);
        sw += w;
        sx += w * x;
        sy += w * y;
        sxx += w * x * x;
        sxy += w * x * y;
    }
    let (scale, bg) = match background_mode {
        BackgroundMode::Fit => {
            let det = sxx * sw - sx * sx;
            let scale = if det.abs() > 1e-20 {
                (sxy * sw - sx * sy) / det
            } else {
                0.0
            }
            .max(0.0);
            let background = if sw > 0.0 {
                (sy - scale * sx) / sw
            } else {
                0.0
            };
            (scale, background)
        }
        BackgroundMode::FixedZero => {
            let scale = if sxx > 0.0 { sxy / sxx } else { 0.0 }.max(0.0);
            (scale, 0.0)
        }
        BackgroundMode::Fixed(background) => {
            let scale = if sxx > 0.0 {
                (sxy - background * sx) / sxx
            } else {
                0.0
            }
            .max(0.0);
            (scale, background)
        }
    };
    // Pepsi-SAXS defines its reported reduced chi-squared with N-1 in the
    // denominator (Grudinin et al. 2017, equation 16). Keeping that convention
    // makes API and black-box comparison values directly comparable.
    let degrees_of_freedom = data.len().saturating_sub(1).max(1) as f64;
    let chi = data
        .intensity
        .iter()
        .zip(theory)
        .zip(&data.sigma)
        .map(|((&y, &x), &s)| ((scale * x + bg - y) / s).powi(2))
        .sum::<f64>()
        / degrees_of_freedom;
    (scale, bg, chi)
}

fn solve_scale_background(
    theory: &[f64],
    data: &ExperimentalCurve,
    scale_mode: ScaleMode,
    background_mode: BackgroundMode,
) -> (f64, f64, f64) {
    if matches!(scale_mode, ScaleMode::Fit) {
        return solve_linear(theory, data, background_mode);
    }
    let scale = match scale_mode {
        ScaleMode::Fit => unreachable!(),
        ScaleMode::Fixed(value) => value.max(0.0),
    };
    let background = match background_mode {
        BackgroundMode::Fit => {
            let mut weighted = 0.0;
            let mut weight = 0.0;
            for ((&model, &observed), &sigma) in theory.iter().zip(&data.intensity).zip(&data.sigma)
            {
                let w = 1.0 / (sigma * sigma);
                weighted += w * (observed - scale * model);
                weight += w;
            }
            if weight > 0.0 {
                weighted / weight
            } else {
                0.0
            }
        }
        BackgroundMode::FixedZero => 0.0,
        BackgroundMode::Fixed(value) => value,
    };
    let dof = data.len().saturating_sub(1).max(1) as f64;
    let chi = data
        .intensity
        .iter()
        .zip(theory)
        .zip(&data.sigma)
        .map(|((&observed, &model), &sigma)| {
            ((scale * model + background - observed) / sigma).powi(2)
        })
        .sum::<f64>()
        / dof;
    (scale, background, chi)
}

struct SolventObjective<'a> {
    basis: &'a [Vec<f64>; 6],
    q_values: &'a [f64],
    data: &'a ExperimentalCurve,
    c1_bounds: (f64, f64),
    c2_bounds: (f64, f64),
    background_mode: BackgroundMode,
    scale_mode: ScaleMode,
}

impl CostFunction for SolventObjective<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> std::result::Result<Self::Output, ArgminError> {
        let c1 = param[0];
        let c2 = param[1];
        if !(self.c1_bounds.0..=self.c1_bounds.1).contains(&c1)
            || !(self.c2_bounds.0..=self.c2_bounds.1).contains(&c2)
        {
            let c1_distance = (self.c1_bounds.0 - c1).max(0.0) + (c1 - self.c1_bounds.1).max(0.0);
            let c2_distance = (self.c2_bounds.0 - c2).max(0.0) + (c2 - self.c2_bounds.1).max(0.0);
            return Ok(1e20 + 1e20 * (c1_distance * c1_distance + c2_distance * c2_distance));
        }
        let curve = crate::spherical::combine_solvent_basis(self.basis, self.q_values, c1, c2);
        Ok(solve_scale_background(&curve, self.data, self.scale_mode, self.background_mode).2)
    }
}

pub fn fit_to_experimental(
    structure: &Structure,
    experimental: &ExperimentalCurve,
    initial_guess: FitParams,
) -> Result<FitResult> {
    fit_to_experimental_with_options(
        structure,
        experimental,
        initial_guess,
        FitOptions::default(),
    )
}

pub fn fit_to_experimental_with_options(
    structure: &Structure,
    experimental: &ExperimentalCurve,
    initial: FitParams,
    options: FitOptions,
) -> Result<FitResult> {
    if experimental.is_empty()
        || structure.atoms.is_empty()
        || experimental.q.len() != experimental.intensity.len()
        || experimental.q.len() != experimental.sigma.len()
        || experimental.q.iter().any(|q| !q.is_finite() || *q < 0.0)
        || experimental.q.windows(2).any(|pair| pair[1] < pair[0])
        || experimental
            .intensity
            .iter()
            .any(|value| !value.is_finite())
        || experimental
            .sigma
            .iter()
            .any(|sigma| !sigma.is_finite() || *sigma <= 0.0)
    {
        return Err(SaxsError::InvalidInput(
            "fit requires a non-empty structure and equal-length finite q, intensity, and positive-sigma arrays on a non-decreasing q grid".into(),
        ));
    }
    if options.q_sampling_stride == 0
        || options.max_iterations == 0
        || !options.c1_bounds.0.is_finite()
        || !options.c1_bounds.1.is_finite()
        || options.c1_bounds.0 <= 0.0
        || options.c1_bounds.0 >= options.c1_bounds.1
        || !options.c2_bounds.0.is_finite()
        || !options.c2_bounds.1.is_finite()
        || options.c2_bounds.0 >= options.c2_bounds.1
        || matches!(options.scale_mode, ScaleMode::Fixed(value) if !value.is_finite() || value < 0.0)
        || matches!(options.background_mode, BackgroundMode::Fixed(value) if !value.is_finite())
    {
        return Err(SaxsError::InvalidInput(
            "fit bounds, iteration limit, and q sampling stride are invalid".into(),
        ));
    }
    let centre = structure
        .atoms
        .iter()
        .fold(nalgebra::Vector3::zeros(), |sum, atom| sum + atom.pos)
        / structure.atoms.len().max(1) as f64;
    let rmax = structure
        .atoms
        .iter()
        .map(|atom| (atom.pos - centre).norm())
        .fold(0.0_f64, f64::max);
    let qmax = experimental.q.iter().copied().fold(0.0_f64, f64::max);
    // Allocate through the largest per-q order used by the published-style
    // qR + 8 truncation rule. Using the full experimental grid also makes the
    // reported chi² exactly the objective that was optimized rather than a
    // downsampled surrogate.
    let l_max = ((qmax * rmax).ceil() as usize + 8).clamp(8, 64);
    let sample_indices = sampled_indices(experimental.len(), options.q_sampling_stride);
    let mut sampled_q = sample_indices
        .iter()
        .map(|&index| experimental.q[index])
        .collect::<Vec<_>>();
    sampled_q.dedup();
    let implicit_hydrogen = matches!(options.hydrogen_mode, HydrogenMode::Implicit);
    let method = match options.method {
        ScatteringMethod::Auto if options.solvent.hydration_model != HydrationModel::Analytic => {
            ScatteringMethod::Multipole
        }
        ScatteringMethod::Auto if structure.atoms.len() < 500 => ScatteringMethod::Debye,
        ScatteringMethod::Auto => ScatteringMethod::Multipole,
        method => method,
    };
    let sampled_basis = match method {
        ScatteringMethod::Debye => crate::solvent::debye_solvent_basis(
            structure,
            &sampled_q,
            &options.solvent,
            implicit_hydrogen,
        ),
        ScatteringMethod::Multipole => crate::spherical::multipole_solvent_basis(
            structure,
            &sampled_q,
            &crate::spherical::MultipoleParams {
                l_max,
                hydrogen_mode: options.hydrogen_mode,
            },
            &options.solvent,
        ),
        ScatteringMethod::Auto => unreachable!(),
    }?;
    let basis = if sampled_q.len() == experimental.len() {
        sampled_basis
    } else {
        std::array::from_fn(|component| {
            cubic_hermite_interpolate(&sampled_q, &sampled_basis[component], &experimental.q)
        })
    };
    let initial_c1 = initial.c1.clamp(options.c1_bounds.0, options.c1_bounds.1);
    let initial_c2 = initial.c2.clamp(options.c2_bounds.0, options.c2_bounds.1);
    let step1 = (options.c1_bounds.1 - options.c1_bounds.0) * 0.1;
    let step2 = (options.c2_bounds.1 - options.c2_bounds.0) * 0.1;
    let offset = |value: f64, step: f64, bounds: (f64, f64)| {
        if value + step <= bounds.1 {
            value + step
        } else {
            value - step
        }
    };
    let simplex = vec![
        vec![initial_c1, initial_c2],
        vec![offset(initial_c1, step1, options.c1_bounds), initial_c2],
        vec![initial_c1, offset(initial_c2, step2, options.c2_bounds)],
    ];
    let solver = NelderMead::new(simplex)
        .with_sd_tolerance(1e-8)
        .map_err(|error| SaxsError::FitDidNotConverge(error.to_string()))?;
    let optimization = Executor::new(
        SolventObjective {
            basis: &basis,
            q_values: &experimental.q,
            data: experimental,
            c1_bounds: options.c1_bounds,
            c2_bounds: options.c2_bounds,
            background_mode: options.background_mode,
            scale_mode: options.scale_mode,
        },
        solver,
    )
    .configure(|state| state.max_iters(options.max_iterations as u64))
    .run()
    .map_err(|error| SaxsError::FitDidNotConverge(error.to_string()))?;
    let state = optimization.state();
    let best_param = state
        .get_best_param()
        .ok_or_else(|| SaxsError::FitDidNotConverge("optimizer returned no parameters".into()))?;
    let c1 = best_param[0].clamp(options.c1_bounds.0, options.c1_bounds.1);
    let c2 = best_param[1].clamp(options.c2_bounds.0, options.c2_bounds.1);
    let raw = crate::spherical::combine_solvent_basis(&basis, &experimental.q, c1, c2);
    let (scale, background, chi2) = solve_scale_background(
        &raw,
        experimental,
        options.scale_mode,
        options.background_mode,
    );
    if !chi2.is_finite() {
        return Err(SaxsError::FitDidNotConverge(
            "objective was not finite".into(),
        ));
    }
    let fitted = raw.iter().map(|x| scale * x + background).collect();
    Ok(FitResult {
        params: FitParams {
            scale,
            background,
            c1,
            c2,
        },
        chi2,
        fitted_curve: fitted,
        iterations: state.get_iter() as usize,
        converged: matches!(
            state.get_termination_reason(),
            Some(TerminationReason::SolverConverged | TerminationReason::TargetCostReached)
        ),
        termination: state.get_termination_status().to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::{Atom, Element, Structure};
    use nalgebra::Vector3;

    fn structure() -> Structure {
        Structure {
            atoms: vec![
                Atom {
                    pos: Vector3::new(0.0, 0.0, 0.0),
                    element: Element::C,
                    occupancy: 1.0,
                    is_hetatm: false,
                    atom_name: "C1".into(),
                    residue_name: "TST".into(),
                    chain_id: "A".into(),
                    residue_id: 1,
                },
                Atom {
                    pos: Vector3::new(2.0, 0.0, 0.0),
                    element: Element::O,
                    occupancy: 1.0,
                    is_hetatm: false,
                    atom_name: "O1".into(),
                    residue_name: "TST".into(),
                    chain_id: "A".into(),
                    residue_id: 1,
                },
            ],
        }
    }

    #[test]
    fn recovers_scale_and_background_on_synthetic_curve() {
        let s = structure();
        let q = (1..=8).map(|i| i as f64 * 0.03).collect::<Vec<_>>();
        let p = SolventParams {
            c1: 1.05,
            c2: 0.4,
            ..Default::default()
        };
        let raw = crate::spherical::multipole_solvent_intensity(
            &s,
            &q,
            &crate::spherical::MultipoleParams {
                l_max: 8,
                hydrogen_mode: HydrogenMode::Explicit,
            },
            &p,
        )
        .unwrap();
        let data = ExperimentalCurve {
            q: q.clone(),
            intensity: raw.iter().map(|v| 1.7 * v + 0.02).collect(),
            sigma: vec![0.01; q.len()],
            has_errors: true,
        };
        let (scale, background, chi2) = solve_linear(&raw, &data, BackgroundMode::Fit);
        assert!((scale - 1.7).abs() < 1e-8);
        assert!((background - 0.02).abs() < 1e-8);
        assert!(chi2 < 1e-16);
    }

    #[test]
    fn fixed_zero_background_uses_weighted_scale_only() {
        let data = ExperimentalCurve {
            q: vec![0.1, 0.2, 0.3],
            intensity: vec![2.0, 4.0, 6.0],
            sigma: vec![1.0, 2.0, 1.0],
            has_errors: true,
        };
        let (scale, background, chi2) =
            solve_linear(&[1.0, 2.0, 3.0], &data, BackgroundMode::FixedZero);
        assert!((scale - 2.0).abs() < 1e-12);
        assert_eq!(background, 0.0);
        assert!(chi2 < 1e-24);
    }

    #[test]
    fn recovers_solvent_parameters_with_deterministic_noise() {
        let s = structure();
        let q = (1..=80).map(|i| i as f64 * 0.006).collect::<Vec<_>>();
        let expected = SolventParams {
            c1: 1.02,
            c2: 0.55,
            ..Default::default()
        };
        let raw = crate::spherical::multipole_solvent_intensity(
            &s,
            &q,
            &crate::spherical::MultipoleParams {
                l_max: 8,
                hydrogen_mode: HydrogenMode::Explicit,
            },
            &expected,
        )
        .unwrap();
        let sigma = vec![0.002; q.len()];
        let mut random_state = 0x5eed_u64;
        let mut gaussian_noise = || {
            let mut uniform = || {
                random_state = random_state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                ((random_state >> 11) as f64 / ((1u64 << 53) as f64)).max(f64::MIN_POSITIVE)
            };
            let u1 = uniform();
            let u2 = uniform();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        let data = ExperimentalCurve {
            q: q.clone(),
            intensity: raw
                .iter()
                .map(|value| 1.4e-4 * value + 0.03 + 0.0001 * gaussian_noise())
                .collect(),
            sigma,
            has_errors: true,
        };
        let result = fit_to_experimental_with_options(
            &s,
            &data,
            FitParams {
                scale: 1.4e-4,
                background: 0.03,
                c1: expected.c1,
                c2: expected.c2,
            },
            FitOptions {
                method: ScatteringMethod::Multipole,
                hydrogen_mode: HydrogenMode::Explicit,
                max_iterations: 100,
                q_sampling_stride: 1,
                ..Default::default()
            },
        )
        .unwrap();
        assert!((result.params.c1 - expected.c1).abs() < 0.02, "{result:?}");
        assert!((result.params.c2 - expected.c2).abs() < 0.08, "{result:?}");
        assert!((result.params.scale - 1.4e-4).abs() < 0.4e-4, "{result:?}");
        assert!((result.params.background - 0.03).abs() < 0.01, "{result:?}");
        assert!(result.chi2 < 0.01, "{result:?}");
    }

    #[test]
    fn sampled_basis_matches_full_basis_fit() {
        let structure = structure();
        let q = (1..=101)
            .map(|index| index as f64 * 0.005)
            .collect::<Vec<_>>();
        let solvent = SolventParams {
            c1: 1.03,
            c2: 0.3,
            ..Default::default()
        };
        let raw = crate::spherical::multipole_solvent_intensity(
            &structure,
            &q,
            &crate::spherical::MultipoleParams {
                l_max: 8,
                hydrogen_mode: HydrogenMode::Explicit,
            },
            &solvent,
        )
        .unwrap();
        let data = ExperimentalCurve {
            q: q.clone(),
            intensity: raw.iter().map(|value| 0.8 * value + 0.01).collect(),
            sigma: vec![0.01; q.len()],
            has_errors: true,
        };
        let common = FitOptions {
            method: ScatteringMethod::Multipole,
            hydrogen_mode: HydrogenMode::Explicit,
            max_iterations: 64,
            ..Default::default()
        };
        let full = fit_to_experimental_with_options(
            &structure,
            &data,
            FitParams::default(),
            FitOptions {
                q_sampling_stride: 1,
                ..common
            },
        )
        .unwrap();
        let sampled =
            fit_to_experimental_with_options(&structure, &data, FitParams::default(), common)
                .unwrap();
        let relative_rmse = (full
            .fitted_curve
            .iter()
            .zip(&sampled.fitted_curve)
            .map(|(left, right)| (left - right).powi(2))
            .sum::<f64>()
            / full
                .fitted_curve
                .iter()
                .map(|value| value * value)
                .sum::<f64>())
        .sqrt();
        assert!(relative_rmse < 1e-3, "relative RMSE was {relative_rmse}");
    }
}
