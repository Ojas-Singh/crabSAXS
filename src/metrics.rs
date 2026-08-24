//! SAXS-derived structural and curve features used for diagnostics and
//! robust model comparison.

use crate::error::{Result, SaxsError};
use crate::fit::FitParams;
use crate::formfactor::form_factor_for_atom;
use crate::io::ExperimentalCurve;
use crate::structure::Structure;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::path::Path;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PrOptions {
    /// Number of bins used for coordinate and indirect-transform P(r).
    pub bins: usize,
    /// Optional maximum distance for the experimental transform.
    pub max_r: Option<f64>,
}

impl Default for PrOptions {
    fn default() -> Self {
        Self {
            bins: 160,
            max_r: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairDistribution {
    pub r: Vec<f64>,
    pub p: Vec<f64>,
    pub rg: Option<f64>,
    pub dmax: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExperimentalAnalysis {
    pub rg: Option<f64>,
    pub rg_quality: Option<f64>,
    pub p_r: Option<PairDistribution>,
}

/// Optional values imported from an external GNOM or AUTORG analysis.
///
/// The internal analysis remains the default.  Call [`merge_external_analysis`]
/// when a validated external result should replace the corresponding internal
/// estimate while retaining internal values for fields the external tool did
/// not provide.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExternalSaxsAnalysis {
    pub rg: Option<f64>,
    pub rg_quality: Option<f64>,
    pub dmax: Option<f64>,
    pub p_r: Option<PairDistribution>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinateFeatures {
    pub rg: f64,
    pub dmax: f64,
    pub p_r: PairDistribution,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SaxsFeatures {
    pub chi2: f64,
    pub reduced_chi2: f64,
    pub scale: f64,
    pub background: f64,
    pub c1: f64,
    pub c2: f64,
    pub rg_exp: Option<f64>,
    pub rg_model: Option<f64>,
    pub rg_abs_error: Option<f64>,
    pub dmax_exp: Option<f64>,
    pub dmax_model: Option<f64>,
    pub dmax_abs_error: Option<f64>,
    pub kratky_deviation: Option<f64>,
    pub pr_rms: Option<f64>,
    pub pearson_correlation: Option<f64>,
    pub r_factor: Option<f64>,
    pub weighted_rmse: Option<f64>,
    pub points: usize,
}

/// Estimate experimental Guinier and indirect-transform features.
pub fn analyze_experimental(curve: &ExperimentalCurve, options: PrOptions) -> ExperimentalAnalysis {
    let (rg, quality) = estimate_rg(curve);
    ExperimentalAnalysis {
        rg,
        rg_quality: quality,
        p_r: estimate_pr(curve, options),
    }
}

/// Parse a text GNOM result, including labelled Rg/Dmax values and a tabular
/// `r, P(r)` section when present.
pub fn parse_gnom_output(text: &str) -> Result<ExternalSaxsAnalysis> {
    parse_external_output(text, true)
}

/// Parse a text AUTORG result. AUTORG commonly reports Rg and a fit quality;
/// any P(r) table is also accepted if the producer included one.
pub fn parse_autorg_output(text: &str) -> Result<ExternalSaxsAnalysis> {
    parse_external_output(text, true)
}

/// Read and parse a GNOM output file.
pub fn read_gnom_output(path: impl AsRef<Path>) -> Result<ExternalSaxsAnalysis> {
    parse_gnom_output(&std::fs::read_to_string(path)?)
}

/// Read and parse an AUTORG output file.
pub fn read_autorg_output(path: impl AsRef<Path>) -> Result<ExternalSaxsAnalysis> {
    parse_autorg_output(&std::fs::read_to_string(path)?)
}

/// Overlay optional external values on an internally computed analysis.
pub fn merge_external_analysis(
    mut internal: ExperimentalAnalysis,
    external: &ExternalSaxsAnalysis,
) -> ExperimentalAnalysis {
    if external.rg.is_some() {
        internal.rg = external.rg;
    }
    if external.rg_quality.is_some() {
        internal.rg_quality = external.rg_quality;
    }
    if let Some(mut p_r) = external.p_r.clone() {
        if external.rg.is_some() {
            p_r.rg = external.rg;
        }
        if external.dmax.is_some() {
            p_r.dmax = external.dmax;
        }
        internal.p_r = Some(p_r);
    } else if let Some(dmax) = external.dmax {
        if let Some(p_r) = internal.p_r.as_mut() {
            p_r.dmax = Some(dmax);
        }
    }
    internal
}

fn parse_external_output(text: &str, include_pr: bool) -> Result<ExternalSaxsAnalysis> {
    let rg = labelled_value(text, &["radius of gyration", "rg"]);
    let rg_quality = labelled_value(text, &["quality", "r^2", "r2"]);
    let dmax = labelled_value(text, &["dmax", "d-max", "maximum dimension"]);
    let p_r = include_pr.then(|| parse_pr_table(text)).flatten();
    if rg.is_none() && rg_quality.is_none() && dmax.is_none() && p_r.is_none() {
        return Err(SaxsError::InvalidInput(
            "external GNOM/AUTORG output did not contain recognizable SAXS values".into(),
        ));
    }
    Ok(ExternalSaxsAnalysis {
        rg,
        rg_quality,
        dmax,
        p_r,
    })
}

fn labelled_value(text: &str, labels: &[&str]) -> Option<f64> {
    text.lines().find_map(|line| {
        let lower = line.to_ascii_lowercase();
        labels.iter().find_map(|label| {
            let start = lower.find(label)? + label.len();
            numeric_tokens(&line[start..])
                .into_iter()
                .find(|value| value.is_finite())
        })
    })
}

fn parse_pr_table(text: &str) -> Option<PairDistribution> {
    let mut in_table = false;
    let mut rows = Vec::new();
    for line in text.lines() {
        let lower = line.to_ascii_lowercase();
        if !in_table
            && (lower.contains("p(r)")
                || (lower.contains("distance") && lower.contains("distribution")))
        {
            in_table = true;
            continue;
        }
        if !in_table {
            continue;
        }
        let values = numeric_tokens(line);
        if values.len() >= 2 && values[0] >= 0.0 && values[1] >= 0.0 {
            rows.push((values[0], values[1]));
        } else if !rows.is_empty() && line.trim().is_empty() {
            break;
        }
    }
    if rows.len() < 3 {
        return None;
    }
    rows.sort_by(|left, right| left.0.total_cmp(&right.0));
    let (r, p): (Vec<_>, Vec<_>) = rows.into_iter().unzip();
    normalize_distribution(r, p)
}

fn numeric_tokens(line: &str) -> Vec<f64> {
    let mut values = Vec::new();
    let mut token = String::new();
    let mut has_digit = false;
    let flush = |token: &mut String, has_digit: &mut bool, values: &mut Vec<f64>| {
        if *has_digit {
            if let Ok(value) = token.parse::<f64>() {
                values.push(value);
            }
        }
        token.clear();
        *has_digit = false;
    };
    for character in line.chars() {
        if character.is_ascii_digit() || matches!(character, '.' | '+' | '-' | 'e' | 'E') {
            token.push(character);
            has_digit |= character.is_ascii_digit();
        } else if !token.is_empty() {
            flush(&mut token, &mut has_digit, &mut values);
        }
    }
    if !token.is_empty() {
        flush(&mut token, &mut has_digit, &mut values);
    }
    values
}

/// Compute scattering-weighted coordinate Rg, Dmax, and a normalized P(r).
pub fn coordinate_features(
    structure: &Structure,
    options: PrOptions,
) -> Result<CoordinateFeatures> {
    if structure.atoms.is_empty() {
        return Err(SaxsError::InvalidInput(
            "coordinate features require a non-empty structure".into(),
        ));
    }
    let weights = structure
        .atoms
        .iter()
        .map(|atom| form_factor_for_atom(atom, true, 0.0).abs().max(1.0e-6))
        .collect::<Vec<_>>();
    let total_weight = weights.iter().sum::<f64>();
    let centre = structure
        .atoms
        .iter()
        .zip(&weights)
        .fold(nalgebra::Vector3::zeros(), |sum, (atom, weight)| {
            sum + atom.pos * *weight
        })
        / total_weight;
    let rg = (structure
        .atoms
        .iter()
        .zip(&weights)
        .map(|(atom, weight)| weight * (atom.pos - centre).norm_squared())
        .sum::<f64>()
        / total_weight)
        .sqrt();

    let (dmax, p_r) = coordinate_pr(structure, &weights, options.bins.max(16));
    Ok(CoordinateFeatures { rg, dmax, p_r })
}

/// Combine a fitted curve with structural and experimental diagnostics.
pub fn features_from_fit(
    experimental: &ExperimentalCurve,
    fitted_curve: &[f64],
    params: FitParams,
    coordinates: Option<&CoordinateFeatures>,
    experimental_analysis: Option<&ExperimentalAnalysis>,
) -> Result<SaxsFeatures> {
    let comparison = crate::analysis::compare_curves(experimental, fitted_curve)?;
    let analysis = experimental_analysis
        .cloned()
        .unwrap_or_else(|| analyze_experimental(experimental, PrOptions::default()));
    let rg_model = coordinates.map(|value| value.rg);
    let dmax_model = coordinates.map(|value| value.dmax);
    let rg_abs_error = match (analysis.rg, rg_model) {
        (Some(exp), Some(model)) => Some((exp - model).abs()),
        _ => None,
    };
    let dmax_exp = analysis.p_r.as_ref().and_then(|value| value.dmax);
    let dmax_abs_error = match (dmax_exp, dmax_model) {
        (Some(exp), Some(model)) => Some((exp - model).abs()),
        _ => None,
    };
    let model_pr = coordinates.map(|value| &value.p_r);
    let pr_rms = match (analysis.p_r.as_ref(), model_pr) {
        (Some(exp), Some(model)) => Some(pair_distribution_rms(exp, model)),
        _ => None,
    };
    Ok(SaxsFeatures {
        chi2: comparison.chi2,
        reduced_chi2: comparison.reduced_chi2,
        scale: params.scale,
        background: params.background,
        c1: params.c1,
        c2: params.c2,
        rg_exp: analysis.rg,
        rg_model,
        rg_abs_error,
        dmax_exp,
        dmax_model,
        dmax_abs_error,
        kratky_deviation: kratky_deviation(experimental, fitted_curve),
        pr_rms,
        pearson_correlation: Some(comparison.pearson_correlation),
        r_factor: Some(comparison.r_factor),
        weighted_rmse: Some(comparison.weighted_rmse),
        points: experimental.len(),
    })
}

fn estimate_rg(curve: &ExperimentalCurve) -> (Option<f64>, Option<f64>) {
    let points = curve
        .q
        .iter()
        .zip(&curve.intensity)
        .filter_map(|(&q, &intensity)| {
            (q > 0.0 && intensity > 0.0 && q.is_finite() && intensity.is_finite()).then_some((
                q * q,
                intensity.ln(),
                q,
            ))
        })
        .collect::<Vec<_>>();
    if points.len() < 4 {
        return (None, None);
    }
    let max_prefix = points.len().min(30);
    let mut best = None;
    for prefix in 4..=max_prefix {
        let sample = &points[..prefix];
        let (slope, _intercept, r2) = regression(
            &sample.iter().map(|point| point.0).collect::<Vec<_>>(),
            &sample.iter().map(|point| point.1).collect::<Vec<_>>(),
        );
        if slope >= 0.0 || !slope.is_finite() {
            continue;
        }
        let rg = (-3.0 * slope).sqrt();
        let q_rg = sample.last().map_or(f64::INFINITY, |point| point.2 * rg);
        if rg.is_finite() && q_rg <= 1.6 && r2.is_finite() {
            let score = r2 - 0.002 * (prefix as f64);
            if best
                .as_ref()
                .is_none_or(|current: &(f64, f64, f64)| score > current.0)
            {
                best = Some((score, rg, r2));
            }
        }
    }
    if let Some((_score, rg, quality)) = best {
        return (Some(rg), Some(quality));
    }
    let sample = &points[..4];
    let xs = sample.iter().map(|point| point.0).collect::<Vec<_>>();
    let ys = sample.iter().map(|point| point.1).collect::<Vec<_>>();
    let (slope, _, quality) = regression(&xs, &ys);
    (slope < 0.0)
        .then_some((-3.0 * slope).sqrt())
        .map_or((None, None), |rg| {
            (
                rg.is_finite().then_some(rg),
                quality.is_finite().then_some(quality),
            )
        })
}

fn regression(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    if x.len() != y.len() || x.len() < 2 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let mean_x = x.iter().sum::<f64>() / x.len() as f64;
    let mean_y = y.iter().sum::<f64>() / y.len() as f64;
    let denominator = x.iter().map(|value| (value - mean_x).powi(2)).sum::<f64>();
    if denominator <= 0.0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let slope = x
        .iter()
        .zip(y)
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum::<f64>()
        / denominator;
    let intercept = mean_y - slope * mean_x;
    let total = y.iter().map(|value| (value - mean_y).powi(2)).sum::<f64>();
    let residual = x
        .iter()
        .zip(y)
        .map(|(x, y)| (y - (slope * x + intercept)).powi(2))
        .sum::<f64>();
    let r2 = if total > 0.0 {
        1.0 - residual / total
    } else {
        0.0
    };
    (slope, intercept, r2)
}

fn estimate_pr(curve: &ExperimentalCurve, options: PrOptions) -> Option<PairDistribution> {
    let samples = curve
        .q
        .iter()
        .zip(&curve.intensity)
        .filter(|(q, intensity)| **q > 0.0 && intensity.is_finite())
        .map(|(&q, &intensity)| (q, intensity))
        .collect::<Vec<_>>();
    if samples.len() < 4 {
        return None;
    }
    let q_min = samples.first()?.0;
    let q_max = samples.last()?.0;
    let max_r = options
        .max_r
        .unwrap_or_else(|| (PI / q_min).min(500.0).max(20.0));
    if !max_r.is_finite() || max_r <= 0.0 || q_max <= q_min {
        return None;
    }
    let bins = options.bins.max(16);
    let step = max_r / bins as f64;
    let tail_count = samples.len().min(8);
    let mut tail = samples
        .iter()
        .rev()
        .take(tail_count)
        .map(|(_, intensity)| *intensity)
        .collect::<Vec<_>>();
    tail.sort_by(f64::total_cmp);
    let background = tail[tail.len() / 2].max(0.0);
    let r = (0..bins)
        .map(|index| (index as f64 + 0.5) * step)
        .collect::<Vec<_>>();
    // The q-dependent quadrature factors do not depend on r.  Precomputing
    // them removes two sinc evaluations and several multiplies from every
    // r/q pair, which matters for the 998-point experimental curves used by
    // the Re-Glyco reports.
    let segments = samples
        .windows(2)
        .map(|pair| {
            let (q0, i0) = pair[0];
            let (q1, i1) = pair[1];
            let factor = 0.5 * (q1 - q0);
            (
                q0,
                q1,
                factor * q0 * (i0 - background) * crate::scattering::sinc(PI * q0 / q_max),
                factor * q1 * (i1 - background) * crate::scattering::sinc(PI * q1 / q_max),
            )
        })
        .collect::<Vec<_>>();
    let p = r
        .par_iter()
        .map(|&distance| {
            let integral = segments
                .iter()
                .map(|&(q0, q1, factor0, factor1)| {
                    factor0 * (q0 * distance).sin() + factor1 * (q1 * distance).sin()
                })
                .sum::<f64>();
            (distance * integral / (2.0 * PI * PI)).max(0.0)
        })
        .collect::<Vec<_>>();
    normalize_distribution(r, p)
}

fn coordinate_pr(structure: &Structure, weights: &[f64], bins: usize) -> (f64, PairDistribution) {
    let positions = structure
        .atoms
        .iter()
        .map(|atom| [atom.pos.x, atom.pos.y, atom.pos.z])
        .collect::<Vec<_>>();
    let dmax_squared = (0..positions.len())
        .into_par_iter()
        .map(|left| {
            positions[(left + 1).min(positions.len())..]
                .iter()
                .map(|right| squared_distance(positions[left], *right))
                .fold(0.0, f64::max)
        })
        .reduce(|| 0.0, f64::max);
    if dmax_squared <= 0.0 {
        return (
            0.0,
            PairDistribution {
                r: vec![0.0],
                p: vec![1.0],
                rg: Some(0.0),
                dmax: Some(0.0),
            },
        );
    }
    let dmax = dmax_squared.sqrt();
    let step = dmax / bins as f64;
    // Work in deterministic coarse chunks.  This keeps the reduction stable
    // while allowing all atom-pair rows to run concurrently.  More
    // importantly, the distance square root is now evaluated only once per
    // pair: the Dmax pass above works entirely in squared distances.
    const ROW_CHUNK: usize = 64;
    let chunk_starts = (0..positions.len()).step_by(ROW_CHUNK).collect::<Vec<_>>();
    let histograms = chunk_starts
        .par_iter()
        .map(|&start| {
            let end = (start + ROW_CHUNK).min(positions.len());
            let mut histogram = vec![0.0; bins];
            for left in start..end {
                for right in (left + 1)..positions.len() {
                    let distance = squared_distance(positions[left], positions[right]).sqrt();
                    let index = ((distance / step).floor() as usize).min(bins - 1);
                    histogram[index] += weights[left] * weights[right];
                }
            }
            histogram
        })
        .collect::<Vec<_>>();
    let mut p = vec![0.0; bins];
    for histogram in histograms {
        for (value, contribution) in p.iter_mut().zip(histogram) {
            *value += contribution;
        }
    }
    let r = (0..bins)
        .map(|index| (index as f64 + 0.5) * step)
        .collect::<Vec<_>>();
    let p_r = normalize_distribution(r, p).unwrap_or(PairDistribution {
        r: vec![step * 0.5],
        p: vec![1.0],
        rg: Some(0.0),
        dmax: Some(dmax),
    });
    (dmax, p_r)
}

#[inline]
fn squared_distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    let dx = left[0] - right[0];
    let dy = left[1] - right[1];
    let dz = left[2] - right[2];
    dx * dx + dy * dy + dz * dz
}

fn normalize_distribution(r: Vec<f64>, mut p: Vec<f64>) -> Option<PairDistribution> {
    if r.len() != p.len() || p.is_empty() {
        return None;
    }
    let area = r
        .windows(2)
        .zip(p.windows(2))
        .map(|(r, p)| 0.5 * (r[1] - r[0]) * (p[0] + p[1]))
        .sum::<f64>();
    if !area.is_finite() || area <= 0.0 {
        return None;
    }
    for value in &mut p {
        *value /= area;
    }
    let moment = r
        .windows(2)
        .zip(p.windows(2))
        .map(|(r, p)| 0.5 * (r[1] - r[0]) * (r[0] * r[0] * p[0] + r[1] * r[1] * p[1]))
        .sum::<f64>();
    let rg = (moment / 2.0).max(0.0).sqrt();
    let maximum = p.iter().copied().fold(0.0, f64::max);
    let dmax = p
        .iter()
        .enumerate()
        .rev()
        .find(|(_, value)| **value > maximum * 0.01)
        .map(|(index, _)| r[index]);
    Some(PairDistribution {
        r,
        p,
        rg: rg.is_finite().then_some(rg),
        dmax,
    })
}

fn kratky_deviation(experimental: &ExperimentalCurve, fitted: &[f64]) -> Option<f64> {
    if experimental.len() != fitted.len() || experimental.is_empty() {
        return None;
    }
    let observed_scale = experimental
        .intensity
        .iter()
        .copied()
        .find(|value| *value > 0.0)?;
    let model_scale = fitted.iter().copied().find(|value| *value > 0.0)?;
    let sum = experimental
        .q
        .iter()
        .zip(&experimental.intensity)
        .zip(fitted)
        .map(|((&q, observed), model)| {
            let left = q * q * observed / observed_scale;
            let right = q * q * model.max(0.0) / model_scale;
            (left - right).powi(2)
        })
        .sum::<f64>();
    Some((sum / experimental.len() as f64).sqrt())
}

fn pair_distribution_rms(left: &PairDistribution, right: &PairDistribution) -> f64 {
    if left.r.is_empty() || right.r.is_empty() {
        return f64::NAN;
    }
    let sum = left
        .r
        .iter()
        .zip(&left.p)
        .map(|(&r, &value)| (value - interpolate(&right.r, &right.p, r)).powi(2))
        .sum::<f64>();
    (sum / left.r.len() as f64).sqrt()
}

fn interpolate(x: &[f64], y: &[f64], target: f64) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }
    if target <= x[0] {
        return y[0];
    }
    if target >= *x.last().unwrap_or(&x[0]) {
        return 0.0;
    }
    let mut index = 0;
    while index + 1 < x.len() && x[index + 1] < target {
        index += 1;
    }
    let denominator = x[index + 1] - x[index];
    if denominator <= 0.0 {
        return y[index];
    }
    let t = (target - x[index]) / denominator;
    y[index] * (1.0 - t) + y[index + 1] * t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::{Atom, Element};
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
                    residue_name: "GLY".into(),
                    chain_id: "A".into(),
                    residue_id: 1,
                },
                Atom {
                    pos: Vector3::new(4.0, 0.0, 0.0),
                    element: Element::C,
                    occupancy: 1.0,
                    is_hetatm: false,
                    atom_name: "C2".into(),
                    residue_name: "GLY".into(),
                    chain_id: "A".into(),
                    residue_id: 1,
                },
            ],
        }
    }

    #[test]
    fn coordinate_features_report_pair_distance() {
        let features = coordinate_features(&structure(), PrOptions::default()).unwrap();
        assert!((features.dmax - 4.0).abs() < 1e-12);
        assert!(features.rg > 1.0);
        assert!(!features.p_r.p.is_empty());
    }

    #[test]
    fn guinier_fit_recovers_synthetic_rg() {
        let rg = 18.0;
        let q = (1..80).map(|i| i as f64 * 0.001).collect::<Vec<_>>();
        let intensity = q
            .iter()
            .map(|q| 100.0 * (-(rg * rg * q * q) / 3.0).exp())
            .collect::<Vec<_>>();
        let curve = ExperimentalCurve {
            q,
            intensity,
            sigma: vec![0.1; 79],
            has_errors: true,
        };
        let analysis = analyze_experimental(&curve, PrOptions::default());
        assert!((analysis.rg.unwrap() - rg).abs() < 0.1);
    }

    #[test]
    fn imports_gnom_values_and_pair_distribution() {
        let text = "Rg = 18.2 A\nDmax = 64.0 A\n\nr (A) P(r)\n0 0\n10 1\n20 2\n30 1\n40 0\n";
        let imported = parse_gnom_output(text).unwrap();
        assert_eq!(imported.rg, Some(18.2));
        assert_eq!(imported.dmax, Some(64.0));
        assert!(imported.p_r.is_some());
        let merged = merge_external_analysis(
            ExperimentalAnalysis {
                rg: Some(10.0),
                rg_quality: None,
                p_r: None,
            },
            &imported,
        );
        assert_eq!(merged.rg, Some(18.2));
        assert_eq!(merged.p_r.unwrap().dmax, Some(64.0));
    }

    #[test]
    fn imports_autorg_radius_and_quality() {
        let imported = parse_autorg_output("Rg = 22.5 A\nquality = 0.98\n").unwrap();
        assert_eq!(imported.rg, Some(22.5));
        assert_eq!(imported.rg_quality, Some(0.98));
    }
}
