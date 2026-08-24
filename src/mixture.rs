//! Generic candidate-combination ranking and marginalization.

use crate::error::{Result, SaxsError};
use crate::metrics::SaxsFeatures;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Assignment {
    pub site: String,
    pub candidate: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateCombination {
    pub assignments: Vec<Assignment>,
    pub prior: f64,
    pub features: SaxsFeatures,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RankWeights {
    pub chi2: f64,
    pub kratky: f64,
    pub rg: f64,
    pub dmax: f64,
    pub pr: f64,
}

impl Default for RankWeights {
    fn default() -> Self {
        Self {
            chi2: 1.0,
            kratky: 1.0,
            rg: 1.0,
            dmax: 1.0,
            pr: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombinationPosterior {
    pub assignments: Vec<Assignment>,
    pub prior: f64,
    /// `None` represents a zero-support combination (for example, a zero
    /// supplied prior), because JSON has no representation for -∞.
    pub log_likelihood: Option<f64>,
    pub likelihood: f64,
    pub posterior: f64,
    /// Higher is better; this is deliberately separate from posterior.
    pub robust_score: f64,
    pub rank_components: BTreeMap<String, f64>,
    pub features: SaxsFeatures,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CombinationAnalysis {
    pub combinations: Vec<CombinationPosterior>,
    pub site_occupancy: BTreeMap<String, f64>,
    pub glycoform_distribution: BTreeMap<String, BTreeMap<String, f64>>,
}

/// Rank combinations by equal-weight normalized feature ranks and calculate
/// chi-squared likelihood marginals for each site.
pub fn rank_and_marginalize(
    combinations: &[CandidateCombination],
    weights: RankWeights,
) -> Result<CombinationAnalysis> {
    if combinations.is_empty() {
        return Err(SaxsError::InvalidInput(
            "candidate analysis requires at least one combination".into(),
        ));
    }
    if combinations
        .iter()
        .any(|combo| !combo.features.chi2.is_finite() || combo.features.chi2 < 0.0)
    {
        return Err(SaxsError::InvalidInput(
            "candidate χ² values must be finite and non-negative".into(),
        ));
    }
    if [
        weights.chi2,
        weights.kratky,
        weights.rg,
        weights.dmax,
        weights.pr,
    ]
    .iter()
    .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(SaxsError::InvalidInput(
            "candidate rank weights must be finite and non-negative".into(),
        ));
    }
    let chi2_min = combinations
        .iter()
        .map(|combo| combo.features.chi2)
        .filter(|value| value.is_finite())
        .min_by(f64::total_cmp)
        .ok_or_else(|| SaxsError::InvalidInput("candidate χ² values are not finite".into()))?;
    let log_likelihoods = combinations
        .iter()
        .map(|combo| {
            if !combo.prior.is_finite() || combo.prior < 0.0 {
                return Err(SaxsError::InvalidInput(
                    "candidate priors must be finite and non-negative".into(),
                ));
            }
            let delta = (combo.features.chi2 - chi2_min).max(0.0);
            Ok(if combo.prior > 0.0 {
                combo.prior.ln() - 0.5 * delta
            } else {
                f64::NEG_INFINITY
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let maximum_log_likelihood = log_likelihoods
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !maximum_log_likelihood.is_finite() {
        return Err(SaxsError::InvalidInput(
            "candidate priors and likelihoods have zero total support".into(),
        ));
    }
    // Keep the relative likelihoods in a numerically stable [0, 1] range.
    // Ratios below the smallest positive f64 are retained at that floor so
    // reports say "effectively zero" instead of losing the row as 0.0.
    let likelihoods = log_likelihoods
        .iter()
        .map(|value| {
            if value.is_finite() {
                (*value - maximum_log_likelihood)
                    .exp()
                    .max(f64::MIN_POSITIVE)
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let likelihood_sum = likelihoods.iter().sum::<f64>();
    if !likelihood_sum.is_finite() || likelihood_sum <= 0.0 {
        return Err(SaxsError::InvalidInput(
            "candidate priors and likelihoods have zero total support".into(),
        ));
    }

    let rank_specs = [
        (
            "chi2",
            weights.chi2,
            combinations
                .iter()
                .map(|combo| Some(combo.features.chi2))
                .collect::<Vec<_>>(),
        ),
        (
            "kratky",
            weights.kratky,
            combinations
                .iter()
                .map(|combo| combo.features.kratky_deviation)
                .collect::<Vec<_>>(),
        ),
        (
            "rg_abs_error",
            weights.rg,
            combinations
                .iter()
                .map(|combo| combo.features.rg_abs_error)
                .collect::<Vec<_>>(),
        ),
        (
            "dmax_abs_error",
            weights.dmax,
            combinations
                .iter()
                .map(|combo| combo.features.dmax_abs_error)
                .collect::<Vec<_>>(),
        ),
        (
            "pr_rms",
            weights.pr,
            combinations
                .iter()
                .map(|combo| combo.features.pr_rms)
                .collect::<Vec<_>>(),
        ),
    ];
    let ranked_metrics = rank_specs
        .iter()
        .map(|(name, weight, values)| (*name, *weight, normalized_ranks(values)))
        .collect::<Vec<_>>();

    let mut rows = Vec::with_capacity(combinations.len());
    for (index, combo) in combinations.iter().enumerate() {
        let mut rank_components = BTreeMap::new();
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for (name, weight, ranks) in &ranked_metrics {
            if *weight <= 0.0 {
                continue;
            }
            if let Some(rank) = ranks[index] {
                let component = 1.0 - rank;
                rank_components.insert((*name).into(), component);
                numerator += weight * component;
                denominator += weight;
            }
        }
        rows.push(CombinationPosterior {
            assignments: combo.assignments.clone(),
            prior: combo.prior,
            log_likelihood: log_likelihoods[index]
                .is_finite()
                .then_some(log_likelihoods[index]),
            likelihood: likelihoods[index],
            posterior: likelihoods[index] / likelihood_sum,
            robust_score: if denominator > 0.0 {
                numerator / denominator
            } else {
                0.0
            },
            rank_components,
            features: combo.features.clone(),
        });
    }
    rows.sort_by(|left, right| {
        right
            .robust_score
            .total_cmp(&left.robust_score)
            .then_with(|| left.features.chi2.total_cmp(&right.features.chi2))
    });

    let mut site_occupancy = BTreeMap::new();
    let mut glycoform_distribution: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();
    for row in &rows {
        for assignment in &row.assignments {
            site_occupancy.entry(assignment.site.clone()).or_insert(0.0);
            if !is_none_candidate(&assignment.candidate) {
                *site_occupancy.entry(assignment.site.clone()).or_insert(0.0) += row.posterior;
            }
            *glycoform_distribution
                .entry(assignment.site.clone())
                .or_default()
                .entry(assignment.candidate.clone())
                .or_insert(0.0) += row.posterior;
        }
    }
    Ok(CombinationAnalysis {
        combinations: rows,
        site_occupancy,
        glycoform_distribution,
    })
}

fn normalized_ranks(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut finite = values
        .iter()
        .enumerate()
        .filter_map(|(index, value)| {
            value
                .filter(|value| value.is_finite())
                .map(|value| (index, value))
        })
        .collect::<Vec<_>>();
    finite.sort_by(|left, right| left.1.total_cmp(&right.1));
    let denominator = finite.len().saturating_sub(1).max(1) as f64;
    let mut result = vec![None; values.len()];
    for (rank, (index, _)) in finite.into_iter().enumerate() {
        result[index] = Some(rank as f64 / denominator);
    }
    result
}

fn is_none_candidate(candidate: &str) -> bool {
    matches!(
        candidate.trim().to_ascii_lowercase().as_str(),
        "none" | "no_glycan" | "noglycan" | "absent"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn combo(name: &str, chi2: f64, rg: f64) -> CandidateCombination {
        CandidateCombination {
            assignments: vec![Assignment {
                site: "A:1".into(),
                candidate: name.into(),
            }],
            prior: 1.0,
            features: SaxsFeatures {
                chi2,
                kratky_deviation: Some(chi2),
                rg_abs_error: Some(rg),
                dmax_abs_error: Some(chi2),
                pr_rms: Some(chi2),
                ..SaxsFeatures::default()
            },
        }
    }

    #[test]
    fn ranking_and_marginals_are_normalized() {
        let analysis = rank_and_marginalize(
            &[combo("glycan", 1.0, 0.1), combo("none", 4.0, 2.0)],
            RankWeights::default(),
        )
        .unwrap();
        assert!(analysis.site_occupancy["A:1"] > 0.0);
        assert!(analysis.site_occupancy["A:1"] < 1.0);
        let distribution = &analysis.glycoform_distribution["A:1"];
        assert!((distribution.values().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(analysis.combinations[0].robust_score >= analysis.combinations[1].robust_score);
    }

    #[test]
    fn likelihoods_remain_explicit_when_chi2_delta_underflows() {
        let analysis = rank_and_marginalize(
            &[combo("best", 1.0, 0.1), combo("remote", 5000.0, 2.0)],
            RankWeights::default(),
        )
        .unwrap();
        assert!(analysis.combinations.iter().all(|row| row.likelihood > 0.0));
        assert!(analysis
            .combinations
            .iter()
            .all(|row| row.log_likelihood.is_some_and(f64::is_finite)));
        assert!(
            (analysis
                .combinations
                .iter()
                .map(|row| row.posterior)
                .sum::<f64>()
                - 1.0)
                .abs()
                < 1.0e-12
        );
    }

    #[test]
    fn zero_prior_is_json_serializable_without_negative_infinity() {
        let mut excluded = combo("excluded", 2.0, 0.2);
        excluded.prior = 0.0;
        let analysis = rank_and_marginalize(
            &[combo("supported", 1.0, 0.1), excluded],
            RankWeights::default(),
        )
        .unwrap();
        assert!(analysis
            .combinations
            .iter()
            .any(|row| row.log_likelihood.is_none() && row.likelihood == 0.0));
        serde_json::to_string(&analysis).unwrap();
    }
}
