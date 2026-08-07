//! Debye reference scattering and the public computation boundary.

use crate::error::{Result, SaxsError};
use crate::formfactor::form_factor_for_atom;
use crate::solvent::SolventParams;
use crate::structure::Structure;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum HydrogenMode {
    Implicit,
    Explicit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScatteringMethod {
    Auto,
    Debye,
    Multipole,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MultipoleOptions {
    pub l_max: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeOptions {
    pub method: ScatteringMethod,
    pub hydrogen_mode: HydrogenMode,
    pub include_hetatm: bool,
    pub multipole: MultipoleOptions,
    pub solvent: Option<SolventParams>,
}

impl Default for ComputeOptions {
    fn default() -> Self {
        Self {
            method: ScatteringMethod::Auto,
            hydrogen_mode: HydrogenMode::Implicit,
            include_hetatm: false,
            multipole: MultipoleOptions::default(),
            solvent: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurveMetadata {
    pub method: String,
    pub atom_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScatteringCurve {
    pub q: Vec<f64>,
    pub intensity: Vec<f64>,
    pub metadata: CurveMetadata,
}

#[inline]
pub fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-8 {
        1.0
    } else {
        x.sin() / x
    }
}

fn validate_q(q_values: &[f64]) -> Result<()> {
    if q_values.iter().any(|q| !q.is_finite() || *q < 0.0)
        || q_values.windows(2).any(|pair| pair[1] < pair[0])
    {
        return Err(SaxsError::InvalidInput(
            "q values must be finite, non-negative, and non-decreasing".into(),
        ));
    }
    Ok(())
}

pub fn debye_intensity(structure: &Structure, q_values: &[f64]) -> Vec<f64> {
    debye_intensity_with_hydrogens(structure, q_values, true)
}

pub fn debye_intensity_with_hydrogens(
    structure: &Structure,
    q_values: &[f64],
    implicit_hydrogen: bool,
) -> Vec<f64> {
    let atoms = &structure.atoms;
    q_values
        .par_iter()
        .map(|&q| {
            let f: Vec<f64> = atoms
                .iter()
                .map(|a| form_factor_for_atom(a, implicit_hydrogen, q))
                .collect();
            let mut intensity = 0.0;
            for i in 0..atoms.len() {
                intensity += f[i] * f[i];
                for j in (i + 1)..atoms.len() {
                    let r = (atoms[i].pos - atoms[j].pos).norm();
                    intensity += 2.0 * f[i] * f[j] * sinc(q * r);
                }
            }
            intensity.max(0.0)
        })
        .collect()
}

pub fn compute_curve(
    structure: &Structure,
    q_values: &[f64],
    options: &ComputeOptions,
) -> Result<ScatteringCurve> {
    validate_q(q_values)?;
    if let Some(solvent) = options.solvent {
        if !solvent.c1.is_finite()
            || solvent.c1 <= 0.0
            || !solvent.c2.is_finite()
            || !solvent.shell_thickness_angstrom.is_finite()
            || solvent.shell_thickness_angstrom <= 0.0
            || !solvent.bulk_density_e_per_a3.is_finite()
            || solvent.bulk_density_e_per_a3 <= 0.0
            || !solvent.hydration_grid_spacing_angstrom.is_finite()
            || solvent.hydration_grid_spacing_angstrom <= 0.0
        {
            return Err(SaxsError::InvalidInput(
                "solvent parameters must be finite with positive dimensions, density, and c1"
                    .into(),
            ));
        }
    }
    if options
        .multipole
        .l_max
        .is_some_and(|order| !(8..=64).contains(&order))
    {
        return Err(SaxsError::InvalidInput(
            "multipole l_max must be between 8 and 64".into(),
        ));
    }
    let use_implicit = matches!(options.hydrogen_mode, HydrogenMode::Implicit);
    let method = match options.method {
        ScatteringMethod::Auto
            if options.solvent.as_ref().is_some_and(|solvent| {
                solvent.hydration_model != crate::solvent::HydrationModel::Analytic
            }) =>
        {
            ScatteringMethod::Multipole
        }
        ScatteringMethod::Auto if structure.atoms.len() < 500 => ScatteringMethod::Debye,
        ScatteringMethod::Auto | ScatteringMethod::Multipole => ScatteringMethod::Multipole,
        ScatteringMethod::Debye => ScatteringMethod::Debye,
    };
    let qmax = q_values.iter().copied().fold(0.0_f64, f64::max);
    let centre = structure
        .atoms
        .iter()
        .fold(nalgebra::Vector3::zeros(), |sum, a| sum + a.pos)
        / structure.atoms.len().max(1) as f64;
    let maximum_radius = structure
        .atoms
        .iter()
        .map(|atom| (atom.pos - centre).norm())
        .fold(0.0_f64, f64::max);
    let adaptive_lmax = options
        .multipole
        .l_max
        // The addition theorem is truncated only after q times the complete
        // molecular radius plus eight guard orders, then bounded to the
        // supported range. Per-q evaluation applies the same rule below this
        // maximum, so low-q points remain inexpensive.
        .unwrap_or_else(|| ((qmax * maximum_radius).ceil() as usize + 8).clamp(8, 64));
    let intensity = match method {
        ScatteringMethod::Debye => match options.solvent.as_ref() {
            None => debye_intensity_with_hydrogens(structure, q_values, use_implicit),
            Some(p) => {
                crate::solvent::solvent_corrected_intensity(structure, q_values, p, use_implicit)?
            }
        },
        ScatteringMethod::Multipole => match options.solvent.as_ref() {
            None => crate::spherical::multipole_intensity(
                structure,
                q_values,
                &crate::spherical::MultipoleParams {
                    l_max: adaptive_lmax,
                    hydrogen_mode: options.hydrogen_mode,
                },
            ),
            Some(p) => crate::spherical::multipole_solvent_intensity(
                structure,
                q_values,
                &crate::spherical::MultipoleParams {
                    l_max: adaptive_lmax,
                    hydrogen_mode: options.hydrogen_mode,
                },
                p,
            )?,
        },
        ScatteringMethod::Auto => unreachable!(),
    };
    Ok(ScatteringCurve {
        q: q_values.to_vec(),
        intensity,
        metadata: CurveMetadata {
            method: format!("{method:?}"),
            atom_count: structure.atoms.len(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::{Atom, Element};
    use nalgebra::Vector3;
    fn atoms() -> Structure {
        Structure {
            atoms: vec![
                Atom {
                    pos: Vector3::new(0.0, 0.0, 0.0),
                    element: Element::C,
                    occupancy: 1.0,
                    is_hetatm: false,
                    atom_name: "C".into(),
                    residue_name: "TST".into(),
                    chain_id: "A".into(),
                    residue_id: 1,
                },
                Atom {
                    pos: Vector3::new(2.0, 0.0, 0.0),
                    element: Element::C,
                    occupancy: 1.0,
                    is_hetatm: false,
                    atom_name: "C2".into(),
                    residue_name: "TST".into(),
                    chain_id: "A".into(),
                    residue_id: 1,
                },
            ],
        }
    }
    #[test]
    fn sinc_at_zero_is_one() {
        assert_eq!(sinc(0.0), 1.0);
    }
    #[test]
    fn two_atoms_match_closed_form() {
        let s = atoms();
        let q = 0.2;
        let got = debye_intensity(&s, &[q])[0];
        let f = crate::formfactor::form_factor_for_atom(&s.atoms[0], true, q);
        let expected = 2.0 * f * f * (1.0 + sinc(q * 2.0));
        assert!((got - expected).abs() < 1e-10);
    }
    #[test]
    fn rejects_bad_q() {
        assert!(compute_curve(&atoms(), &[f64::NAN], &ComputeOptions::default()).is_err());
        assert!(compute_curve(&atoms(), &[0.2, 0.1], &ComputeOptions::default()).is_err());
    }

    #[test]
    fn auto_solution_curve_uses_validated_multipole_path() {
        let curve = compute_curve(
            &atoms(),
            &[0.05, 0.1],
            &ComputeOptions {
                solvent: Some(SolventParams {
                    c2: 0.02,
                    hydration_model: crate::solvent::HydrationModel::AdaptiveGrid,
                    hydration_grid_spacing_angstrom: 8.0,
                    ..Default::default()
                }),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(curve.metadata.method, "Multipole");
        assert!(curve
            .intensity
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0));
    }

    #[test]
    fn debye_supports_adaptive_grid_solution_curve() {
        let curve = compute_curve(
            &atoms(),
            &[0.05, 0.1],
            &ComputeOptions {
                method: ScatteringMethod::Debye,
                solvent: Some(SolventParams {
                    c2: 0.02,
                    hydration_model: crate::solvent::HydrationModel::AdaptiveGrid,
                    hydration_grid_spacing_angstrom: 8.0,
                    ..Default::default()
                }),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(curve.metadata.method, "Debye");
        assert!(curve
            .intensity
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0));
    }

    #[test]
    fn adaptive_public_multipole_curve_matches_debye_below_one_tenth_percent() {
        let atoms = (0..96)
            .map(|index| {
                let angle = index as f64 * 2.399_963_229_728_653;
                let radial = 4.0 + (index % 13) as f64;
                Atom {
                    pos: Vector3::new(
                        radial * angle.cos(),
                        radial * angle.sin(),
                        (index as f64 - 47.5) * 0.31,
                    ),
                    element: match index % 5 {
                        0 => Element::C,
                        1 => Element::N,
                        2 => Element::O,
                        3 => Element::S,
                        _ => Element::P,
                    },
                    occupancy: 1.0,
                    is_hetatm: false,
                    atom_name: format!("A{index}"),
                    residue_name: "TST".into(),
                    chain_id: "A".into(),
                    residue_id: index as isize,
                }
            })
            .collect();
        let structure = Structure { atoms };
        let q = (1..=100)
            .map(|index| index as f64 * 0.005)
            .collect::<Vec<_>>();
        let multipole = compute_curve(
            &structure,
            &q,
            &ComputeOptions {
                method: ScatteringMethod::Multipole,
                ..Default::default()
            },
        )
        .unwrap()
        .intensity;
        let debye = debye_intensity(&structure, &q);
        let relative_l2 = (multipole
            .iter()
            .zip(&debye)
            .map(|(actual, expected)| (actual - expected).powi(2))
            .sum::<f64>()
            / debye.iter().map(|value| value * value).sum::<f64>())
        .sqrt();
        assert!(
            relative_l2 <= 0.001,
            "adaptive multipole/Debye error was {relative_l2}"
        );
    }
}
