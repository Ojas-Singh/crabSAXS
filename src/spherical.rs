//! Isotropic spherical-harmonic multipole expansion.
//!
//! The orientation average of the plane-wave expansion is
//! `I(q)=4π Σ_l Σ_m |Σ_i f_i(q) j_l(q r_i) Y_lm(Ω_i)|²`.
//! Coordinates are centred on the atomic centroid. The implementation uses
//! normalized real harmonics and stable recurrences for `P_l^m` and `j_l`.

use crate::error::Result;
use crate::formfactor::form_factor_for_atom;
use crate::scattering::HydrogenMode;
use crate::solvent::{
    build_adaptive_hydration_grid, build_hydration_grid, excluded_volume_for_atom,
    excluded_volume_scale, water_form_factor, AdaptiveHydrationOptions, HydrationModel,
    SolventGridOptions, SolventParams,
};
use crate::structure::{Atom, Structure};
use rayon::prelude::*;

pub struct MultipoleParams {
    pub l_max: usize,
    pub hydrogen_mode: HydrogenMode,
}

impl Default for MultipoleParams {
    fn default() -> Self {
        Self {
            l_max: 20,
            hydrogen_mode: HydrogenMode::Implicit,
        }
    }
}

fn factorial_ratio(l: usize, m: usize) -> f64 {
    let mut ratio = 1.0;
    for k in (l - m + 1)..=(l + m) {
        ratio /= k as f64;
    }
    ratio
}

fn harmonic_norms(l_max: usize) -> Vec<f64> {
    let mut norms = vec![0.0; (l_max + 1) * (l_max + 1)];
    for l in 0..=l_max {
        for m in 0..=l {
            norms[l * l + l + m] = (((2 * l + 1) as f64 / (4.0 * std::f64::consts::PI))
                * factorial_ratio(l, m))
            .sqrt();
        }
    }
    norms
}

/// Fill every normalized real harmonic using one Legendre recurrence.
fn real_harmonics_all(l_max: usize, x: f64, phi: f64, norms: &[f64], out: &mut [f64]) {
    out.fill(0.0);
    let sin_theta = (1.0 - x * x).max(0.0).sqrt();
    let (sin_phi, cos_phi) = phi.sin_cos();
    let (mut sin_m_phi, mut cos_m_phi) = (0.0, 1.0);
    let mut pmm = 1.0;
    for m in 0..=l_max {
        if m > 0 {
            pmm *= -((2 * m - 1) as f64) * sin_theta;
            let next_cos = cos_m_phi * cos_phi - sin_m_phi * sin_phi;
            sin_m_phi = sin_m_phi * cos_phi + cos_m_phi * sin_phi;
            cos_m_phi = next_cos;
        }

        let mut emit = |l: usize, p: f64| {
            let positive = l * l + l + m;
            if m == 0 {
                out[positive] = norms[positive] * p;
            } else {
                let value = 2.0f64.sqrt() * norms[positive] * p;
                out[positive] = value * cos_m_phi;
                out[l * l + l - m] = value * sin_m_phi;
            }
        };
        emit(m, pmm);
        if m == l_max {
            continue;
        }
        let pm1 = x * (2 * m + 1) as f64 * pmm;
        emit(m + 1, pm1);
        let (mut previous, mut current) = (pmm, pm1);
        for l in (m + 2)..=l_max {
            let next =
                ((2 * l - 1) as f64 * x * current - (l + m - 1) as f64 * previous) / (l - m) as f64;
            emit(l, next);
            previous = current;
            current = next;
        }
    }
}

fn spherical_bessel_all_into(j: &mut [f64], work: &mut [f64], x: f64) {
    j.fill(0.0);
    if x.abs() < 1e-8 {
        j[0] = 1.0;
        return;
    }
    if j.len() == 1 {
        j[0] = x.sin() / x;
        return;
    }

    // Miller's downward recurrence computes every order in O(l_max) and is
    // stable in the important x <= l region. The previous independent power
    // series was accurate but repeated the same work for every order and was
    // the dominant fitting cost.
    let l_max = j.len() - 1;
    let start = l_max + 24;
    debug_assert!(work.len() >= start + 2);
    work[..start + 2].fill(0.0);
    work[start] = 1.0;
    for l in (1..=start).rev() {
        work[l - 1] = (2 * l + 1) as f64 * work[l] / x - work[l + 1];
        if work[l - 1].abs() > 1e100 {
            for value in &mut work[l - 1..=start + 1] {
                *value *= 1e-100;
            }
        }
    }
    let exact_j0 = x.sin() / x;
    let exact_j1 = x.sin() / (x * x) - x.cos() / x;
    let scale = if exact_j0.abs() >= exact_j1.abs() {
        exact_j0 / work[0]
    } else {
        exact_j1 / work[1]
    };
    for (target, source) in j.iter_mut().zip(work.iter()) {
        *target = *source * scale;
    }
}

#[cfg(test)]
fn spherical_bessel_all(l_max: usize, x: f64) -> Vec<f64> {
    let mut values = vec![0.0; l_max + 1];
    let mut work = vec![0.0; l_max + 26];
    spherical_bessel_all_into(&mut values, &mut work, x);
    values
}

struct MultipoleGeometry {
    radii: Vec<f64>,
    /// Real spherical harmonics in atom-major, packed `(l, m)` order.
    harmonics: Vec<f64>,
    coefficient_count: usize,
    max_radius: f64,
}

fn multipole_geometry(structure: &Structure, l_max: usize) -> MultipoleGeometry {
    let centre = structure
        .atoms
        .iter()
        .fold(nalgebra::Vector3::zeros(), |sum, atom| sum + atom.pos)
        / structure.atoms.len() as f64;
    let positions = structure
        .atoms
        .iter()
        .map(|atom| atom.pos)
        .collect::<Vec<_>>();
    multipole_geometry_for_points(&positions, centre, l_max)
}

fn multipole_geometry_for_points(
    positions: &[nalgebra::Vector3<f64>],
    centre: nalgebra::Vector3<f64>,
    l_max: usize,
) -> MultipoleGeometry {
    let coefficient_count = (l_max + 1) * (l_max + 1);
    let radii: Vec<f64> = positions
        .par_iter()
        .map(|position| (position - centre).norm())
        .collect();
    let norms = harmonic_norms(l_max);
    let mut harmonics = vec![0.0; positions.len() * coefficient_count];
    harmonics
        .par_chunks_mut(coefficient_count)
        .zip(positions.par_iter())
        .zip(radii.par_iter())
        .for_each(|((out, point), &radius)| {
            let position = point - centre;
            let (cos_theta, phi) = if radius < 1e-12 {
                (1.0, 0.0)
            } else {
                (
                    (position.z / radius).clamp(-1.0, 1.0),
                    position.y.atan2(position.x),
                )
            };
            real_harmonics_all(l_max, cos_theta, phi, &norms, out);
        });
    MultipoleGeometry {
        max_radius: radii.iter().copied().fold(0.0, f64::max),
        radii,
        harmonics,
        coefficient_count,
    }
}

fn adaptive_l_max(q: f64, geometry: &MultipoleGeometry, maximum: usize) -> usize {
    ((q * geometry.max_radius).ceil() as usize + 8).clamp(maximum.min(8), maximum)
}

fn solvent_components_for_atom(
    atom: &Atom,
    q: f64,
    implicit_hydrogen: bool,
    solvent: &SolventParams,
) -> [f64; 3] {
    let volume = excluded_volume_for_atom(atom, implicit_hydrogen);
    let radius = (3.0 * volume / (4.0 * std::f64::consts::PI)).cbrt();
    let occupancy = atom.occupancy;
    let excluded = occupancy
        * solvent.bulk_density_e_per_a3
        * volume
        * (-volume.powf(2.0 / 3.0) * q * q / (4.0 * std::f64::consts::PI)).exp();
    let shell_radius = radius + solvent.shell_thickness_angstrom / 2.0;
    let shell_volume =
        4.0 * std::f64::consts::PI * shell_radius.powi(2) * solvent.shell_thickness_angstrom;
    let hydration = occupancy
        * solvent.bulk_density_e_per_a3
        * 0.1
        * shell_volume
        * (-q * q * shell_radius * shell_radius / 4.0).exp();
    [
        form_factor_for_atom(atom, implicit_hydrogen, q),
        excluded,
        hydration,
    ]
}

fn hydration_grid_geometry(
    structure: &Structure,
    l_max: usize,
    solvent: &SolventParams,
) -> Result<Option<(MultipoleGeometry, f64)>> {
    let grid = match solvent.hydration_model {
        HydrationModel::Analytic => return Ok(None),
        HydrationModel::Grid => build_hydration_grid(
            structure,
            &SolventGridOptions {
                shell_thickness_angstrom: solvent.shell_thickness_angstrom,
                ..Default::default()
            },
        )?,
        HydrationModel::AdaptiveGrid => build_adaptive_hydration_grid(
            structure,
            &AdaptiveHydrationOptions {
                spacing_angstrom: solvent.hydration_grid_spacing_angstrom,
                ..Default::default()
            },
        )?,
    };
    let centre = structure
        .atoms
        .iter()
        .fold(nalgebra::Vector3::zeros(), |sum, atom| sum + atom.pos)
        / structure.atoms.len() as f64;
    Ok(Some((
        multipole_geometry_for_points(&grid.sites, centre, l_max),
        grid.voxel_volume_a3,
    )))
}

pub fn multipole_intensity(
    structure: &Structure,
    q_values: &[f64],
    params: &MultipoleParams,
) -> Vec<f64> {
    if structure.atoms.is_empty() {
        return vec![0.0; q_values.len()];
    }
    let l_max = params.l_max;
    let coefficient_count = (l_max + 1) * (l_max + 1);
    // Avoid an O(N L²) resident harmonic cache for very large assemblies.
    // The streaming reduction computes each atom's angular terms once and
    // keeps only one coefficient matrix per Rayon worker.
    if structure.atoms.len() * coefficient_count * std::mem::size_of::<f64>() > 256 * 1024 * 1024 {
        return multipole_intensity_streaming(structure, q_values, params);
    }
    let geometry = multipole_geometry(structure, l_max);
    q_values
        .par_iter()
        .map(|&q| {
            let q_l_max = adaptive_l_max(q, &geometry, l_max);
            let mut coeff = vec![0.0; geometry.coefficient_count];
            let mut js = vec![0.0; q_l_max + 1];
            let mut bessel_work = vec![0.0; q_l_max + 26];
            for (atom_index, atom) in structure.atoms.iter().enumerate() {
                let f = form_factor_for_atom(
                    atom,
                    matches!(params.hydrogen_mode, HydrogenMode::Implicit),
                    q,
                );
                spherical_bessel_all_into(
                    &mut js,
                    &mut bessel_work,
                    q * geometry.radii[atom_index],
                );
                let harmonics = &geometry.harmonics[atom_index * geometry.coefficient_count
                    ..(atom_index + 1) * geometry.coefficient_count];
                for (l, &jl) in js.iter().enumerate() {
                    for k in l * l..(l + 1) * (l + 1) {
                        coeff[k] += f * jl * harmonics[k];
                    }
                }
            }
            (4.0 * std::f64::consts::PI * coeff.iter().map(|v| v * v).sum::<f64>()).max(0.0)
        })
        .collect()
}

fn multipole_intensity_streaming(
    structure: &Structure,
    q_values: &[f64],
    params: &MultipoleParams,
) -> Vec<f64> {
    let l_max = params.l_max;
    let coefficient_count = (l_max + 1) * (l_max + 1);
    let centre = structure
        .atoms
        .iter()
        .fold(nalgebra::Vector3::zeros(), |sum, atom| sum + atom.pos)
        / structure.atoms.len() as f64;
    let max_radius = structure
        .atoms
        .iter()
        .map(|atom| (atom.pos - centre).norm())
        .fold(0.0_f64, f64::max);
    let q_orders = q_values
        .iter()
        .map(|q| ((q * max_radius).ceil() as usize + 8).clamp(l_max.min(8), l_max))
        .collect::<Vec<_>>();
    let norms = harmonic_norms(l_max);
    let implicit_hydrogen = matches!(params.hydrogen_mode, HydrogenMode::Implicit);
    let state = structure
        .atoms
        .par_iter()
        .fold(
            || {
                (
                    vec![0.0; q_values.len() * coefficient_count],
                    vec![0.0; coefficient_count],
                    vec![0.0; l_max + 1],
                    vec![0.0; l_max + 26],
                )
            },
            |mut state, atom| {
                let position = atom.pos - centre;
                let radius = position.norm();
                let (cos_theta, phi) = if radius < 1e-12 {
                    (1.0, 0.0)
                } else {
                    (
                        (position.z / radius).clamp(-1.0, 1.0),
                        position.y.atan2(position.x),
                    )
                };
                real_harmonics_all(l_max, cos_theta, phi, &norms, &mut state.1);
                for (q_index, (&q, &q_l_max)) in q_values.iter().zip(&q_orders).enumerate() {
                    let form_factor = form_factor_for_atom(atom, implicit_hydrogen, q);
                    spherical_bessel_all_into(&mut state.2[..=q_l_max], &mut state.3, q * radius);
                    let row = &mut state.0
                        [q_index * coefficient_count..(q_index + 1) * coefficient_count];
                    for (l, &jl) in state.2[..=q_l_max].iter().enumerate() {
                        let range = l * l..(l + 1) * (l + 1);
                        row[range.clone()].iter_mut().zip(&state.1[range]).for_each(
                            |(coefficient, harmonic)| {
                                *coefficient += form_factor * jl * harmonic;
                            },
                        );
                    }
                }
                state
            },
        )
        .reduce(
            || {
                (
                    vec![0.0; q_values.len() * coefficient_count],
                    vec![0.0; coefficient_count],
                    vec![0.0; l_max + 1],
                    vec![0.0; l_max + 26],
                )
            },
            |mut left, right| {
                left.0
                    .iter_mut()
                    .zip(right.0)
                    .for_each(|(left, right)| *left += right);
                left
            },
        );
    state
        .0
        .chunks(coefficient_count)
        .map(|row| {
            (4.0 * std::f64::consts::PI * row.iter().map(|value| value * value).sum::<f64>())
                .max(0.0)
        })
        .collect()
}

/// Multipole evaluation with the three solvent amplitudes combined per atom.
/// This is used by fitting so each objective evaluation remains O(N·L²), not
/// an O(N²) Debye recomputation.
pub fn multipole_solvent_intensity(
    structure: &Structure,
    q_values: &[f64],
    params: &MultipoleParams,
    solvent: &SolventParams,
) -> Result<Vec<f64>> {
    if structure.atoms.is_empty() {
        return Ok(vec![0.0; q_values.len()]);
    }
    let l_max = params.l_max;
    let geometry = multipole_geometry(structure, l_max);
    let grid_geometry = hydration_grid_geometry(structure, l_max, solvent)?;
    Ok(q_values
        .par_iter()
        .map(|&q| {
            let q_l_max = grid_geometry.as_ref().map_or_else(
                || adaptive_l_max(q, &geometry, l_max),
                |(grid, _)| adaptive_l_max(q, &geometry, l_max).max(adaptive_l_max(q, grid, l_max)),
            );
            let mut coeff = vec![0.0; geometry.coefficient_count];
            let mut js = vec![0.0; q_l_max + 1];
            let mut bessel_work = vec![0.0; q_l_max + 26];
            for idx in 0..structure.atoms.len() {
                let atom = &structure.atoms[idx];
                let amplitude = solvent_components_for_atom(
                    atom,
                    q,
                    matches!(params.hydrogen_mode, HydrogenMode::Implicit),
                    solvent,
                );
                let f = amplitude[0] - excluded_volume_scale(q, solvent.c1) * amplitude[1]
                    + if solvent.hydration_model == HydrationModel::Analytic {
                        solvent.c2 * amplitude[2]
                    } else {
                        0.0
                    };
                spherical_bessel_all_into(&mut js, &mut bessel_work, q * geometry.radii[idx]);
                let harmonics = &geometry.harmonics
                    [idx * geometry.coefficient_count..(idx + 1) * geometry.coefficient_count];
                for (l, &jl) in js.iter().enumerate() {
                    for k in l * l..(l + 1) * (l + 1) {
                        coeff[k] += f * jl * harmonics[k];
                    }
                }
            }
            if let Some((grid, voxel_volume)) = &grid_geometry {
                let site_amplitude =
                    solvent.bulk_density_e_per_a3 / 10.0 * voxel_volume * water_form_factor(q);
                for idx in 0..grid.radii.len() {
                    spherical_bessel_all_into(&mut js, &mut bessel_work, q * grid.radii[idx]);
                    let harmonics = &grid.harmonics
                        [idx * grid.coefficient_count..(idx + 1) * grid.coefficient_count];
                    for (l, &jl) in js.iter().enumerate() {
                        for k in l * l..(l + 1) * (l + 1) {
                            coeff[k] += solvent.c2 * site_amplitude * jl * harmonics[k];
                        }
                    }
                }
            }
            (4.0 * std::f64::consts::PI * coeff.iter().map(|v| v * v).sum::<f64>()).max(0.0)
        })
        .collect())
}

/// Return the six quadratic intensity terms for vacuum, excluded volume and
/// hydration amplitudes: vv, ee, hh, ve, vh, eh.
pub fn multipole_solvent_basis(
    structure: &Structure,
    q_values: &[f64],
    params: &MultipoleParams,
    solvent: &SolventParams,
) -> Result<[Vec<f64>; 6]> {
    if structure.atoms.is_empty() {
        return Ok(std::array::from_fn(|_| vec![0.0; q_values.len()]));
    }
    let l_max = params.l_max;
    let geometry = multipole_geometry(structure, l_max);
    let grid_geometry = hydration_grid_geometry(structure, l_max, solvent)?;
    let rows: Vec<[f64; 6]> = q_values
        .par_iter()
        .map(|&q| {
            let q_l_max = grid_geometry.as_ref().map_or_else(
                || adaptive_l_max(q, &geometry, l_max),
                |(grid, _)| adaptive_l_max(q, &geometry, l_max).max(adaptive_l_max(q, grid, l_max)),
            );
            let mut coeff = [
                vec![0.0; geometry.coefficient_count],
                vec![0.0; geometry.coefficient_count],
                vec![0.0; geometry.coefficient_count],
            ];
            let mut js = vec![0.0; q_l_max + 1];
            let mut bessel_work = vec![0.0; q_l_max + 26];
            for idx in 0..structure.atoms.len() {
                let atom = &structure.atoms[idx];
                let amplitude = solvent_components_for_atom(
                    atom,
                    q,
                    matches!(params.hydrogen_mode, HydrogenMode::Implicit),
                    solvent,
                );
                spherical_bessel_all_into(&mut js, &mut bessel_work, q * geometry.radii[idx]);
                let harmonics = &geometry.harmonics
                    [idx * geometry.coefficient_count..(idx + 1) * geometry.coefficient_count];
                for (l, &jl) in js.iter().enumerate() {
                    for k in l * l..(l + 1) * (l + 1) {
                        let c = jl * harmonics[k];
                        coeff[0][k] += amplitude[0] * c;
                        coeff[1][k] += amplitude[1] * c;
                        if solvent.hydration_model == HydrationModel::Analytic {
                            coeff[2][k] += amplitude[2] * c;
                        }
                    }
                }
            }
            if let Some((grid, voxel_volume)) = &grid_geometry {
                let site_amplitude =
                    solvent.bulk_density_e_per_a3 / 10.0 * voxel_volume * water_form_factor(q);
                for idx in 0..grid.radii.len() {
                    spherical_bessel_all_into(&mut js, &mut bessel_work, q * grid.radii[idx]);
                    let harmonics = &grid.harmonics
                        [idx * grid.coefficient_count..(idx + 1) * grid.coefficient_count];
                    for (l, &jl) in js.iter().enumerate() {
                        for k in l * l..(l + 1) * (l + 1) {
                            coeff[2][k] += site_amplitude * jl * harmonics[k];
                        }
                    }
                }
            }
            let scale = 4.0 * std::f64::consts::PI;
            [
                scale * coeff[0].iter().map(|v| v * v).sum::<f64>(),
                scale * coeff[1].iter().map(|v| v * v).sum::<f64>(),
                scale * coeff[2].iter().map(|v| v * v).sum::<f64>(),
                scale
                    * coeff[0]
                        .iter()
                        .zip(&coeff[1])
                        .map(|(a, b)| a * b)
                        .sum::<f64>(),
                scale
                    * coeff[0]
                        .iter()
                        .zip(&coeff[2])
                        .map(|(a, b)| a * b)
                        .sum::<f64>(),
                scale
                    * coeff[1]
                        .iter()
                        .zip(&coeff[2])
                        .map(|(a, b)| a * b)
                        .sum::<f64>(),
            ]
        })
        .collect();
    Ok(std::array::from_fn(|k| {
        rows.iter().map(|row| row[k]).collect()
    }))
}

pub fn combine_solvent_basis(
    basis: &[Vec<f64>; 6],
    q_values: &[f64],
    c1: f64,
    c2: f64,
) -> Vec<f64> {
    assert_eq!(basis[0].len(), q_values.len());
    (0..basis[0].len())
        .map(|i| {
            let excluded_scale = excluded_volume_scale(q_values[i], c1);
            basis[0][i] + excluded_scale * excluded_scale * basis[1][i] + c2 * c2 * basis[2][i]
                - 2.0 * excluded_scale * basis[3][i]
                + 2.0 * c2 * basis[4][i]
                - 2.0 * excluded_scale * c2 * basis[5][i]
        })
        .map(|v| v.max(0.0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::{Atom, Element};
    use nalgebra::Vector3;
    #[test]
    fn bessel_zero_limit() {
        assert_eq!(spherical_bessel_all(4, 0.0), vec![1.0, 0.0, 0.0, 0.0, 0.0]);
    }
    #[test]
    fn multipole_matches_debye_for_two_atoms() {
        let s = Structure {
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
        };
        let q = vec![0.1, 0.2, 0.5];
        let a = multipole_intensity(
            &s,
            &q,
            &MultipoleParams {
                l_max: 32,
                hydrogen_mode: HydrogenMode::Implicit,
            },
        );
        let b = crate::scattering::debye_intensity(&s, &q);
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() / y < 1e-6, "{x} {y}");
        }
    }

    #[test]
    fn multipole_matches_debye_at_large_qr() {
        let s = Structure {
            atoms: vec![
                Atom {
                    pos: Vector3::new(-20.0, 0.0, 0.0),
                    element: Element::C,
                    occupancy: 1.0,
                    is_hetatm: false,
                    atom_name: "C1".into(),
                    residue_name: "TST".into(),
                    chain_id: "A".into(),
                    residue_id: 1,
                },
                Atom {
                    pos: Vector3::new(20.0, 0.0, 0.0),
                    element: Element::O,
                    occupancy: 1.0,
                    is_hetatm: false,
                    atom_name: "O1".into(),
                    residue_name: "TST".into(),
                    chain_id: "A".into(),
                    residue_id: 1,
                },
            ],
        };
        let q = vec![0.2, 0.35, 0.5];
        let multipole = multipole_intensity(
            &s,
            &q,
            &MultipoleParams {
                l_max: 32,
                hydrogen_mode: HydrogenMode::Implicit,
            },
        );
        let debye = crate::scattering::debye_intensity(&s, &q);
        for (actual, expected) in multipole.iter().zip(debye) {
            assert!(
                (actual - expected).abs() / expected < 1e-6,
                "{actual} != {expected}"
            );
        }
    }

    #[test]
    fn multipole_curve_matches_debye_below_one_tenth_percent() {
        let atoms = (0..24)
            .map(|index| {
                let angle = index as f64 * 0.71;
                Atom {
                    pos: Vector3::new(
                        7.0 * angle.cos(),
                        7.0 * angle.sin(),
                        index as f64 * 0.45 - 5.0,
                    ),
                    element: match index % 4 {
                        0 => Element::C,
                        1 => Element::N,
                        2 => Element::O,
                        _ => Element::S,
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
        let multipole = multipole_intensity(
            &structure,
            &q,
            &MultipoleParams {
                l_max: 24,
                hydrogen_mode: HydrogenMode::Implicit,
            },
        );
        let debye = crate::scattering::debye_intensity(&structure, &q);
        let nrmse = (multipole
            .iter()
            .zip(&debye)
            .map(|(actual, expected)| (actual - expected).powi(2))
            .sum::<f64>()
            / debye.iter().map(|value| value * value).sum::<f64>())
        .sqrt();
        assert!(nrmse <= 0.001, "multipole/Debye NRMSE was {nrmse}");
    }

    #[test]
    fn adaptive_grid_solvent_basis_matches_debye_oracle() {
        let structure = Structure {
            atoms: (0..8)
                .map(|index| Atom {
                    pos: Vector3::new(
                        index as f64 * 1.4 - 4.9,
                        (index as f64 * 0.9).sin() * 2.0,
                        (index as f64 * 0.7).cos(),
                    ),
                    element: match index % 4 {
                        0 => Element::C,
                        1 => Element::N,
                        2 => Element::O,
                        _ => Element::S,
                    },
                    occupancy: 1.0,
                    is_hetatm: false,
                    atom_name: format!("A{index}"),
                    residue_name: "TST".into(),
                    chain_id: "A".into(),
                    residue_id: index as isize,
                })
                .collect(),
        };
        let q = (1..=50)
            .map(|index| index as f64 * 0.01)
            .collect::<Vec<_>>();
        let solvent = SolventParams {
            hydration_model: HydrationModel::AdaptiveGrid,
            hydration_grid_spacing_angstrom: 8.0,
            ..Default::default()
        };
        let multipole = multipole_solvent_basis(
            &structure,
            &q,
            &MultipoleParams {
                l_max: 32,
                hydrogen_mode: HydrogenMode::Implicit,
            },
            &solvent,
        )
        .unwrap();
        let debye = crate::solvent::debye_solvent_basis(&structure, &q, &solvent, true).unwrap();
        for component in 0..6 {
            let error = (multipole[component]
                .iter()
                .zip(&debye[component])
                .map(|(actual, expected)| (actual - expected).powi(2))
                .sum::<f64>()
                / debye[component]
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .max(1e-30))
            .sqrt();
            assert!(error <= 0.001, "component {component} error was {error}");
        }
    }
}
