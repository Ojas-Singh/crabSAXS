//! Excluded-volume and hydration-shell contrast model.
//!
//! The parity default uses Gaussian spheres centred on input atoms. An
//! explicit deterministic exterior voxel shell is also available through
//! `HydrationModel::Grid`; both share the same six-term fit parameterization.

use crate::error::{Result, SaxsError};
use crate::formfactor::{form_factor_for_atom, implicit_hydrogen_count_for_atom};
use crate::structure::{Element, Structure};
use nalgebra::Vector3;
use rayon::prelude::*;
use std::collections::VecDeque;

/// Mean united-atom radius used by the published global Fraser scaling.
pub const MEAN_ATOMIC_RADIUS_ANGSTROM: f64 = 1.62;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum HydrationModel {
    /// Fast atom-centred Gaussian shell used by the parity-tuned default.
    Analytic,
    /// Deterministic solvent-accessible voxel shell.
    Grid,
    /// Adaptive coarse grid from Grudinin et al. (2017), section 2.3.
    AdaptiveGrid,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct SolventParams {
    pub c1: f64,
    pub c2: f64,
    pub shell_thickness_angstrom: f64,
    pub bulk_density_e_per_a3: f64,
    pub hydration_model: HydrationModel,
    /// Linear spacing for the adaptive hydration grid. The published normal
    /// mode uses 3-4 Å; Pepsi-style fast mode doubles this spacing.
    pub hydration_grid_spacing_angstrom: f64,
}

impl Default for SolventParams {
    fn default() -> Self {
        Self {
            c1: 1.0,
            c2: 0.0,
            shell_thickness_angstrom: 3.0,
            bulk_density_e_per_a3: 0.334,
            hydration_model: HydrationModel::Analytic,
            hydration_grid_spacing_angstrom: 4.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SolventGridOptions {
    pub spacing_angstrom: f64,
    pub probe_radius_angstrom: f64,
    pub shell_thickness_angstrom: f64,
}

impl Default for SolventGridOptions {
    fn default() -> Self {
        Self {
            spacing_angstrom: 1.0,
            probe_radius_angstrom: 1.4,
            shell_thickness_angstrom: 3.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HydrationGrid {
    pub sites: Vec<Vector3<f64>>,
    pub voxel_volume_a3: f64,
}

/// Pepsi-SAXS paper section 2.3 grid parameters. A 4 Å grid corresponds to
/// the documented coarse (`-fast`) representation used by the parity harness.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveHydrationOptions {
    pub spacing_angstrom: f64,
    pub molecular_clearance_angstrom: f64,
    pub padding_angstrom: f64,
}

impl Default for AdaptiveHydrationOptions {
    fn default() -> Self {
        Self {
            spacing_angstrom: 4.0,
            molecular_clearance_angstrom: 3.0,
            padding_angstrom: 12.0,
        }
    }
}

/// Linear 3–5 Å shell-width rule from Grudinin et al. (2017), section 2.3:
/// 3 Å at `Rg <= 15 Å`, 5 Å at `Rg >= 20 Å`, interpolated in between.
pub fn adaptive_shell_width(radius_of_gyration: f64) -> f64 {
    if radius_of_gyration <= 15.0 {
        3.0
    } else if radius_of_gyration >= 20.0 {
        5.0
    } else {
        3.0 + 0.4 * (radius_of_gyration - 15.0)
    }
}

fn adaptive_site_volume(spacing: f64) -> f64 {
    if spacing > 4.0 {
        // Published fast mode halves the sampling-point density. Its doubled
        // linear spacing changes positions, while each retained point carries
        // twice the normal-grid volume rather than a full 2^3 voxel volume.
        2.0 * (spacing / 2.0).powi(3)
    } else {
        spacing.powi(3)
    }
}

/// Build the published coarse hydration shell: retain grid-cell centres whose
/// nearest atomic centre is between 3 Å and `3 Å + shell_width`. Unlike the
/// solvent-accessible grid, this model intentionally includes enclosed voids,
/// matching the method described in the Pepsi-SAXS paper.
pub fn build_adaptive_hydration_grid(
    structure: &Structure,
    options: &AdaptiveHydrationOptions,
) -> Result<HydrationGrid> {
    if structure.atoms.is_empty() {
        return Ok(HydrationGrid {
            sites: Vec::new(),
            voxel_volume_a3: adaptive_site_volume(options.spacing_angstrom),
        });
    }
    let spacing = options.spacing_angstrom;
    if !spacing.is_finite()
        || spacing <= 0.0
        || !options.molecular_clearance_angstrom.is_finite()
        || options.molecular_clearance_angstrom <= 0.0
        || !options.padding_angstrom.is_finite()
        || options.padding_angstrom < 0.0
    {
        return Err(SaxsError::InvalidInput(
            "adaptive hydration-grid dimensions must be finite and positive".into(),
        ));
    }

    let centre = structure
        .atoms
        .iter()
        .fold(Vector3::zeros(), |sum, atom| sum + atom.pos)
        / structure.atoms.len() as f64;
    let radius_of_gyration = (structure
        .atoms
        .iter()
        .map(|atom| (atom.pos - centre).norm_squared())
        .sum::<f64>()
        / structure.atoms.len() as f64)
        .sqrt();
    let shell_width = adaptive_shell_width(radius_of_gyration);
    let inner_radius = options.molecular_clearance_angstrom;
    let outer_radius = inner_radius + shell_width;

    let mut minimum = structure.atoms[0].pos;
    let mut maximum = structure.atoms[0].pos;
    for atom in &structure.atoms[1..] {
        minimum = minimum.zip_map(&atom.pos, f64::min);
        maximum = maximum.zip_map(&atom.pos, f64::max);
    }
    minimum -= Vector3::repeat(options.padding_angstrom);
    maximum += Vector3::repeat(options.padding_angstrom);
    let dimensions = [
        (((maximum.x - minimum.x) / spacing) - 1e-10).ceil() as usize + 1,
        (((maximum.y - minimum.y) / spacing) - 1e-10).ceil() as usize + 1,
        (((maximum.z - minimum.z) / spacing) - 1e-10).ceil() as usize + 1,
    ];
    let voxel_count = dimensions
        .into_iter()
        .try_fold(1usize, |product, size| product.checked_mul(size))
        .ok_or_else(|| SaxsError::InvalidInput("adaptive hydration grid is too large".into()))?;
    let [nx, ny, nz] = dimensions;
    let index = |x: usize, y: usize, z: usize| (z * ny + y) * nx + x;
    let point = |x: usize, y: usize, z: usize| {
        minimum + Vector3::new(x as f64 * spacing, y as f64 * spacing, z as f64 * spacing)
    };
    let mut shell = vec![false; voxel_count];
    let mut molecular = vec![false; voxel_count];
    let rasterize = |radius: f64, mask: &mut [bool]| {
        let radius_squared = radius * radius;
        for atom in &structure.atoms {
            let lower = (atom.pos - Vector3::repeat(radius) - minimum) / spacing;
            let upper = (atom.pos + Vector3::repeat(radius) - minimum) / spacing;
            let x0 = (lower.x - 1e-10).floor().max(0.0) as usize;
            let y0 = (lower.y - 1e-10).floor().max(0.0) as usize;
            let z0 = (lower.z - 1e-10).floor().max(0.0) as usize;
            let x1 = ((upper.x + 1e-10).ceil() as usize).min(nx - 1);
            let y1 = ((upper.y + 1e-10).ceil() as usize).min(ny - 1);
            let z1 = ((upper.z + 1e-10).ceil() as usize).min(nz - 1);
            for z in z0..=z1 {
                for y in y0..=y1 {
                    for x in x0..=x1 {
                        if (point(x, y, z) - atom.pos).norm_squared() <= radius_squared + 1e-10 {
                            mask[index(x, y, z)] = true;
                        }
                    }
                }
            }
        }
    };
    rasterize(outer_radius, &mut shell);
    rasterize(inner_radius, &mut molecular);
    let mut sites = Vec::new();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let cell = index(x, y, z);
                if shell[cell] && !molecular[cell] {
                    sites.push(point(x, y, z));
                }
            }
        }
    }
    Ok(HydrationGrid {
        sites,
        voxel_volume_a3: adaptive_site_volume(spacing),
    })
}

/// Orientationally averaged water-molecule amplitude, using the same
/// united-group approximation as equation (8) of Grudinin et al. (2017).
pub fn water_form_factor(q: f64) -> f64 {
    let oxygen = crate::formfactor::coefficients_for(Element::O).evaluate(q);
    let hydrogen = crate::formfactor::coefficients_for(Element::H).evaluate(q);
    oxygen + 2.0 * hydrogen * crate::scattering::sinc(q * 0.9572)
}

/// Bondi-style van der Waals radii in Å, with conventional biological-ion
/// extensions. See `docs/REFERENCES.md` for provenance.
pub fn van_der_waals_radius_for(element: Element) -> f64 {
    match element {
        Element::H => 1.20,
        Element::C => 1.70,
        Element::N => 1.55,
        Element::O => 1.52,
        Element::S => 1.80,
        Element::P => 1.80,
        Element::Na => 2.27,
        Element::Mg => 1.73,
        Element::Cl => 1.75,
        Element::Ca => 2.31,
        Element::Fe => 1.72,
        Element::Zn => 1.39,
        Element::K => 2.75,
        Element::I => 1.98,
        Element::Other => 1.70,
    }
}

/// Build a deterministic exterior hydration shell. Atomic van der Waals
/// spheres expanded by a 1.4 Å probe form the inaccessible volume; a
/// six-connected boundary flood fill identifies exterior solvent, and the
/// next 3 Å of exterior voxels form the hydration layer.
pub fn build_hydration_grid(
    structure: &Structure,
    options: &SolventGridOptions,
) -> Result<HydrationGrid> {
    if structure.atoms.is_empty() {
        return Ok(HydrationGrid {
            sites: Vec::new(),
            voxel_volume_a3: options.spacing_angstrom.powi(3),
        });
    }
    let spacing = options.spacing_angstrom;
    if !spacing.is_finite()
        || spacing <= 0.0
        || !options.probe_radius_angstrom.is_finite()
        || options.probe_radius_angstrom < 0.0
        || !options.shell_thickness_angstrom.is_finite()
        || options.shell_thickness_angstrom <= 0.0
    {
        return Err(SaxsError::InvalidInput(
            "solvent-grid dimensions must be finite and positive".into(),
        ));
    }

    let margin = structure
        .atoms
        .iter()
        .map(|atom| van_der_waals_radius_for(atom.element))
        .fold(0.0, f64::max)
        + options.probe_radius_angstrom
        + options.shell_thickness_angstrom
        + spacing;
    let mut minimum = structure.atoms[0].pos;
    let mut maximum = structure.atoms[0].pos;
    for atom in &structure.atoms[1..] {
        minimum = minimum.zip_map(&atom.pos, f64::min);
        maximum = maximum.zip_map(&atom.pos, f64::max);
    }
    minimum -= Vector3::repeat(margin);
    maximum += Vector3::repeat(margin);
    let dimensions = [
        (((maximum.x - minimum.x) / spacing) - 1e-10).ceil() as usize + 1,
        (((maximum.y - minimum.y) / spacing) - 1e-10).ceil() as usize + 1,
        (((maximum.z - minimum.z) / spacing) - 1e-10).ceil() as usize + 1,
    ];
    let voxel_count = dimensions
        .into_iter()
        .try_fold(1usize, |product, size| product.checked_mul(size))
        .ok_or_else(|| SaxsError::InvalidInput("solvent grid is too large".into()))?;
    if voxel_count > 250_000_000 {
        return Err(SaxsError::InvalidInput(format!(
            "solvent grid requires {voxel_count} voxels; limit is 250000000"
        )));
    }
    let [nx, ny, nz] = dimensions;
    let index = |x: usize, y: usize, z: usize| (z * ny + y) * nx + x;
    let point = |x: usize, y: usize, z: usize| {
        minimum + Vector3::new(x as f64 * spacing, y as f64 * spacing, z as f64 * spacing)
    };
    let mut state = vec![0u8; voxel_count]; // 0 unknown, 1 molecular, 2 exterior

    let rasterize = |radius_extra: f64, mask: &mut [u8], value: u8, exterior_only: bool| {
        for atom in &structure.atoms {
            let radius = van_der_waals_radius_for(atom.element) + radius_extra;
            let lower = (atom.pos - Vector3::repeat(radius) - minimum) / spacing;
            let upper = (atom.pos + Vector3::repeat(radius) - minimum) / spacing;
            let x0 = (lower.x - 1e-10).floor().max(0.0) as usize;
            let y0 = (lower.y - 1e-10).floor().max(0.0) as usize;
            let z0 = (lower.z - 1e-10).floor().max(0.0) as usize;
            let x1 = ((upper.x + 1e-10).ceil() as usize).min(nx - 1);
            let y1 = ((upper.y + 1e-10).ceil() as usize).min(ny - 1);
            let z1 = ((upper.z + 1e-10).ceil() as usize).min(nz - 1);
            let radius2 = radius * radius;
            for z in z0..=z1 {
                for y in y0..=y1 {
                    for x in x0..=x1 {
                        let cell = index(x, y, z);
                        if (!exterior_only || mask[cell] == 2)
                            && (point(x, y, z) - atom.pos).norm_squared() <= radius2 + 1e-10
                        {
                            mask[cell] = value;
                        }
                    }
                }
            }
        }
    };
    rasterize(options.probe_radius_angstrom, &mut state, 1, false);

    let mut queue = VecDeque::new();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if x != 0 && x + 1 != nx && y != 0 && y + 1 != ny && z != 0 && z + 1 != nz {
                    continue;
                }
                let cell = index(x, y, z);
                if state[cell] == 0 {
                    state[cell] = 2;
                    queue.push_back((x, y, z));
                }
            }
        }
    }
    while let Some((x, y, z)) = queue.pop_front() {
        let neighbours = [
            x.checked_sub(1).map(|next| (next, y, z)),
            (x + 1 < nx).then_some((x + 1, y, z)),
            y.checked_sub(1).map(|next| (x, next, z)),
            (y + 1 < ny).then_some((x, y + 1, z)),
            z.checked_sub(1).map(|next| (x, y, next)),
            (z + 1 < nz).then_some((x, y, z + 1)),
        ];
        for (next_x, next_y, next_z) in neighbours.into_iter().flatten() {
            let cell = index(next_x, next_y, next_z);
            if state[cell] == 0 {
                state[cell] = 2;
                queue.push_back((next_x, next_y, next_z));
            }
        }
    }

    // Mark exterior voxels inside the probe-expanded surface plus the shell.
    rasterize(
        options.probe_radius_angstrom + options.shell_thickness_angstrom,
        &mut state,
        3,
        true,
    );
    let mut sites = Vec::new();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if state[index(x, y, z)] == 3 {
                    sites.push(point(x, y, z));
                }
            }
        }
    }
    Ok(HydrationGrid {
        sites,
        voxel_volume_a3: spacing.powi(3),
    })
}

/// Displaced solvent volumes in Å³. H/C/N/O are the Fraser values; P and the
/// supported biological ions use the explicitly sourced radius-derived table
/// documented in `docs/REFERENCES.md`.
pub fn excluded_volume_for(element: Element) -> f64 {
    match element {
        Element::H => 5.15,
        Element::C => 16.44,
        Element::N => 2.49,
        Element::O => 9.13,
        Element::S => 26.09,
        Element::P => 3.37,
        Element::Na => 4.45,
        Element::Mg => 1.56,
        Element::Cl => 24.84,
        Element::Ca => 4.19,
        Element::Fe => 7.99,
        Element::Zn => 9.85,
        Element::K => 11.01,
        Element::I => 44.60,
        Element::Other => 0.0,
    }
}

/// Fraser excluded volume for a united atom in Å³. The published elemental
/// volumes are extended by 5.15 Å³ per covalently attached implicit H atom.
pub fn excluded_volume_for_atom(atom: &crate::structure::Atom, implicit_hydrogen: bool) -> f64 {
    if implicit_hydrogen && atom.element == Element::H {
        return 0.0;
    }
    excluded_volume_for(atom.element)
        + if implicit_hydrogen {
            implicit_hydrogen_count_for_atom(atom) * excluded_volume_for(Element::H)
        } else {
            0.0
        }
}

/// Corrected Fraser Gaussian-sphere form factor using q in Å⁻¹.
fn fraser_gaussian(q: f64, volume: f64) -> f64 {
    volume * (-volume.powf(2.0 / 3.0) * q * q / (4.0 * std::f64::consts::PI)).exp()
}

/// Global excluded-volume radius scaling used by CRYSOL and Pepsi-SAXS.
/// `radius_scale=1` is the tabulated reference volume. The common-radius
/// approximation uses the published mean united-atom radius of 1.62 Å.
pub fn excluded_volume_scale(q: f64, radius_scale: f64) -> f64 {
    radius_scale.powi(3)
        * (-(radius_scale * radius_scale - 1.0) * MEAN_ATOMIC_RADIUS_ANGSTROM.powi(2) * q * q / 4.0)
            .exp()
}

pub fn component_amplitudes(
    structure: &Structure,
    q: f64,
    implicit_hydrogen: bool,
    params: &SolventParams,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    structure
        .atoms
        .iter()
        .map(|atom| {
            let vacuum = form_factor_for_atom(atom, implicit_hydrogen, q);
            let volume = excluded_volume_for_atom(atom, implicit_hydrogen);
            let radius = (3.0 * volume / (4.0 * std::f64::consts::PI)).cbrt();
            let excluded =
                atom.occupancy * params.bulk_density_e_per_a3 * fraser_gaussian(q, volume);
            let shell_volume = 4.0
                * std::f64::consts::PI
                * (radius + params.shell_thickness_angstrom / 2.0).powi(2)
                * params.shell_thickness_angstrom;
            let shell = atom.occupancy
                * params.bulk_density_e_per_a3
                * 0.1
                * fraser_gaussian(q, shell_volume);
            (vacuum, excluded, shell)
        })
        .fold(
            (Vec::new(), Vec::new(), Vec::new()),
            |mut out, (v, e, h)| {
                out.0.push(v);
                out.1.push(e);
                out.2.push(h);
                out
            },
        )
}

pub fn solvent_corrected_intensity(
    structure: &Structure,
    q_values: &[f64],
    params: &SolventParams,
    implicit_hydrogen: bool,
) -> Result<Vec<f64>> {
    let basis = debye_solvent_basis(structure, q_values, params, implicit_hydrogen)?;
    Ok(crate::spherical::combine_solvent_basis(
        &basis, q_values, params.c1, params.c2,
    ))
}

/// Return the six Debye self/cross terms for vacuum, excluded volume, and
/// hydration amplitudes: `vv`, `ee`, `hh`, `ve`, `vh`, and `eh`.
pub fn debye_solvent_basis(
    structure: &Structure,
    q_values: &[f64],
    params: &SolventParams,
    implicit_hydrogen: bool,
) -> Result<[Vec<f64>; 6]> {
    let grid = match params.hydration_model {
        HydrationModel::Analytic => None,
        HydrationModel::Grid => Some(build_hydration_grid(
            structure,
            &SolventGridOptions {
                shell_thickness_angstrom: params.shell_thickness_angstrom,
                ..Default::default()
            },
        )?),
        HydrationModel::AdaptiveGrid => Some(build_adaptive_hydration_grid(
            structure,
            &AdaptiveHydrationOptions {
                spacing_angstrom: params.hydration_grid_spacing_angstrom,
                ..Default::default()
            },
        )?),
    };
    let rows: Vec<[f64; 6]> = q_values
        .par_iter()
        .map(|&q| {
            let (vacuum, excluded, analytic_hydration) =
                component_amplitudes(structure, q, implicit_hydrogen, params);
            let mut terms = [0.0; 6];
            for i in 0..structure.atoms.len() {
                terms[0] += vacuum[i] * vacuum[i];
                terms[1] += excluded[i] * excluded[i];
                terms[3] += vacuum[i] * excluded[i];
                if grid.is_none() {
                    terms[2] += analytic_hydration[i] * analytic_hydration[i];
                    terms[4] += vacuum[i] * analytic_hydration[i];
                    terms[5] += excluded[i] * analytic_hydration[i];
                }
                for j in (i + 1)..structure.atoms.len() {
                    let distance = (structure.atoms[i].pos - structure.atoms[j].pos).norm();
                    let sinc = crate::scattering::sinc(q * distance);
                    terms[0] += 2.0 * vacuum[i] * vacuum[j] * sinc;
                    terms[1] += 2.0 * excluded[i] * excluded[j] * sinc;
                    terms[3] += (vacuum[i] * excluded[j] + vacuum[j] * excluded[i]) * sinc;
                    if grid.is_none() {
                        terms[2] += 2.0 * analytic_hydration[i] * analytic_hydration[j] * sinc;
                        terms[4] += (vacuum[i] * analytic_hydration[j]
                            + vacuum[j] * analytic_hydration[i])
                            * sinc;
                        terms[5] += (excluded[i] * analytic_hydration[j]
                            + excluded[j] * analytic_hydration[i])
                            * sinc;
                    }
                }
            }
            if let Some(grid) = &grid {
                let site_amplitude = params.bulk_density_e_per_a3 / 10.0
                    * grid.voxel_volume_a3
                    * water_form_factor(q);
                for (i, site) in grid.sites.iter().enumerate() {
                    terms[2] += site_amplitude * site_amplitude;
                    for other in &grid.sites[(i + 1)..] {
                        terms[2] += 2.0
                            * site_amplitude
                            * site_amplitude
                            * crate::scattering::sinc(q * (site - other).norm());
                    }
                    for (atom_index, atom) in structure.atoms.iter().enumerate() {
                        let sinc = crate::scattering::sinc(q * (atom.pos - site).norm());
                        terms[4] += vacuum[atom_index] * site_amplitude * sinc;
                        terms[5] += excluded[atom_index] * site_amplitude * sinc;
                    }
                }
            }
            terms
        })
        .collect();
    Ok(std::array::from_fn(|term| {
        rows.iter().map(|row| row[term]).collect()
    }))
}

pub fn excluded_solvent_contribution(
    structure: &Structure,
    q_values: &[f64],
    params: &SolventParams,
) -> Vec<f64> {
    q_values
        .par_iter()
        .map(|&q| {
            component_amplitudes(structure, q, false, params)
                .1
                .into_iter()
                .sum()
        })
        .collect()
}

pub fn hydration_shell_contribution(
    structure: &Structure,
    q_values: &[f64],
    params: &SolventParams,
) -> Vec<f64> {
    q_values
        .par_iter()
        .map(|&q| {
            component_amplitudes(structure, q, false, params)
                .2
                .into_iter()
                .sum()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::Atom;

    fn two_atom_structure() -> Structure {
        Structure {
            atoms: vec![
                Atom {
                    pos: Vector3::new(-1.0, 0.0, 0.0),
                    element: Element::C,
                    occupancy: 1.0,
                    is_hetatm: false,
                    atom_name: "C".into(),
                    residue_name: "TST".into(),
                    chain_id: "A".into(),
                    residue_id: 1,
                },
                Atom {
                    pos: Vector3::new(1.0, 0.0, 0.0),
                    element: Element::O,
                    occupancy: 1.0,
                    is_hetatm: false,
                    atom_name: "O".into(),
                    residue_name: "TST".into(),
                    chain_id: "A".into(),
                    residue_id: 1,
                },
            ],
        }
    }

    #[test]
    fn solvent_basis_recombines_to_direct_debye_curve() {
        let structure = two_atom_structure();
        let q = [0.0, 0.1, 0.3, 0.5];
        let params = SolventParams {
            c1: 1.04,
            c2: 0.35,
            ..Default::default()
        };
        let basis = debye_solvent_basis(&structure, &q, &params, true).unwrap();
        let combined = crate::spherical::combine_solvent_basis(&basis, &q, params.c1, params.c2);
        let direct = solvent_corrected_intensity(&structure, &q, &params, true).unwrap();
        for (actual, expected) in combined.iter().zip(direct) {
            assert!((actual - expected).abs() <= expected.abs().max(1.0) * 1e-12);
        }
    }

    #[test]
    fn vacuum_minus_solvent_is_finite_for_structure_fixture() {
        let structure = Structure::from_pdb_file("examples/data/test_structure.pdb").unwrap();
        let q = (0..=100)
            .map(|index| index as f64 * 0.005)
            .collect::<Vec<_>>();
        let curve = solvent_corrected_intensity(
            &structure,
            &q,
            &SolventParams {
                c1: 1.0,
                c2: 0.0,
                ..Default::default()
            },
            true,
        )
        .unwrap();
        assert!(curve.iter().all(|value| value.is_finite() && *value >= 0.0));
        assert!(curve[0] > 0.0);
    }

    #[test]
    fn hydration_grid_is_deterministic_and_exterior() {
        let structure = Structure {
            atoms: vec![two_atom_structure().atoms.into_iter().next().unwrap()],
        };
        let options = SolventGridOptions::default();
        let first = build_hydration_grid(&structure, &options).unwrap();
        let second = build_hydration_grid(&structure, &options).unwrap();
        assert!(!first.sites.is_empty());
        assert_eq!(first.sites, second.sites);
        let exclusion = van_der_waals_radius_for(Element::C) + options.probe_radius_angstrom;
        let outer = exclusion + options.shell_thickness_angstrom;
        for site in &first.sites {
            let distance = (site - structure.atoms[0].pos).norm();
            assert!(distance > exclusion);
            assert!(distance <= outer + 1e-12);
        }
    }

    #[test]
    fn adaptive_shell_width_matches_published_piecewise_rule() {
        assert_eq!(adaptive_shell_width(10.0), 3.0);
        assert_eq!(adaptive_shell_width(15.0), 3.0);
        assert!((adaptive_shell_width(17.5) - 4.0).abs() < f64::EPSILON);
        assert_eq!(adaptive_shell_width(20.0), 5.0);
        assert_eq!(adaptive_shell_width(30.0), 5.0);
    }

    #[test]
    fn adaptive_hydration_grid_is_deterministic() {
        let structure = two_atom_structure();
        let options = AdaptiveHydrationOptions::default();
        let first = build_adaptive_hydration_grid(&structure, &options).unwrap();
        let second = build_adaptive_hydration_grid(&structure, &options).unwrap();
        assert!(!first.sites.is_empty());
        assert_eq!(first.voxel_volume_a3, options.spacing_angstrom.powi(3));
        assert_eq!(first.sites, second.sites);
    }

    #[test]
    fn adaptive_fast_grid_halves_linear_point_density_weight() {
        let grid = build_adaptive_hydration_grid(
            &two_atom_structure(),
            &AdaptiveHydrationOptions {
                spacing_angstrom: 8.0,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(grid.voxel_volume_a3, 2.0 * 4.0_f64.powi(3));
    }

    #[test]
    fn hydration_grids_are_translation_invariant() {
        let structure = two_atom_structure();
        let shift = Vector3::new(3.17, -8.4, 2.6);
        let translated = Structure {
            atoms: structure
                .atoms
                .iter()
                .cloned()
                .map(|mut atom| {
                    atom.pos += shift;
                    atom
                })
                .collect(),
        };
        let adaptive_options = AdaptiveHydrationOptions::default();
        let adaptive = build_adaptive_hydration_grid(&structure, &adaptive_options).unwrap();
        let adaptive_translated =
            build_adaptive_hydration_grid(&translated, &adaptive_options).unwrap();
        assert_eq!(adaptive.sites.len(), adaptive_translated.sites.len());
        for (original, moved) in adaptive.sites.iter().zip(&adaptive_translated.sites) {
            assert!((moved - original - shift).norm() < 1e-12);
        }

        let exterior_options = SolventGridOptions::default();
        let exterior = build_hydration_grid(&structure, &exterior_options).unwrap();
        let exterior_translated = build_hydration_grid(&translated, &exterior_options).unwrap();
        assert_eq!(exterior.sites.len(), exterior_translated.sites.len());
        for (original, moved) in exterior.sites.iter().zip(&exterior_translated.sites) {
            assert!((moved - original - shift).norm() < 1e-12);
        }
    }

    #[test]
    fn water_form_factor_has_ten_electrons_at_zero() {
        // Cromer-Mann coefficients are rounded tabulations.
        assert!((water_form_factor(0.0) - 10.0).abs() < 1e-3);
    }

    #[test]
    fn displaced_volume_table_matches_cited_values() {
        for (element, expected) in [
            (Element::H, 5.15),
            (Element::C, 16.44),
            (Element::N, 2.49),
            (Element::O, 9.13),
            (Element::P, 3.37),
            (Element::S, 26.09),
            (Element::Na, 4.45),
            (Element::Mg, 1.56),
            (Element::Cl, 24.84),
            (Element::K, 11.01),
            (Element::Ca, 4.19),
            (Element::Fe, 7.99),
            (Element::Zn, 9.85),
            (Element::I, 44.60),
        ] {
            assert_eq!(excluded_volume_for(element), expected);
        }
    }
}
