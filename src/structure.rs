//! Structure input and the small, stable atom representation used by the
//! scattering engines. Coordinates and all derived distances are in Å.

use crate::error::{Result, SaxsError};
use nalgebra::Vector3;
use pdbtbx::{
    ContainsAtomConformer, ContainsAtomConformerResidue, ContainsAtomConformerResidueChain,
    ReadOptions, StrictnessLevel,
};
use std::collections::HashMap;
use std::path::Path;

/// Selection policy for alternate atom locations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AltlocPolicy {
    #[default]
    Highest,
    First,
    All,
}

/// Broad residue classification used to keep carbohydrate HETATM records by
/// default without silently including ligands or crystallization additives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidueKind {
    Polymer,
    Glycan,
    Water,
    Other,
}

/// Common carbohydrate residue codes from the wwPDB chemical component
/// dictionary. The list is intentionally conservative; unknown codes are
/// reported by `inspect` and can be enabled with `--include-hetatm`.
pub fn residue_kind(name: &str) -> ResidueKind {
    let code = name.trim().to_ascii_uppercase();
    if matches!(code.as_str(), "HOH" | "WAT" | "DOD") {
        return ResidueKind::Water;
    }
    const GLYCANS: &[&str] = &[
        "NAG", "NDG", "BMA", "MAN", "GLC", "GAL", "FUC", "SIA", "NAN", "NEU", "XYS", "XYP", "ARA",
        "ARB", "RIB", "FRU", "G6D", "G1P", "BGC", "AFD", "AGS", "AMU", "API", "DGA", "DGL", "DGU",
        "GCU", "GTR", "IDR", "LBT", "MMA", "RM4", "SUC", "TRE", "MAL", "LMT", "KDO", "KDN", "DAN",
        "XUL",
    ];
    if GLYCANS.contains(&code.as_str()) {
        ResidueKind::Glycan
    } else if matches!(
        code.as_str(),
        "ALA"
            | "ARG"
            | "ASN"
            | "ASP"
            | "CYS"
            | "GLN"
            | "GLU"
            | "GLY"
            | "HIS"
            | "ILE"
            | "LEU"
            | "LYS"
            | "MET"
            | "PHE"
            | "PRO"
            | "SER"
            | "THR"
            | "TRP"
            | "TYR"
            | "VAL"
            | "A"
            | "C"
            | "G"
            | "U"
            | "T"
            | "DA"
            | "DC"
            | "DG"
            | "DT"
    ) {
        ResidueKind::Polymer
    } else {
        ResidueKind::Other
    }
}

/// Elements for which the calculation has calibrated X-ray coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Element {
    H,
    C,
    N,
    O,
    S,
    P,
    Na,
    Mg,
    Cl,
    Ca,
    Fe,
    Zn,
    K,
    I,
    Other,
}

impl Element {
    pub fn from_symbol(symbol: &str) -> Self {
        match symbol.trim().to_ascii_uppercase().as_str() {
            "H" | "D" => Self::H,
            "C" => Self::C,
            "N" => Self::N,
            "O" => Self::O,
            "S" => Self::S,
            "P" => Self::P,
            "NA" => Self::Na,
            "MG" => Self::Mg,
            "CL" => Self::Cl,
            "CA" => Self::Ca,
            "FE" => Self::Fe,
            "ZN" => Self::Zn,
            "K" => Self::K,
            "I" => Self::I,
            _ => Self::Other,
        }
    }

    pub fn symbol(self) -> &'static str {
        match self {
            Self::H => "H",
            Self::C => "C",
            Self::N => "N",
            Self::O => "O",
            Self::S => "S",
            Self::P => "P",
            Self::Na => "Na",
            Self::Mg => "Mg",
            Self::Cl => "Cl",
            Self::Ca => "Ca",
            Self::Fe => "Fe",
            Self::Zn => "Zn",
            Self::K => "K",
            Self::I => "I",
            Self::Other => "?",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Atom {
    pub pos: Vector3<f64>,
    pub element: Element,
    pub occupancy: f64,
    pub is_hetatm: bool,
    pub atom_name: String,
    pub residue_name: String,
    pub chain_id: String,
    pub residue_id: isize,
}

#[derive(Debug, Clone, Default)]
pub struct Structure {
    pub atoms: Vec<Atom>,
}

/// Occupancy assigned after selecting one alternate conformation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OccupancyPolicy {
    /// Preserve the occupancy reported for the selected coordinate.
    #[default]
    Reported,
    /// Treat the selected conformer as one complete structural model.
    SelectedConformerFull,
}

#[derive(Debug, Clone)]
pub struct ParseOptions {
    pub include_hetatm: bool,
    pub include_waters: bool,
    pub include_glycans: bool,
    pub model_index: usize,
    pub allow_unknown_elements: bool,
    pub occupancy_policy: OccupancyPolicy,
    pub altloc_policy: AltlocPolicy,
    pub chains: Option<Vec<String>>,
}

impl Default for ParseOptions {
    fn default() -> Self {
        Self {
            include_hetatm: false,
            include_waters: false,
            include_glycans: true,
            model_index: 0,
            allow_unknown_elements: false,
            occupancy_policy: OccupancyPolicy::default(),
            altloc_policy: AltlocPolicy::default(),
            chains: None,
        }
    }
}

impl Structure {
    pub fn from_pdb_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::from_pdb_file_with_options(path, ParseOptions::default())
    }

    pub fn from_pdb_file_with_options<P: AsRef<Path>>(
        path: P,
        options: ParseOptions,
    ) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let mut read_options = ReadOptions::new();
        read_options
            .set_level(StrictnessLevel::Loose)
            .set_only_first_model(false);
        let (pdb, _warnings) = read_options.read(&path_str).map_err(|errors| {
            SaxsError::StructureParse(
                errors
                    .into_iter()
                    .map(|error| error.to_string())
                    .collect::<Vec<_>>()
                    .join("; "),
            )
        })?;
        let model = pdb.models().nth(options.model_index).ok_or_else(|| {
            SaxsError::StructureParse(format!("model {} is not present", options.model_index))
        })?;
        let mut candidates: HashMap<(String, isize, String, String), Vec<Atom>> = HashMap::new();
        for hierarchy in model.atoms_with_hierarchy() {
            let source = hierarchy.atom();
            let is_hetatm = source.hetero();
            let residue_name = hierarchy.residue().name().unwrap_or("UNK");
            let kind = residue_kind(residue_name);
            if options
                .chains
                .as_ref()
                .is_some_and(|chains| !chains.iter().any(|chain| chain == hierarchy.chain().id()))
            {
                continue;
            }
            if is_hetatm {
                if kind == ResidueKind::Water && !options.include_waters {
                    continue;
                }
                if !options.include_hetatm
                    && !(options.include_glycans && kind == ResidueKind::Glycan)
                {
                    continue;
                }
            }
            let element = source
                .element()
                .map(|element| Element::from_symbol(element.symbol()))
                .unwrap_or_else(|| Element::from_symbol(source.name()));
            if element == Element::Other && !options.allow_unknown_elements {
                return Err(SaxsError::UnsupportedElement(source.name().to_string()));
            }
            let key = (
                hierarchy.chain().id().to_string(),
                hierarchy.residue().serial_number(),
                hierarchy
                    .residue()
                    .insertion_code()
                    .unwrap_or("")
                    .to_string(),
                source.name().to_string(),
            );
            let atom = Atom {
                pos: Vector3::new(source.x(), source.y(), source.z()),
                element,
                occupancy: source.occupancy().max(0.0),
                is_hetatm,
                atom_name: source.name().to_string(),
                residue_name: residue_name.to_string(),
                chain_id: hierarchy.chain().id().to_string(),
                residue_id: hierarchy.residue().serial_number(),
            };
            candidates.entry(key).or_default().push(atom);
        }
        let mut atoms = Vec::with_capacity(candidates.len());
        for mut alternatives in candidates.into_values() {
            if options.altloc_policy == AltlocPolicy::All {
                atoms.append(&mut alternatives);
                continue;
            }
            let candidate_count = alternatives.len();
            let has_alternatives = candidate_count > 1;
            let same_position = has_alternatives
                && alternatives
                    .iter()
                    .skip(1)
                    .all(|atom| atom.pos == alternatives[0].pos);
            let mut selected = match options.altloc_policy {
                AltlocPolicy::Highest => alternatives
                    .drain(..)
                    .max_by(|left, right| left.occupancy.total_cmp(&right.occupancy))
                    .expect("candidate groups are never empty"),
                AltlocPolicy::First => alternatives
                    .drain(..)
                    .next()
                    .expect("candidate groups are never empty"),
                AltlocPolicy::All => unreachable!(),
            };
            selected.occupancy = match options.occupancy_policy {
                OccupancyPolicy::SelectedConformerFull if has_alternatives => 1.0,
                OccupancyPolicy::Reported if same_position => {
                    // pdbtbx copies blank-conformer atoms into each named
                    // conformer and divides their occupancy by the original
                    // conformer count. Restore the original reported value.
                    (selected.occupancy * (candidate_count + 1) as f64).min(1.0)
                }
                _ => selected.occupancy,
            };
            atoms.push(selected);
        }
        atoms.sort_by(|left, right| {
            (&left.chain_id, left.residue_id, &left.atom_name).cmp(&(
                &right.chain_id,
                right.residue_id,
                &right.atom_name,
            ))
        });
        Ok(Self { atoms })
    }

    /// Read every model in a multi-model PDB/mmCIF file. A single-model file
    /// returns a one-element vector; malformed files return their parse error.
    pub fn from_pdb_file_models_with_options<P: AsRef<Path>>(
        path: P,
        options: ParseOptions,
    ) -> Result<Vec<Self>> {
        let mut models = Vec::new();
        for model_index in 0.. {
            let mut model_options = options.clone();
            model_options.model_index = model_index;
            match Self::from_pdb_file_with_options(&path, model_options) {
                Ok(model) => models.push(model),
                Err(_error) if model_index > 0 && !models.is_empty() => break,
                Err(error) => return Err(error),
            }
        }
        Ok(models)
    }

    pub fn atom_count(&self) -> usize {
        self.atoms.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_known_atom_count_and_altloc() {
        let structure = Structure::from_pdb_file("examples/data/test_structure.pdb").unwrap();
        assert_eq!(structure.atom_count(), 2);
        assert!(
            (structure
                .atoms
                .iter()
                .find(|atom| atom.atom_name == "CA")
                .unwrap()
                .occupancy
                - 0.8)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn selected_conformer_can_be_treated_as_full_occupancy() {
        let structure = Structure::from_pdb_file_with_options(
            "examples/data/test_structure.pdb",
            ParseOptions {
                occupancy_policy: OccupancyPolicy::SelectedConformerFull,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            structure
                .atoms
                .iter()
                .find(|atom| atom.atom_name == "CA")
                .unwrap()
                .occupancy,
            1.0
        );
    }

    #[test]
    fn applies_hetatm_water_and_model_policies() {
        let path = "examples/data/test_structure.pdb";
        let hetatm_without_water = Structure::from_pdb_file_with_options(
            path,
            ParseOptions {
                include_hetatm: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(hetatm_without_water.atom_count(), 2);

        let with_water = Structure::from_pdb_file_with_options(
            path,
            ParseOptions {
                include_hetatm: true,
                include_waters: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(with_water.atom_count(), 3);

        let second_model = Structure::from_pdb_file_with_options(
            path,
            ParseOptions {
                model_index: 1,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(second_model.atom_count(), 1);
        assert_eq!(second_model.atoms[0].pos.x, 9.0);
    }

    #[test]
    fn parses_first_model_from_mmcif() {
        let structure = Structure::from_pdb_file("examples/data/test_structure.cif").unwrap();
        assert_eq!(structure.atom_count(), 2);
        assert_eq!(structure.atoms[0].chain_id, "A");
    }

    #[test]
    fn malformed_structure_returns_an_error() {
        assert!(Structure::from_pdb_file("examples/data/malformed.pdb").is_err());
    }
}
