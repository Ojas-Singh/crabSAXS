//! Atomic and chemically typed united-atom X-ray form factors.
//!
//! Cromer–Mann tables use `s = q/(4π)` with q in Å⁻¹. Coefficients are the
//! standard neutral-atom values from International Tables for Crystallography
//! Vol. C, Table 6.1.1.4. Standard protein atoms use the five-Gaussian
//! united-atom coefficients published in Grudinin et al. (2017), Table 2.
//! Those coefficients approximate their equation (8), including the
//! heavy-atom/H distance from Table 1, over `q=0..0.95 Å⁻¹`.

use crate::structure::{Atom, Element};

#[derive(Debug, Clone, Copy)]
pub struct CromerMann {
    pub a: [f64; 4],
    pub b: [f64; 4],
    pub c: f64,
}

#[derive(Debug, Clone, Copy)]
struct FiveGaussian {
    a: [f64; 5],
    b: [f64; 5],
    c: f64,
}

impl FiveGaussian {
    fn evaluate(self, q: f64) -> f64 {
        // The tabulated Gaussian exponents use the crystallographic
        // `s = sin(theta)/lambda = q/(4*pi)` convention.
        let s2 = (q / (4.0 * std::f64::consts::PI)).powi(2);
        self.a
            .iter()
            .zip(self.b)
            .map(|(a, b)| a * (-b * s2).exp())
            .sum::<f64>()
            + self.c
    }
}

impl CromerMann {
    pub fn evaluate(&self, q: f64) -> f64 {
        let s2 = (q / (4.0 * std::f64::consts::PI)).powi(2);
        self.a
            .iter()
            .zip(self.b)
            .map(|(a, b)| a * (-b * s2).exp())
            .sum::<f64>()
            + self.c
    }
}

const H: CromerMann = CromerMann {
    a: [0.493002, 0.322912, 0.140191, 0.04081],
    b: [10.5109, 26.1257, 3.14236, 57.7997],
    c: 0.003038,
};
const C: CromerMann = CromerMann {
    a: [2.31, 1.02, 1.5886, 0.865],
    b: [20.8439, 10.2075, 0.5687, 51.6512],
    c: 0.2156,
};
const N: CromerMann = CromerMann {
    a: [12.2126, 3.1322, 2.0125, 1.1663],
    b: [0.0057, 9.8933, 28.9975, 0.5826],
    c: -11.529,
};
const O: CromerMann = CromerMann {
    a: [3.0485, 2.2868, 1.5463, 0.867],
    b: [13.2771, 5.7011, 0.3239, 32.9089],
    c: 0.2508,
};
const S: CromerMann = CromerMann {
    a: [6.9053, 5.2034, 1.4379, 1.5863],
    b: [1.4679, 22.2151, 0.2536, 56.172],
    c: 0.8669,
};
const P: CromerMann = CromerMann {
    a: [6.4345, 4.1791, 1.7800, 1.4908],
    b: [1.9067, 27.1570, 0.5260, 68.1645],
    c: 1.1149,
};
const NA: CromerMann = CromerMann {
    a: [4.7626, 3.1736, 1.2674, 1.1128],
    b: [3.285, 8.8422, 0.3136, 129.424],
    c: 0.676,
};
const MG: CromerMann = CromerMann {
    a: [5.4204, 2.1735, 1.2269, 2.3073],
    b: [2.8275, 79.2611, 0.3808, 7.1937],
    c: 0.8584,
};
const CL: CromerMann = CromerMann {
    a: [11.4604, 7.1964, 6.2556, 1.6455],
    b: [0.0104, 1.1662, 18.5194, 47.7784],
    c: -9.5574,
};
const K: CromerMann = CromerMann {
    a: [8.2186, 7.4398, 1.0519, 0.8659],
    b: [12.7949, 0.7748, 213.187, 41.6841],
    c: 1.4228,
};
const CA: CromerMann = CromerMann {
    a: [8.6266, 7.3873, 1.5899, 1.0211],
    b: [10.4421, 0.6599, 85.7484, 178.437],
    c: 1.3751,
};
const FE: CromerMann = CromerMann {
    a: [11.7695, 7.3573, 3.5222, 2.3045],
    b: [4.7611, 0.3072, 15.3535, 76.8805],
    c: 1.0369,
};
const ZN: CromerMann = CromerMann {
    a: [14.0743, 7.0318, 5.1652, 2.41],
    b: [3.2655, 0.2333, 10.3163, 58.7097],
    c: 1.3041,
};
const I: CromerMann = CromerMann {
    a: [20.1472, 18.9949, 7.5138, 2.2735],
    b: [4.347, 0.3814, 27.766, 66.8776],
    c: 4.0712,
};

pub fn coefficients_for(element: Element) -> CromerMann {
    match element {
        Element::H => H,
        Element::C => C,
        Element::N => N,
        Element::O => O,
        Element::S => S,
        Element::P => P,
        Element::Na => NA,
        Element::Mg => MG,
        Element::Cl => CL,
        Element::Ca => CA,
        Element::Fe => FE,
        Element::Zn => ZN,
        Element::K => K,
        Element::I => I,
        Element::Other => CromerMann {
            a: [0.0; 4],
            b: [0.0; 4],
            c: 0.0,
        },
    }
}

fn add(mut a: CromerMann, b: CromerMann, factor: f64) -> CromerMann {
    for i in 0..4 {
        a.a[i] += b.a[i] * factor;
    }
    a.c += b.c * factor;
    a
}

/// Legacy element-only correction retained for API compatibility. Production
/// calculations use [`form_factor_for_atom`], which applies chemically typed
/// united-atom form factors for standard residues.
pub fn apply_implicit_hydrogen_correction(base: CromerMann, element: Element) -> CromerMann {
    let count = match element {
        Element::C => 1.0,
        Element::N | Element::O => 0.5,
        _ => 0.0,
    };
    add(base, H, count)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UnitedAtomGroup {
    CSp3H,
    CSp3H2,
    CSp3H3,
    CAromaticH,
    OAlcoholH,
    OAcidH,
    OResonance,
    NH,
    NH2,
    NHPositive,
    NH3Positive,
    GuanidineNH,
    GuanidineNH2,
    SH,
}

impl UnitedAtomGroup {
    const fn implicit_hydrogen_count(self) -> f64 {
        match self {
            Self::CSp3H
            | Self::CAromaticH
            | Self::OAlcoholH
            | Self::OAcidH
            | Self::NH
            | Self::NHPositive
            | Self::GuanidineNH
            | Self::SH => 1.0,
            Self::CSp3H2 | Self::NH2 | Self::GuanidineNH2 => 2.0,
            Self::CSp3H3 | Self::NH3Positive => 3.0,
            Self::OResonance => 0.0,
        }
    }
}

impl UnitedAtomGroup {
    fn coefficients(self) -> FiveGaussian {
        match self {
            Self::CSp3H => FiveGaussian {
                a: [2.909530, 0.485267, 1.516151, 0.206905, 1.541626],
                b: [13.933084, 23.221524, 41.990403, 4.974183, 0.679266],
                c: 0.337670,
            },
            Self::CSp3H2 => FiveGaussian {
                a: [3.275723, 0.870037, 1.534606, 0.395078, 1.544562],
                b: [13.408502, 23.785175, 41.922444, 5.019072, 0.724439],
                c: 0.377096,
            },
            Self::CSp3H3 => FiveGaussian {
                a: [3.681341, 1.228691, 1.549320, 0.574033, 1.554377],
                b: [13.026207, 24.131974, 41.869426, 4.984373, 0.765769],
                c: 0.409294,
            },
            Self::CAromaticH => FiveGaussian {
                a: [2.168070, 1.275811, 1.561096, 0.742395, -6.151144],
                b: [12.642907, 18.420069, 41.768517, 1.535360, -0.045937],
                c: 7.400917,
            },
            Self::OAlcoholH => FiveGaussian {
                a: [0.456221, 3.219608, 0.812773, 2.666928, 1.380927],
                b: [21.503498, 13.397134, 34.547137, 5.826620, 0.412902],
                c: 0.463202,
            },
            Self::OAcidH => FiveGaussian {
                a: [3.213280, 0.463019, 0.815724, 2.664450, 1.384266],
                b: [13.383078, 21.362223, 34.531415, 5.823549, 0.410805],
                c: 0.458919,
            },
            Self::OResonance => FiveGaussian {
                a: [0.688944, 2.929687, 0.416472, 2.606983, 1.319232],
                b: [29.319200, 6.572228, 64.951658, 16.267799, 0.455640],
                c: 0.537548,
            },
            Self::NH => FiveGaussian {
                a: [1.650531, 0.429639, 2.144736, 1.851894, 1.408921],
                b: [10.603730, 6.987283, 29.939901, 10.573859, 0.611678],
                c: 0.510589,
            },
            Self::NH2 => FiveGaussian {
                a: [1.904157, 1.942536, 2.435585, 0.730512, 1.379728],
                b: [10.803702, 10.792421, 29.610479, 6.847755, 0.709687],
                c: 0.603738,
            },
            Self::NHPositive => FiveGaussian {
                a: [1.426540, 0.426903, 1.878894, 1.608251, 1.200216],
                b: [10.652268, 7.017651, 29.878525, 10.619493, 0.631765],
                c: 0.456024,
            },
            Self::NH3Positive => FiveGaussian {
                a: [1.882162, 1.933200, 2.465843, 0.927311, 1.190889],
                b: [10.975157, 10.956008, 29.208572, 6.663555, 0.843650],
                c: 0.597322,
            },
            Self::GuanidineNH => FiveGaussian {
                a: [3.630164, 0.228310, 1.869734, 0.170550, 1.440894],
                b: [10.267139, 25.118086, 30.241288, 3.412776, 0.486644],
                c: 0.323504,
            },
            Self::GuanidineNH2 => FiveGaussian {
                a: [1.792216, 0.724464, 2.347044, 1.903020, 1.313042],
                b: [10.830060, 6.846763, 29.579607, 10.800018, 0.720448],
                c: 0.583312,
            },
            Self::SH => FiveGaussian {
                a: [0.570042, 6.337416, 1.641643, 5.398549, 1.527982],
                b: [11.447986, 1.197657, 55.401032, 22.420955, 2.356552],
                c: 1.523944,
            },
        }
    }
}

fn is_amino_acid(residue: &str) -> bool {
    matches!(
        residue,
        "ALA"
            | "ARG"
            | "ASN"
            | "ASP"
            | "ASH"
            | "CYS"
            | "CYX"
            | "GLN"
            | "GLU"
            | "GLH"
            | "GLY"
            | "HIS"
            | "HID"
            | "HIE"
            | "HIP"
            | "ILE"
            | "LEU"
            | "LYS"
            | "MET"
            | "MSE"
            | "PHE"
            | "PRO"
            | "SER"
            | "THR"
            | "TRP"
            | "TYR"
            | "VAL"
            | "PCA"
    )
}

fn nucleotide_base(residue: &str) -> Option<char> {
    match residue {
        "A" | "DA" | "RA" | "ADE" => Some('A'),
        "C" | "DC" | "RC" | "CYT" => Some('C'),
        "G" | "DG" | "RG" | "GUA" => Some('G'),
        "U" | "DU" | "RU" | "URA" => Some('U'),
        "T" | "DT" | "RT" | "THY" => Some('T'),
        _ => None,
    }
}

fn nucleotide_group(residue: &str, atom_name: &str) -> Option<UnitedAtomGroup> {
    let base = nucleotide_base(residue)?;
    let name = atom_name.replace('*', "'");
    let deoxy = residue.starts_with('D') || base == 'T';
    match name.as_str() {
        "C1'" | "C3'" | "C4'" => return Some(UnitedAtomGroup::CSp3H),
        "C2'" => {
            return Some(if deoxy {
                UnitedAtomGroup::CSp3H2
            } else {
                UnitedAtomGroup::CSp3H
            });
        }
        "C5'" => return Some(UnitedAtomGroup::CSp3H2),
        "O2'" if !deoxy => return Some(UnitedAtomGroup::OAlcoholH),
        "OP1" | "OP2" | "O1P" | "O2P" => return Some(UnitedAtomGroup::OResonance),
        "C8" if matches!(base, 'A' | 'G') => return Some(UnitedAtomGroup::CAromaticH),
        _ => {}
    }
    match (base, name.as_str()) {
        ('A', "C2") => Some(UnitedAtomGroup::CAromaticH),
        ('A', "N6") => Some(UnitedAtomGroup::NH2),
        ('G', "N1") => Some(UnitedAtomGroup::NH),
        ('G', "N2") => Some(UnitedAtomGroup::NH2),
        ('C', "N4") => Some(UnitedAtomGroup::NH2),
        ('C', "C5" | "C6") | ('U', "C5" | "C6") | ('T', "C6") => Some(UnitedAtomGroup::CAromaticH),
        ('U' | 'T', "N3") => Some(UnitedAtomGroup::NH),
        ('T', "C7" | "C5M") => Some(UnitedAtomGroup::CSp3H3),
        _ => None,
    }
}

fn united_atom_group(atom: &Atom) -> Option<UnitedAtomGroup> {
    let residue = atom.residue_name.trim().to_ascii_uppercase();
    let name = atom.atom_name.trim().to_ascii_uppercase();
    if nucleotide_base(&residue).is_some() {
        return nucleotide_group(&residue, &name);
    }
    if !is_amino_acid(&residue) {
        return None;
    }

    if name == "N" {
        return (!matches!(residue.as_str(), "PRO" | "PCA")).then_some(UnitedAtomGroup::NH);
    }
    if name == "CA" {
        return Some(if residue == "GLY" {
            UnitedAtomGroup::CSp3H2
        } else {
            UnitedAtomGroup::CSp3H
        });
    }

    match (residue.as_str(), name.as_str()) {
        ("ALA", "CB") => Some(UnitedAtomGroup::CSp3H3),
        ("ARG", "CB" | "CG" | "CD") => Some(UnitedAtomGroup::CSp3H2),
        ("ARG", "NE") => Some(UnitedAtomGroup::GuanidineNH),
        ("ARG", "NH1" | "NH2") => Some(UnitedAtomGroup::GuanidineNH2),
        ("ASN", "CB") | ("ASP" | "ASH", "CB") | ("CYS" | "CYX", "CB") => {
            Some(UnitedAtomGroup::CSp3H2)
        }
        ("ASN", "ND2") => Some(UnitedAtomGroup::NH2),
        ("ASP" | "GLU", "OD1" | "OD2" | "OE1" | "OE2") => Some(UnitedAtomGroup::OResonance),
        ("ASH", "OD2") | ("GLH", "OE2") => Some(UnitedAtomGroup::OAcidH),
        ("CYS", "SG") => Some(UnitedAtomGroup::SH),
        ("GLN", "CB" | "CG") | ("GLU" | "GLH", "CB" | "CG") => Some(UnitedAtomGroup::CSp3H2),
        ("GLN", "NE2") => Some(UnitedAtomGroup::NH2),
        ("HIS" | "HID" | "HIE" | "HIP", "CB") => Some(UnitedAtomGroup::CSp3H2),
        ("HIS" | "HID", "ND1") | ("HIE", "NE2") => Some(UnitedAtomGroup::NH),
        ("HIP", "ND1" | "NE2") => Some(UnitedAtomGroup::NHPositive),
        ("HIS" | "HID" | "HIE" | "HIP", "CD2" | "CE1") => Some(UnitedAtomGroup::CAromaticH),
        ("ILE", "CB") | ("LEU", "CG") | ("THR", "CB") | ("VAL", "CB") => {
            Some(UnitedAtomGroup::CSp3H)
        }
        ("ILE", "CG1") | ("LEU", "CB") => Some(UnitedAtomGroup::CSp3H2),
        ("ILE", "CG2" | "CD1") | ("LEU", "CD1" | "CD2") => Some(UnitedAtomGroup::CSp3H3),
        ("LYS", "CB" | "CG" | "CD" | "CE") => Some(UnitedAtomGroup::CSp3H2),
        ("LYS", "NZ") => Some(UnitedAtomGroup::NH3Positive),
        ("MET" | "MSE", "CB" | "CG") => Some(UnitedAtomGroup::CSp3H2),
        ("MET" | "MSE", "CE") => Some(UnitedAtomGroup::CSp3H3),
        ("PHE" | "TYR", "CB") | ("TRP", "CB") => Some(UnitedAtomGroup::CSp3H2),
        ("PHE", "CD1" | "CD2" | "CE1" | "CE2" | "CZ")
        | ("TYR", "CD1" | "CD2" | "CE1" | "CE2")
        | ("TRP", "CD1" | "CE3" | "CZ2" | "CZ3" | "CH2") => Some(UnitedAtomGroup::CAromaticH),
        ("PRO", "CB" | "CG" | "CD") => Some(UnitedAtomGroup::CSp3H2),
        ("SER", "CB") => Some(UnitedAtomGroup::CSp3H2),
        ("SER", "OG") | ("THR", "OG1") | ("TYR", "OH") => Some(UnitedAtomGroup::OAlcoholH),
        ("THR", "CG2") | ("VAL", "CG1" | "CG2") => Some(UnitedAtomGroup::CSp3H3),
        ("TRP", "NE1") => Some(UnitedAtomGroup::NH),
        ("PCA", "CB" | "CG") => Some(UnitedAtomGroup::CSp3H2),
        _ => None,
    }
}

/// Number of covalently attached hydrogens represented by a standard
/// protein united-atom group. Unknown and non-standard atoms return zero.
pub(crate) fn implicit_hydrogen_count_for_atom(atom: &Atom) -> f64 {
    united_atom_group(atom).map_or(0.0, UnitedAtomGroup::implicit_hydrogen_count)
}

pub fn form_factor_for_atom(atom: &Atom, implicit_hydrogen: bool, q: f64) -> f64 {
    if implicit_hydrogen && atom.element == Element::H {
        // Explicit hydrogen coordinates are ignored in united-atom mode to
        // prevent double counting. Select explicit mode to use them.
        return 0.0;
    }
    let value = if implicit_hydrogen {
        united_atom_group(atom)
            .map(|group| group.coefficients().evaluate(q))
            .unwrap_or_else(|| coefficients_for(atom.element).evaluate(q))
    } else {
        coefficients_for(atom.element).evaluate(q)
    };
    atom.occupancy * value
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn carbon_form_factor_at_q_zero_is_atomic_number() {
        assert!((coefficients_for(Element::C).evaluate(0.0) - 6.0).abs() < 0.01);
    }
    #[test]
    fn form_factors_decay_with_q() {
        assert!(
            coefficients_for(Element::C).evaluate(0.5) < coefficients_for(Element::C).evaluate(0.0)
        );
    }

    #[test]
    fn supported_ion_and_metal_factors_match_atomic_numbers_at_zero() {
        for (element, electrons) in [
            (Element::Na, 11.0),
            (Element::Mg, 12.0),
            (Element::Cl, 17.0),
            (Element::K, 19.0),
            (Element::Ca, 20.0),
            (Element::Fe, 26.0),
            (Element::Zn, 30.0),
            (Element::I, 53.0),
        ] {
            let factor = coefficients_for(element);
            assert!((factor.evaluate(0.0) - electrons).abs() < 0.02);
            assert!(factor.evaluate(0.5) < factor.evaluate(0.0));
        }
    }

    fn atom(residue: &str, name: &str, element: Element) -> Atom {
        Atom {
            pos: nalgebra::Vector3::zeros(),
            element,
            occupancy: 1.0,
            is_hetatm: false,
            atom_name: name.into(),
            residue_name: residue.into(),
            chain_id: "A".into(),
            residue_id: 1,
        }
    }

    #[test]
    fn united_atom_electron_counts_match_chemistry_at_q_zero() {
        assert!(
            (form_factor_for_atom(&atom("ALA", "CB", Element::C), true, 0.0) - 9.0).abs() < 0.01
        );
        assert!(
            (form_factor_for_atom(&atom("GLY", "CA", Element::C), true, 0.0) - 8.0).abs() < 0.01
        );
        assert!(
            (form_factor_for_atom(&atom("LYS", "NZ", Element::N), true, 0.0) - 9.0).abs() < 0.01
        );
        assert!(
            (form_factor_for_atom(&atom("ASP", "OD1", Element::O), true, 0.0) - 8.5).abs() < 0.01
        );
    }

    #[test]
    fn nucleotide_united_atoms_include_covalent_hydrogens() {
        for (residue, name, element, electrons) in [
            ("DA", "C2'", Element::C, 8.0),
            ("A", "C2'", Element::C, 7.0),
            ("A", "N6", Element::N, 9.0),
            ("DG", "N1", Element::N, 8.0),
            ("DT", "C7", Element::C, 9.0),
        ] {
            let value = form_factor_for_atom(&atom(residue, name, element), true, 0.0);
            assert!(
                (value - electrons).abs() < 0.02,
                "{residue} {name}: {value}"
            );
        }
    }

    #[test]
    fn explicit_hydrogens_are_not_double_counted_in_implicit_mode() {
        let hydrogen = atom("ALA", "HB1", Element::H);
        assert_eq!(form_factor_for_atom(&hydrogen, true, 0.1), 0.0);
        assert!(form_factor_for_atom(&hydrogen, false, 0.1) > 0.0);
    }
}
