//! Experimental curve parsing and simple two-column output.

use crate::error::{Result, SaxsError};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ExperimentalCurve {
    pub q: Vec<f64>,
    pub intensity: Vec<f64>,
    pub sigma: Vec<f64>,
    /// Whether all rows supplied an uncertainty column.
    #[serde(default)]
    pub has_errors: bool,
}

impl ExperimentalCurve {
    pub fn from_dat_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mut curve = Self {
            has_errors: true,
            ..Self::default()
        };
        let mut q_scale = 1.0;
        let mut saw_numeric = false;
        for (line_no, line) in BufReader::new(file).lines().enumerate() {
            let line = line?;
            let trimmed = line.trim();
            let lower = trimmed.to_ascii_lowercase();
            if lower.contains("nm") && (lower.contains('q') || lower.contains("angstrom")) {
                q_scale = 0.1;
            }
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let normalized = trimmed.replace(',', " ");
            let fields = normalized.split_whitespace().collect::<Vec<_>>();
            let first = fields.first().and_then(|value| value.parse::<f64>().ok());
            if first.is_none() {
                // Many deposited curves have a free-form description/header
                // without a leading '#'; ignore only clearly non-numeric lines.
                continue;
            }
            if fields.len() < 2 {
                return Err(SaxsError::ExperimentalDataParse(format!(
                    "expected q and intensity on line {}",
                    line_no + 1
                )));
            }
            let values = fields[..fields.len().min(3)]
                .iter()
                .map(|value| value.parse::<f64>())
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|_| {
                    SaxsError::ExperimentalDataParse(format!(
                        "invalid numeric value on line {}",
                        line_no + 1
                    ))
                })?;
            if !saw_numeric && q_scale == 1.0 && values[0] > 1.0 {
                // A header is preferred; otherwise use the conventional SAXS
                // range heuristic for nm⁻¹ input and normalize to Å⁻¹.
                q_scale = 0.1;
            }
            saw_numeric = true;
            let (q, intensity, sigma) = (
                values[0] * q_scale,
                values[1],
                values.get(2).copied().unwrap_or(1.0),
            );
            // Some SAS tutorial/deposition files terminate their numeric
            // region with an explicit `I=0, sigma=0` sentinel row.
            if intensity == 0.0 && sigma == 0.0 {
                continue;
            }
            if !q.is_finite()
                || q < 0.0
                || !intensity.is_finite()
                || !sigma.is_finite()
                || sigma <= 0.0
            {
                return Err(SaxsError::ExperimentalDataParse(format!(
                    "q and sigma must be positive and all values finite on line {}",
                    line_no + 1
                )));
            }
            if curve.q.last().is_some_and(|previous| q < *previous) {
                return Err(SaxsError::ExperimentalDataParse(format!(
                    "q values must be non-decreasing on line {}",
                    line_no + 1
                )));
            }
            curve.q.push(q);
            curve.intensity.push(intensity);
            curve.sigma.push(sigma);
            if fields.len() < 3 {
                curve.has_errors = false;
            }
        }
        if curve.q.is_empty() {
            return Err(SaxsError::ExperimentalDataParse(
                "no data rows found".into(),
            ));
        }
        if curve.q.len() != curve.intensity.len() || curve.q.len() != curve.sigma.len() {
            return Err(SaxsError::ExperimentalDataParse(
                "column lengths differ".into(),
            ));
        }
        Ok(curve)
    }

    /// Return the portion of a curve at or below `qmax`, preserving row order.
    pub fn through_qmax(&self, qmax: f64) -> Self {
        let mut curve = Self {
            has_errors: self.has_errors,
            ..Self::default()
        };
        for ((&q, &intensity), &sigma) in self.q.iter().zip(&self.intensity).zip(&self.sigma) {
            if q <= qmax {
                curve.q.push(q);
                curve.intensity.push(intensity);
                curve.sigma.push(sigma);
            }
        }
        curve
    }

    pub fn len(&self) -> usize {
        self.q.len()
    }
    pub fn is_empty(&self) -> bool {
        self.q.is_empty()
    }

    /// Internal q convention after parsing and unit normalization.
    pub fn q_unit(&self) -> &'static str {
        "angstrom^-1"
    }
}

pub fn write_curve_dat<P: AsRef<Path>>(path: P, q: &[f64], intensity: &[f64]) -> Result<()> {
    if q.len() != intensity.len() {
        return Err(SaxsError::InvalidInput(
            "q and intensity lengths differ".into(),
        ));
    }
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "# q (1/A) intensity")?;
    for (&q, &i) in q.iter().zip(intensity) {
        writeln!(out, "{q:.8e} {i:.8e}")?;
    }
    Ok(())
}

pub fn write_curve_csv<P: AsRef<Path>>(path: P, q: &[f64], intensity: &[f64]) -> Result<()> {
    if q.len() != intensity.len() {
        return Err(SaxsError::InvalidInput(
            "q and intensity lengths differ".into(),
        ));
    }
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "q,intensity")?;
    for (&q, &intensity) in q.iter().zip(intensity) {
        writeln!(out, "{q:.8e},{intensity:.8e}")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temporary_curve(contents: &str) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(contents.as_bytes()).unwrap();
        file
    }

    #[test]
    fn parses_validated_three_column_curve() {
        let file = temporary_curve("header\n# comment\n0.01 2.0 0.1\n0.02,1.5,0.2\n0.03 0.0 0.0\n");
        let curve = ExperimentalCurve::from_dat_file(file.path()).unwrap();
        assert_eq!(curve.q, [0.01, 0.02]);
        assert_eq!(curve.sigma, [0.1, 0.2]);
    }

    #[test]
    fn accepts_two_column_data_and_rejects_nonpositive_sigma_and_unsorted_q() {
        let two_column = temporary_curve("0.01 2.0\n0.02 1.0\n");
        let parsed = ExperimentalCurve::from_dat_file(two_column.path()).unwrap();
        assert!(!parsed.has_errors);
        assert_eq!(parsed.sigma, [1.0, 1.0]);
        for contents in [
            "0.01 2.0 0\n",
            "0.02 2.0 0.1\n0.01 1.0 0.1\n",
            "-0.01 2.0 0.1\n",
        ] {
            let file = temporary_curve(contents);
            assert!(ExperimentalCurve::from_dat_file(file.path()).is_err());
        }
    }

    #[test]
    fn normalizes_nm_inverse_angstrom_header() {
        let file = temporary_curve("# q (1/nm) I sigma\n1.0 2.0 0.1\n2.0 1.0 0.1\n");
        let curve = ExperimentalCurve::from_dat_file(file.path()).unwrap();
        assert_eq!(curve.q, [0.1, 0.2]);
        assert_eq!(curve.q_unit(), "angstrom^-1");
    }
}
