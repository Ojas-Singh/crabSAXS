//! Publication-style SAXS diagnostic figures.

use crate::error::{Result, SaxsError};
use crate::io::ExperimentalCurve;
use crate::metrics::{PairDistribution, SaxsFeatures};
use plotters::coord::Shift;
use plotters::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotCurve {
    pub name: String,
    pub intensity: Vec<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiagnosticPlot {
    pub title: String,
    pub experimental: ExperimentalCurve,
    pub curves: Vec<PlotCurve>,
    pub experimental_pr: Option<PairDistribution>,
    pub model_pr: Option<PairDistribution>,
    pub features: Option<SaxsFeatures>,
    #[serde(default)]
    pub effective_sample_size: Option<f64>,
}

impl DiagnosticPlot {
    pub fn save_svg(&self, path: impl AsRef<Path>) -> Result<()> {
        let backend = SVGBackend::new(path.as_ref(), (1500, 1050));
        draw_diagnostic(backend.into_drawing_area(), self)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn save_png(&self, path: impl AsRef<Path>) -> Result<()> {
        let backend = BitMapBackend::new(path.as_ref(), (1500, 1050));
        draw_diagnostic(backend.into_drawing_area(), self)
    }

    #[cfg(target_arch = "wasm32")]
    pub fn save_png(&self, _path: impl AsRef<Path>) -> Result<()> {
        Err(SaxsError::InvalidInput(
            "PNG file output is unavailable in browser builds; use SVG output".into(),
        ))
    }
}

pub fn render_diagnostic_svg(plot: &DiagnosticPlot, path: impl AsRef<Path>) -> Result<()> {
    plot.save_svg(path)
}

pub fn render_diagnostic_png(plot: &DiagnosticPlot, path: impl AsRef<Path>) -> Result<()> {
    plot.save_png(path)
}

fn draw_diagnostic<DB>(root: DrawingArea<DB, Shift>, plot: &DiagnosticPlot) -> Result<()>
where
    DB: DrawingBackend,
    DB::ErrorType: std::fmt::Debug,
{
    root.fill(&WHITE).map_err(plot_error)?;
    let areas = root.split_evenly((2, 2));
    let q_range = positive_q_range(&plot.experimental).ok_or_else(|| {
        SaxsError::InvalidInput("diagnostic plots require at least two positive q values".into())
    })?;
    let intensity_range = intensity_range(plot).ok_or_else(|| {
        SaxsError::InvalidInput("diagnostic plots require positive intensities".into())
    })?;

    let mut log_log = ChartBuilder::on(&areas[0])
        .caption(
            format!(
                "{} — SAXS intensity (points: experiment; lines: fit)",
                plot.title
            ),
            ("sans-serif", 22),
        )
        .margin(12)
        .x_label_area_size(42)
        .y_label_area_size(58)
        .build_cartesian_2d(
            (q_range.0..q_range.1).log_scale(),
            (intensity_range.0..intensity_range.1).log_scale(),
        )
        .map_err(plot_error)?;
    log_log
        .configure_mesh()
        .x_desc("q (Å⁻¹)")
        .y_desc("I(q)")
        .x_labels(6)
        .y_labels(6)
        .light_line_style(BLACK.mix(0.08))
        .draw()
        .map_err(plot_error)?;
    draw_experimental_points(&mut log_log, plot, true)?;
    draw_model_lines(&mut log_log, plot, true)?;

    let mut semilog = ChartBuilder::on(&areas[1])
        .caption(
            "Semilog intensity (points: experiment; lines: fit)",
            ("sans-serif", 22),
        )
        .margin(12)
        .x_label_area_size(42)
        .y_label_area_size(58)
        .build_cartesian_2d(
            q_range.0..q_range.1,
            (intensity_range.0..intensity_range.1).log_scale(),
        )
        .map_err(plot_error)?;
    semilog
        .configure_mesh()
        .x_desc("q (Å⁻¹)")
        .y_desc("I(q)")
        .x_labels(6)
        .y_labels(6)
        .light_line_style(BLACK.mix(0.08))
        .draw()
        .map_err(plot_error)?;
    draw_experimental_points(&mut semilog, plot, false)?;
    draw_model_lines(&mut semilog, plot, false)?;

    let kratky_range = kratky_range(plot).unwrap_or((0.0, 1.0));
    let mut kratky = ChartBuilder::on(&areas[2])
        .caption(
            "Normalized Kratky (black: experiment; colored: fit)",
            ("sans-serif", 22),
        )
        .margin(12)
        .x_label_area_size(42)
        .y_label_area_size(58)
        .build_cartesian_2d(q_range.0..q_range.1, kratky_range.0..kratky_range.1)
        .map_err(plot_error)?;
    kratky
        .configure_mesh()
        .x_desc("q (Å⁻¹)")
        .y_desc("q² I(q) / I(0)")
        .x_labels(6)
        .y_labels(6)
        .light_line_style(BLACK.mix(0.08))
        .draw()
        .map_err(plot_error)?;
    draw_experimental_kratky(&mut kratky, plot)?;
    for (index, curve) in plot.curves.iter().enumerate() {
        let scale = low_q_i0(&plot.experimental.q, &curve.intensity).unwrap_or(1.0);
        let series = plot
            .experimental
            .q
            .iter()
            .zip(&curve.intensity)
            .filter_map(|(&q, &value)| {
                (q > 0.0 && value.is_finite() && value > 0.0).then_some((q, q * q * value / scale))
            });
        kratky
            .draw_series(LineSeries::new(
                series,
                Palette99::pick(index + 1).stroke_width(2),
            ))
            .map_err(plot_error)?;
    }

    let pr_max = plot
        .experimental_pr
        .as_ref()
        .and_then(|value| value.dmax)
        .or_else(|| plot.model_pr.as_ref().and_then(|value| value.dmax))
        .unwrap_or(1.0)
        .max(1.0);
    let pr_y_max = plot
        .experimental_pr
        .iter()
        .chain(plot.model_pr.iter())
        .flat_map(|distribution| distribution.p.iter().copied())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .fold(0.0, f64::max)
        .max(1.0e-8);
    let mut pr = ChartBuilder::on(&areas[3])
        .caption(
            "P(r), area normalized (black: experiment; red: model)",
            ("sans-serif", 22),
        )
        .margin(12)
        .x_label_area_size(42)
        .y_label_area_size(58)
        .build_cartesian_2d(0.0..pr_max, 0.0..pr_y_max * 1.1)
        .map_err(plot_error)?;
    pr.configure_mesh()
        .x_desc("r (Å)")
        .y_desc("P(r)  [area = 1]")
        .x_labels(6)
        .y_labels(6)
        .light_line_style(BLACK.mix(0.08))
        .draw()
        .map_err(plot_error)?;
    if let Some(distribution) = &plot.experimental_pr {
        pr.draw_series(LineSeries::new(
            distribution
                .r
                .iter()
                .copied()
                .zip(distribution.p.iter().copied()),
            BLACK.stroke_width(2),
        ))
        .map_err(plot_error)?;
    }
    if let Some(distribution) = &plot.model_pr {
        pr.draw_series(LineSeries::new(
            distribution
                .r
                .iter()
                .copied()
                .zip(distribution.p.iter().copied()),
            RED.stroke_width(2),
        ))
        .map_err(plot_error)?;
    }

    if let Some(features) = &plot.features {
        let mut annotation = format!(
            "reduced χ² {:.3}\nRg exp/model {} / {} Å\nDmax exp/model {} / {} Å\nKratky {}  P(r) RMS {}",
            features.reduced_chi2,
            format_optional(features.rg_exp),
            format_optional(features.rg_model),
            format_optional(features.dmax_exp),
            format_optional(features.dmax_model),
            format_optional(features.kratky_deviation),
            format_optional(features.pr_rms),
        );
        if let Some(ess) = plot.effective_sample_size {
            annotation.push_str(&format!("\nESS {ess:.3}"));
        }
        let lines = annotation.lines().collect::<Vec<_>>();
        let box_height = 18 + lines.len() as i32 * 17;
        areas[2]
            .draw(&Rectangle::new(
                [(420, 42), (738, 42 + box_height)],
                WHITE.mix(0.88).filled(),
            ))
            .map_err(plot_error)?;
        for (index, line) in lines.iter().enumerate() {
            areas[2]
                .draw(&Text::new(
                    (*line).to_owned(),
                    (432, 60 + index as i32 * 17),
                    ("sans-serif", 13).into_font(),
                ))
                .map_err(plot_error)?;
        }
    }
    root.present().map_err(plot_error)
}

fn draw_experimental_points<DB, X, Y>(
    chart: &mut ChartContext<'_, DB, Cartesian2d<X, Y>>,
    plot: &DiagnosticPlot,
    log_x: bool,
) -> Result<()>
where
    DB: DrawingBackend,
    DB::ErrorType: std::fmt::Debug,
    X: Ranged<ValueType = f64>,
    Y: Ranged<ValueType = f64>,
{
    let points = plot
        .experimental
        .q
        .iter()
        .zip(&plot.experimental.intensity)
        .filter_map(|(&q, &intensity)| {
            (intensity > 0.0 && (!log_x || q > 0.0)).then_some((q, intensity))
        });
    if plot.experimental.has_errors {
        let error_bars = plot
            .experimental
            .q
            .iter()
            .zip(&plot.experimental.intensity)
            .zip(&plot.experimental.sigma)
            .filter_map(|((&q, &intensity), &sigma)| {
                if q <= 0.0 || intensity <= 0.0 || !sigma.is_finite() || sigma < 0.0 {
                    return None;
                }
                let minimum = (intensity - sigma).max(intensity * 1.0e-8);
                Some(ErrorBar::new_vertical(
                    q,
                    minimum,
                    intensity,
                    intensity + sigma,
                    BLACK.mix(0.55),
                    5,
                ))
            });
        chart.draw_series(error_bars).map_err(plot_error)?;
    }
    chart
        .draw_series(points.map(|(q, intensity)| Circle::new((q, intensity), 2, BLACK.filled())))
        .map_err(plot_error)?;
    Ok(())
}

fn draw_model_lines<'a, DB, X, Y>(
    chart: &mut ChartContext<'a, DB, Cartesian2d<X, Y>>,
    plot: &DiagnosticPlot,
    log_x: bool,
) -> Result<()>
where
    DB: DrawingBackend + 'a,
    DB::ErrorType: std::fmt::Debug,
    X: Ranged<ValueType = f64>,
    Y: Ranged<ValueType = f64>,
{
    for (index, curve) in plot.curves.iter().enumerate() {
        let series =
            plot.experimental
                .q
                .iter()
                .zip(&curve.intensity)
                .filter_map(|(&q, &intensity)| {
                    (intensity > 0.0 && (!log_x || q > 0.0)).then_some((q, intensity))
                });
        chart
            .draw_series(LineSeries::new(
                series,
                Palette99::pick(index + 1).stroke_width(2),
            ))
            .map_err(plot_error)?;
    }
    Ok(())
}

fn draw_experimental_kratky<DB, X, Y>(
    chart: &mut ChartContext<'_, DB, Cartesian2d<X, Y>>,
    plot: &DiagnosticPlot,
) -> Result<()>
where
    DB: DrawingBackend,
    DB::ErrorType: std::fmt::Debug,
    X: Ranged<ValueType = f64>,
    Y: Ranged<ValueType = f64>,
{
    let scale = low_q_i0(&plot.experimental.q, &plot.experimental.intensity).unwrap_or(1.0);
    let series = plot
        .experimental
        .q
        .iter()
        .zip(&plot.experimental.intensity)
        .filter_map(|(&q, &intensity)| {
            (q > 0.0 && intensity.is_finite() && intensity > 0.0)
                .then_some((q, q * q * intensity / scale))
        });
    if plot.experimental.has_errors {
        let error_bars = plot
            .experimental
            .q
            .iter()
            .zip(&plot.experimental.intensity)
            .zip(&plot.experimental.sigma)
            .filter_map(|((&q, &intensity), &sigma)| {
                if q <= 0.0 || intensity <= 0.0 || !sigma.is_finite() || sigma < 0.0 {
                    return None;
                }
                let factor = q * q / scale;
                Some(ErrorBar::new_vertical(
                    q,
                    factor * (intensity - sigma).max(0.0),
                    factor * intensity,
                    factor * (intensity + sigma),
                    BLACK.mix(0.55),
                    5,
                ))
            });
        chart.draw_series(error_bars).map_err(plot_error)?;
    }
    chart
        .draw_series(LineSeries::new(series, BLACK.stroke_width(2)))
        .map_err(plot_error)?;
    Ok(())
}

fn positive_q_range(curve: &ExperimentalCurve) -> Option<(f64, f64)> {
    let values = curve.q.iter().copied().filter(|value| *value > 0.0);
    let min = values.clone().fold(f64::INFINITY, f64::min);
    let max = values.fold(0.0, f64::max);
    (min.is_finite() && max > min).then_some((min, max))
}

fn intensity_range(plot: &DiagnosticPlot) -> Option<(f64, f64)> {
    let mut values = plot
        .experimental
        .intensity
        .iter()
        .copied()
        .chain(
            plot.curves
                .iter()
                .flat_map(|curve| curve.intensity.iter().copied()),
        )
        .filter(|value| *value > 0.0 && value.is_finite());
    let min = values.next()?;
    let min = values.fold(min, f64::min);
    let max = plot
        .experimental
        .intensity
        .iter()
        .copied()
        .chain(
            plot.curves
                .iter()
                .flat_map(|curve| curve.intensity.iter().copied()),
        )
        .filter(|value| *value > 0.0 && value.is_finite())
        .fold(min, f64::max);
    Some((min.max(max * 1.0e-8), max.max(min * 1.001)))
}

fn kratky_range(plot: &DiagnosticPlot) -> Option<(f64, f64)> {
    let values = plot
        .experimental
        .q
        .iter()
        .zip(&plot.experimental.intensity)
        .filter_map(|(&q, &i)| {
            let scale = low_q_i0(&plot.experimental.q, &plot.experimental.intensity)?;
            (q > 0.0 && i.is_finite() && i > 0.0).then_some(q * q * i / scale)
        })
        .chain(plot.curves.iter().flat_map(|curve| {
            let scale = low_q_i0(&plot.experimental.q, &curve.intensity).unwrap_or(1.0);
            plot.experimental
                .q
                .iter()
                .zip(&curve.intensity)
                .filter_map(move |(&q, &i)| {
                    (q > 0.0 && i.is_finite() && i > 0.0).then_some(q * q * i / scale)
                })
        }));
    let max = values.fold(0.0, f64::max);
    Some((0.0, (max * 1.12).max(1.0e-8)))
}

fn low_q_i0(q: &[f64], intensity: &[f64]) -> Option<f64> {
    let mut values = q
        .iter()
        .zip(intensity)
        .filter_map(|(&q, &intensity)| {
            (q > 0.0 && intensity.is_finite() && intensity > 0.0).then_some(intensity)
        })
        .take(5)
        .collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(values[values.len() / 2])
}

fn format_optional(value: Option<f64>) -> String {
    value.map_or_else(|| "n/a".into(), |value| format!("{value:.3}"))
}

fn plot_error<E: std::fmt::Debug>(error: E) -> SaxsError {
    SaxsError::Other(anyhow::anyhow!("plot rendering failed: {error:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn writes_svg_diagnostic_plot() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("plot.svg");
        let plot = DiagnosticPlot {
            title: "test".into(),
            experimental: ExperimentalCurve {
                q: vec![0.01, 0.02, 0.03],
                intensity: vec![10.0, 5.0, 2.0],
                sigma: vec![0.1, 0.1, 0.1],
                has_errors: true,
            },
            curves: vec![PlotCurve {
                name: "model".into(),
                intensity: vec![9.0, 5.2, 2.1],
            }],
            ..DiagnosticPlot::default()
        };
        plot.save_svg(&path).unwrap();
        assert!(fs::read_to_string(path).unwrap().contains("<svg"));
        let png = directory.path().join("plot.png");
        plot.save_png(&png).unwrap();
        assert!(fs::metadata(png).unwrap().len() > 100);
    }
}
