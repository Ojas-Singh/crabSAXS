use crabsaxs::scattering::{debye_intensity, HydrogenMode};
use crabsaxs::spherical::{multipole_intensity, MultipoleParams};
use crabsaxs::{Atom, Element, Structure};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use nalgebra::Vector3;

fn synthetic_structure(atom_count: usize) -> Structure {
    let atoms = (0..atom_count)
        .map(|index| {
            let angle = index as f64 * 0.618_033_988_75;
            Atom {
                pos: Vector3::new(12.0 * angle.cos(), 12.0 * angle.sin(), index as f64 * 0.08),
                element: match index % 4 {
                    0 => Element::C,
                    1 => Element::N,
                    2 => Element::O,
                    _ => Element::S,
                },
                occupancy: 1.0,
                is_hetatm: false,
                atom_name: format!("A{index}"),
                residue_name: "BEN".into(),
                chain_id: "A".into(),
                residue_id: index as isize,
            }
        })
        .collect();
    Structure { atoms }
}

fn bench_scattering(c: &mut Criterion) {
    let q: Vec<f64> = (1..=50).map(|index| index as f64 * 0.01).collect();
    let mut group = c.benchmark_group("scattering_by_atom_count");
    for atom_count in [16, 64, 256] {
        let structure = synthetic_structure(atom_count);
        group.bench_with_input(
            BenchmarkId::new("debye", atom_count),
            &structure,
            |benchmark, structure| {
                benchmark.iter(|| debye_intensity(black_box(structure), black_box(&q)))
            },
        );
        group.bench_with_input(
            BenchmarkId::new("multipole", atom_count),
            &structure,
            |benchmark, structure| {
                benchmark.iter(|| {
                    multipole_intensity(
                        black_box(structure),
                        black_box(&q),
                        &MultipoleParams {
                            l_max: 16,
                            hydrogen_mode: HydrogenMode::Implicit,
                        },
                    )
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_scattering);
criterion_main!(benches);
