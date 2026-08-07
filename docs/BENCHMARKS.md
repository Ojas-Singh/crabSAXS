# Black-box parity benchmark protocol

The benchmark claims in the README use `saxs-compare` report schema
`crabsaxs-pepsi-parity-v3`. Pepsi-SAXS is executed only as a black box; its
executable is not inspected. Machine-readable reports are committed in
[`benchmark-reports`](benchmark-reports).

## Matching policy

- Both programs run as fresh subprocesses with two warmups and ten measured
  runs.
- Timings include process startup, parsing, computation or fitting, and curve
  output.
- Non-water HETATM records are included and the selected alternate conformer
  is assigned full occupancy.
- Smearing is disabled. Fit comparisons use a fitted constant background and
  coarse adaptive hydration grids.
- Atomic-component curves receive one least-squares scale before error
  calculation. Independently fitted curves receive no additional alignment.
- NRMSE is RMSE divided by the reference intensity range. The report also
  records relative L2 error, mean and maximum pointwise relative differences,
  medians, median absolute deviations, input/binary SHA-256 hashes, host
  architecture, thread count, and Rust version.

## Current report summary

Measurements on the six-core `aarch64` Mac used for development, with Rust
1.97.0 and an Intel Pepsi-SAXS executable running under Rosetta:

| Workflow | NRMSE | crab median | Pepsi median | crab/Pepsi result |
| --- | ---: | ---: | ---: | --- |
| 1UBQ atomic component | 0.625% | 13.10 ms | 41.45 ms | 3.16× faster |
| 2DFC atomic component | 0.0631% | 28.44 ms | 42.38 ms | 1.49× faster |
| 1AON atomic component | 0.0331% | 2.206 s | 2.708 s | 1.23× faster |
| 2DFC/Xylanase fit, stride 20 | 0.0424% | 24.86 ms | 61.79 ms | 2.49× faster |
| 2DFC/Xylanase fit, full q | 0.0598% | 160.86 ms | 60.86 ms | 2.64× slower |

For the default complete fit, crabSAXS achieved reduced χ² 1.14837 versus
Pepsi-SAXS 1.14438. Its fitted excluded-volume radius was 1.60476 Å versus
1.59453 Å, and hydration contrast was 0.01536 versus 0.01603 e/Å³.

## Six-component audit

Pepsi's public output exposes the vacuum, excluded-volume, hydration, and
three cross-term profiles. Each shape receives an independent scalar before
NRMSE is measured; this separates component-model agreement from fitted
parameter agreement.

| Partial profile | 1UBQ NRMSE | 2DFC NRMSE |
| --- | ---: | ---: |
| Atomic | 0.625% | 0.0631% |
| Excluded volume | 0.598% | 0.0656% |
| Hydration | 0.811% | 0.899% |
| Atomic/excluded cross | 0.611% | 0.0641% |
| Atomic/hydration cross | 0.723% | 0.884% |
| Excluded/hydration cross | 0.753% | 0.890% |

Every audited component passes the 1% gate. The 58,870-atom 1AON report
records an explicit component-audit skip to avoid constructing a large
diagnostic basis; its end-to-end atomic curve and runtime remain tested.

Regenerate machine-readable JSON reports rather than copying these timings to
a different machine:

```sh
cargo build --release --bins
target/release/saxs-compare \
  --pepsi-bin ./Pepsi-SAXS \
  --pdb examples/data/benchmarks/2dfc.pdb \
  --exp examples/data/benchmarks/xylanase.dat \
  --repetitions 10 \
  --output docs/benchmark-reports/2dfc-fit-fast.json
```

The stride-20 basis is the validated default performance path. Use
`--q-sampling-stride 1` when an uninterpolated diagnostic basis is required;
the committed full-q report intentionally records that it misses the runtime
gate on this machine instead of hiding that tradeoff.
