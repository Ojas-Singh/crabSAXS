# References

This project reimplements published, open methodology for computing
theoretical SAXS profiles from atomic structures. It does not use or
reference any closed-source code. Keep this list updated as the agent
consults sources during implementation — note the specific equation/table
used, not just the paper title, so choices remain traceable.

## Core algorithm family (spherical harmonics / multipole expansion)

- Grudinin, S., Garkavenko, M., & Kazennov, A. (2017). "Pepsi-SAXS: an
  adaptive method for rapid and accurate computation of small-angle X-ray
  scattering profiles." Acta Crystallographica D73, 449–464.
  DOI: 10.1107/S2059798317005745. Specific uses: equations (1)–(5), the
  vacuum/excluded/hydration multipole decomposition; equations (6)–(8) and
  Tables 1–2, atomic and unified-group form factors; equations (9)–(13),
  Fraser dummy atoms and global radius scaling; section 2.3, adaptive
  hydration grid; equation (14), adaptive expansion order; and equations
  (15)–(16), weighted fitting. The Supporting Information equations (1)–(6)
  provide the six partial intensities and analytical constant-background fit.
- Svergun, D., Barberato, C., & Koch, M. H. J. (1995). "CRYSOL - a program to
  evaluate X-ray solution scattering of biological macromolecules from
  atomic coordinates." Journal of Applied Crystallography, 28, 768-773. —
  the original spherical-harmonics SAXS-from-structure method; defines the
  excluded-volume (c1-like) and hydration-shell (c2-like) fitting parameters
  this project's M4/M5 are based on.
- Schneidman-Duhovny, D., Hammel, M., Tainer, J.A., & Sali, A. (2013). "FoXS:
  a web server for rapid computation and fitting of SAXS profiles." Nucleic
  Acids Research. — a Debye-formula-based alternative worth comparing
  behavior/results against for validation purposes.

## Atomic form factors

- International Tables for Crystallography, Volume C, Table 6.1.1.4 —
  standard source of Cromer-Mann 5-Gaussian coefficients for atomic X-ray
  scattering form factors. The exact machine-readable coefficients used for
  H, C, N, O, P, S, Na, Mg, Cl, K, Ca, Fe, Zn, and I are the ESRF DABAX
  `f0_InterTables.dat` transcription of that table:
  https://ftp.esrf.fr/pub/scisoft/DabaxFiles/f0_InterTables.dat.
- Cromer, D. T. & Mann, J. B. (1968). "X-ray scattering factors computed
  from numerical Hartree-Fock wave functions." Acta Crystallographica A24,
  321-324. — original derivation.

## Excluded volume / hydration shell modeling

- Fraser, R. D. B., MacRae, T. P., & Suzuki, E. (1978). "An improved method
  for calculating the contribution of solvent to the X-ray diffraction
  pattern of biological molecules." Journal of Applied Crystallography, 11,
  693-694. — Gaussian dummy-atom excluded-volume approach. The H/C/N/O
  volumes used for standard protein united groups are 5.15/16.44/2.49/9.13
  Å³. The complete H/C/N/O/Na/Mg/P/S/Cl/K/Ca/Fe/Zn/I table used here is
  reproduced in the official MULCh manual, table 3 (page 11):
  https://smb-research.smb.usyd.edu.au/NCVWeb/docs/readme.pdf.
- Svergun et al. 1995 (above) — hydration shell contrast parameterization.
- Bondi, A. (1964). "van der Waals Volumes and Radii." Journal of Physical
  Chemistry, 68(3), 441–451. — source family for the H/C/N/O/S van der Waals
  radii used to construct the probe-expanded molecular envelope. Conventional
  crystallographic/ionic extensions are used for P and the supported ions and
  are kept together in `solvent.rs` for traceability.
- Lytje, K. & Pedersen, J. S. (2025). "Small-angle X-ray scattering profile
  calculation for high-resolution models of biomacromolecules." Journal of
  Applied Crystallography 58, 1332–1346.
  DOI: 10.1107/S160057672500562X. Equations (3)–(4) give the corrected Fraser
  transform and q-dependent global volume scale. Section 3.3.2 documents
  conversion/typographical errors in older printed derivations and reports
  that current CRYSOL and Pepsi-SAXS calculations use the corrected form.

## Debye formula

- Debye, P. (1915). "Zerstreuung von Röntgenstrahlen." Annalen der Physik,
  46, 809. — the original pairwise scattering formula this project's
  baseline (M3) implementation is named after.

## Structure and carbohydrate classification

- wwPDB Chemical Component Dictionary, public component definitions and
  three-letter identifiers: https://www.wwpdb.org/data/ccd. The conservative
  carbohydrate residue-code set in `structure.rs` is derived from the common
  saccharide components in the CCD; unknown HETATM codes remain opt-in via
  `--include-hetatm` and are reported by `inspect`.

## Rust implementation choices

- `pdbtbx` — PDB/mmCIF parsing.
- `nalgebra` / `ndarray` — linear algebra.
- `rayon` — data parallelism.
- `argmin` + `argmin-math` — optimization for the fitting step (M5).
- Real spherical harmonics use normalized associated-Legendre recurrences;
  spherical Bessel functions use Miller downward recurrence normalized by the
  analytical `j0`/`j1` values. Keeping these small kernels in-tree avoids
  convention conversion and permits contiguous packed storage. The
  implementation is covered by analytical limits, two-atom comparisons, a
  96-atom mixed-element `q=0.005…0.5 Å⁻¹` Debye regression, and independent
  Debye checks of all six adaptive-grid solvent terms.

## Note on Pepsi-SAXS itself

Pepsi-SAXS is closed-source freeware (free for academic/non-commercial use).
This project never decompiles, disassembles, or inspects its source or binary.
The optional black-box harness may execute a user-provided binary and compare
its public curve/timing output. All implementation choices trace to the
published papers above rather than to the executable.
