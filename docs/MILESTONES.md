# Milestones — detailed task breakdown

Intended audience: an autonomous coding agent implementing this crate
incrementally. Work top to bottom. Each milestone should compile, pass its
own tests, and not regress earlier tests before moving on.

## M1 — Structure I/O (`src/structure.rs`)

- [x] Define `Element` enum (or wrap `pdbtbx`'s element type) covering at
      least H, C, N, O, S, P and common metal ions.
- [x] Define `Atom { pos: nalgebra::Vector3<f64>, element: Element, occupancy: f64, is_hetatm: bool }`.
- [x] Define `Structure { atoms: Vec<Atom> }` with a `from_pdb_file(path) -> Result<Structure>` constructor using `pdbtbx`.
- [x] Decide and document a policy for: alternate conformations (take highest
      occupancy), waters/HETATM (exclude by default, flag to include),
      multiple models/NMR ensembles (use model 1 by default).
- [x] Unit test: parse a small known PDB (e.g. lysozyme, 6LYZ or similar) and
      assert atom count matches expectation from the file's ATOM records.

## M2 — Form factors (`src/formfactor.rs`)

- [x] Implement the Cromer-Mann 5-Gaussian form factor:
      `f(q) = sum_{i=1..4} a_i * exp(-b_i * (q/4pi)^2) + c`
      (confirm exact q-vs-s convention used — SAXS literature sometimes uses
      s = q/4pi, sometimes q/2pi; be explicit and document the choice, since
      this is the single most common source of bugs when porting between
      papers/tools).
- [x] Store Cromer-Mann coefficients (a1..a4, b1..b4, c) for at least H, C, N,
      O, S, P as a static table. Cite the source table in a doc comment
      (International Tables for Crystallography Vol. C is the standard
      reference).
- [x] Implement an "implicit hydrogen" / united-atom correction: heavy atoms
      absorb the form factor contribution of attached (unresolved) hydrogens
      based on residue/atom-name heuristics, OR support explicit hydrogens if
      present in the structure. Document which mode is active.
- [x] Unit test: form factor at q=0 for carbon should equal its atomic number
      (6.0) within the fit's known small residual `c` term.

## M3 — Debye formula, vacuum only (`src/scattering.rs`)

- [x] Implement `debye_intensity(structure: &Structure, q_values: &[f64]) -> Vec<f64>`:
      for each q, `I(q) = sum_i sum_j f_i(q) * f_j(q) * sinc(q * r_ij)`
      where `sinc(x) = sin(x)/x` (with `sinc(0) = 1`), `r_ij = |pos_i - pos_j|`.
- [x] Parallelize the outer loop (over q, or over the i-index of atom pairs)
      with `rayon`.
- [x] Note complexity is O(N^2 * |Q|) — fine for validation on small
      structures (a few hundred atoms), too slow for large ones (this is
      what M6 fixes).
- [x] Integration test: compare against a hand-computed or literature I(q)
      curve for a trivial case (e.g. two-atom "molecule" has a closed-form
      I(q), or use a published reference curve for a small test PDB — cite
      the source of any reference curve used in the test file).

## M4 — Excluded volume + hydration shell (`src/solvent.rs`)

- [x] Implement the standard 3-term scattering-amplitude decomposition used
      by CRYSOL/Pepsi-SAXS-style methods:
      `A(q) = A_vacuum(q) - G(q,c1) * A_excluded_solvent(q) + c2 * A_hydration_shell(q)`
      where `c1` is the global excluded-volume radius scale (fit in
      0.95–1.05) and `G` is its published q-dependent factor, while
      `c2` scales hydration shell electron density contrast (fit, typically
      small and can be negative or positive depending on convention).
- [x] Model the excluded volume per atom using a Gaussian-sphere
      approximation (Fraser/Fedorov-style dummy atom volumes) — store
      per-element average volumes as a static table.
- [x] Model the hydration shell with the published adaptive 3–4 Å grid,
      12 Å padding, and Rg-dependent 3–5 Å width. A deterministic 1 Å
      exterior solvent-accessible grid and legacy analytic approximation are
      also available and documented.
- [x] Unit test: c1=1, c2=0 should reduce to a physically sensible
      "vacuum minus mean solvent" curve; verify no NaNs/negative intensities
      over a reasonable q-range for a real test structure.

## M5 — Fitting (`src/fit.rs`)

- [x] Read experimental `.dat` file: three columns `q, I_exp(q), sigma(q)`
      (whitespace or comma separated; skip comment/header lines starting
      with `#`).
- [x] Define the reported reduced chi-squared using the Pepsi-SAXS convention:
      `chi2 = (1/(N-1)) * sum_k ((scale * I_theory(q_k) + bg - I_exp(q_k)) / sigma(q_k))^2`.
- [x] Solve scale and background analytically by weighted least squares for
      each solvent candidate, then use `argmin`/Nelder-Mead to minimize chi2
      over `(c1, c2)`. Use the published Pepsi ranges `c1 in [0.95,1.05]`
      and adaptive-shell relative contrast `c2 in [-0.045,0.1]`; the legacy
      analytic shell retains its separate amplitude range.
- [x] CLI/API surface returns fitted parameters + final chi2 + the fitted
      theoretical curve resampled onto the experimental q-grid.
- [x] Integration test: fit against a synthetic experimental curve generated
      by adding known Gaussian noise to a theoretical curve with known
      scale/background — recovered parameters should be close to ground
      truth.

## M6 — Performance: spherical harmonics / multipole expansion (`src/spherical.rs`)

- [x] Implement the standard spherical-harmonics expansion of the scattering
      amplitude (expand `exp(i q . r)` in spherical Bessel functions and
      spherical harmonics, truncate at some `L_max`), giving O(N * L_max^2)
      scaling instead of O(N^2) — this is the technique underlying
      CRYSOL/Pepsi-SAXS's speed advantage over brute-force Debye.
- [x] Needs: spherical Bessel functions `j_l(x)`, real spherical harmonics
      `Y_lm(theta, phi)`. Consider existing crates before writing these from
      scratch.
- [x] Benchmark against M3's brute-force Debye on the same structure/q-grid;
      confirm numerical agreement within a small tolerance before trusting
      the fast path, and only then make it the default.
- [x] Add a Criterion benchmark comparing both paths across a few structure
      sizes.

## M7 — CLI polish (`src/main.rs`)

- [x] `crabsaxs calc structure.pdb --q-max 0.5 --q-step 0.005 -o out.dat`
- [x] `crabsaxs fit structure.pdb data.dat -o fit/`
- [x] Human-readable summary to stdout (chi2, fitted params); machine
      readable `--json` flag for scripting.
- [x] `--help` output should be self-explanatory; add usage examples to
      README once CLI is stable.

## Ongoing

- Keep `docs/REFERENCES.md` updated with any paper/table your agent consults
  while implementing a milestone, including the specific equation/table
  number used, so choices are traceable later.
- Prefer small, focused commits per checklist item above over large
  multi-milestone commits.
