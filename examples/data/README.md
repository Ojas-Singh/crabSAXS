# Example and test data

Committed fixtures are deliberately tiny and deterministic:

- `test_structure.pdb` covers model selection, alternate locations,
  occupancy, HETATM, and water filtering.
- `test_structure.cif` covers mmCIF parsing and first-model selection.
- `malformed.pdb` verifies malformed numeric input is rejected.
- `small.pdb` is the two-carbon analytical Debye fixture.

Larger public validation data is downloaded into the ignored
`examples/data/benchmarks/` directory by `scripts/fetch_benchmarks.sh`. The
script pins URLs and SHA-256 checksums for 1UBQ, 2DFC, the public Xylanase SAXS
curve, and 1AON. These files are not committed because of their size.

Coordinate structures come from the RCSB Protein Data Bank. The Xylanase
curve comes from the public SAS Tutorials protein tutorial; exact URLs and
checksums live in the fetch script.
