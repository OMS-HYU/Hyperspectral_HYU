# Reproduction

1. Install requirements, supply exact B17 ZIP and run `python -m tests.test_contract --full-forward`.
2. Known camera/tank 660: see 싱글/README.md. Run S1 camera, S2 triple RBC, S3 blind camera 550, S4 blind triple 660; preserve each output separately.
3. Evaluate only after inverse; each run records history, recovery/evaluation JSON, phase/sensor arrays, five-panel figure. No truth in optimization. Ground truth can be used for synthetic measurement generation and post-hoc metrics only.
4. Historical fixed-GD benchmark has distinct solver details; do not report its numbers as migrated performance. Benchmark archive is retained separately, see benchmarks/single_reference/README.md.
5. Multi/continuum are forward foundations, not inverse. Never claim wavelength recovery from a supplied ground-truth plan.
