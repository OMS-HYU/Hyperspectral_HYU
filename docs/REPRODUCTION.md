# Reproduction

1. Install requirements, the original extracted B17 calibration is included; run `python -m tests.test_contract --full-forward`.
2. RBC only: `python -m 싱글.run single --output 싱글/결과/RBC_660` (triple, Known 660 nm, seed 672906, full GD100+Spectral100). Then normal/variable/tilt/double and independent geometry seeds; blind triple 660 remains experimental. Preserve each output separately. For exact historical sensor/checkpoint use the archived benchmark, not the newly generated RBC model.
3. Evaluate only after inverse; each run records history, recovery/evaluation JSON, phase/sensor arrays, five-panel figure. No truth in optimization. Ground truth can be used for synthetic measurement generation and post-hoc metrics only.
4. Historical fixed-GD benchmark has distinct solver details; do not report its numbers as migrated performance. Benchmark archive is retained separately, see benchmarks/single_reference/README.md.
5. Multi/continuum are forward foundations, not inverse. Never claim wavelength recovery from a supplied ground-truth plan.
