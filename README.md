# Hyperspectral_HYU

Single-shot metasurface-based phase and hyperspectral reconstruction | Hanyang University ERICA.

## Research tracks

| Track | Entry point | Current status |
|---|---|---|
| Single (K=1) | `python -m 싱글.run` | RBC research baseline: Full GD100 + Spectral Periodic L-BFGS100; general-image Pyramid64 candidate; auto/compare; integrated inverse regression pending |
| Multi (K=2–3) | `python -m 멀티.run` | full-grid incoherent forward and configuration; inverse is not yet implemented in this clean foundation |
| Continuum | `python -m 연속.run` | quadrature and full-grid intensity forward; inverse is not yet implemented |

One shared original B17 engine. Native propagation 3800², metasurface 1429², object 512², sensor 3000², input pitch 350 nm, camera 2 µm, z1 400 µm, z2 5 mm, wavelength 440–660 nm. Actual phase arrays are canonical [0, 2π). No ground truth is provided to an inverse.

## Quick start

```bash
python -m pip install -r requirements.txt
python -m tests.test_contract --full-forward
python -m 싱글.run single --output 싱글/결과/RBC_660
# full synthetic RBC run; auto selects spectral
python -m 싱글.run single --scene camera --method pyramid --output 싱글/결과/camera660
python -m 멀티.run --wavelengths 500 660 --weights 1 1 --output 멀티/결과/K2_plan
python -m 연속.run --start 440 --stop 660 --step 10 --output 연속/결과/plan
```

Original B17 calibration is included in the extracted `데이터/B17/05_NN_metacam-main/assets/data/Fab/B17/` directory. The loader verifies both original MAT SHA-256 values. A matching ZIP or explicit B17_SOURCE (B17_ZIP also accepted) is supported. See [data instructions](데이터/B17/README.md). No surrogate replaces missing B17. Generated large output is excluded from Git. See [research status](docs/RESEARCH_PROGRESS.md), [experiment guide](docs/REPRODUCTION.md), and [data provenance](docs/DATA_SOURCES.md).

## Single-track decision (2026-11-13)

RBC remains the primary research track: Full-resolution GD100 + Spectral Periodic L-BFGS100 is its baseline; the 250/300-step pyramid variants did not improve RBC ROI RMSE. General images now select Pyramid64 by default; compare evaluates both on the same observation. The new port is not yet full-run validated. [Korean single guide](싱글/가이드북.md). See [decision and metrics](싱글/연구기록/개발방향.md).

## Historical benchmark, not a claim about the migrated solver

660 nm synthetic independent 100-RBC, 50-group field: Spectral Periodic after fixed GD100 had ROI RMSE 0.077086 rad, full RMSE 0.029155 rad, 49/50 groups passing 0.15 rad. One group failed. See [benchmark](benchmarks/single_reference/README.md). Spatial-Spectral was not adopted. These results are from a historical experiment, not new full inverse verification.

Source migration: original private repository MAIN 518e8932bf3b548acf1c1145820278edf385a6d7. Old repository retained unchanged; this repository has independent history. Third-party dataset rights are not granted by this README.
