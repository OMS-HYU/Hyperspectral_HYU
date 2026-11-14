# K=1 GD-only continuation — 2026-11-14

New isolated branch based on single-recommendation-benchmark-20261113.
Original solver, historical results, MAIN, B17 LUT and optical grid unchanged.
This is an implementation/benchmark harness, NOT new full-grid result evidence.

## Purpose
The legacy spectral and pyramid CLI routes include L-BFGS. This separate CLI
runs only GD or FFT-preconditioned GD and produces the existing evaluation format.

| Method | Accepted updates | Strategy |
|---|---:|---|
| full100 | 100 | Existing full-grid unregularized GD baseline |
| full150 | 150 | Full-grid GD; circular prior after 100 |
| pyramid150 | 150 | Existing 64/128/256/512 parameter pyramid, 35/35/40/40; prior after 100 |
| spectral150 | 150 | First 100 identical full GD; FFT preconditioned GD and circular prior after 100 |

All methods use same sensor, original B17, same known/estimated wavelength.
Compare full100 vs full150 (extra steps AND prior); full150 vs spectral150
(preconditioner AND step cap); full150 vs pyramid150 (parameter strategy AND step rules).
Do not attribute a difference to just one factor without a separate ablation.
Historical spectral preconditioner was calibrated at 660 nm, heuristic elsewhere.
No L-BFGS calls. Pyramid internal lift is real-valued; all optical phase inputs
and exported phases use [0,2pi). No truth enters any inversion.
If an algorithm fails line search or any other step, record error, don't report success.
Goal: existing full RMSE <=0.15 rad, sensor L2 <=2.5%, wavelength error <=0.5 nm,
all present RBC ROI/layers/groups <=0.15, and complete iteration budget.

## Reproduce (PowerShell, project root)

```powershell
git fetch origin
git switch --track origin/chatgpt/single-pure-gd-20261114
python -m unittest tests.test_single_pure_gd
python -m 싱글.코드.pure_gd --help

# A new camera scene, not guaranteed identical to a historical archive:
python -m 싱글.run generate --scene camera --true-wavelength 650 --seed 672906 --output 싱글/결과/gd01_camera650_obs
python -m 싱글.코드.pure_gd --sensor 싱글/결과/gd01_camera650_obs/sensor_only.npz --truth 싱글/결과/gd01_camera650_obs/private_truth.npz --mode known --wavelength 650 --output 싱글/결과/gd01_camera650_known

# Exact historical RBC sensor: use original frozen archive and verify hash.
python -m 싱글.run reference --archive "D:/capstone/B17_K1_RBC_0_2pi_Spectral_Periodic_LBFGS100_FULL_20261112.zip" --generate-only --output 싱글/결과/gd01_rbc660_ref
python -m 싱글.코드.pure_gd --sensor 싱글/결과/gd01_rbc660_ref/observation/sensor_only.npz --truth 싱글/결과/gd01_rbc660_ref/observation/private_truth.npz --mode known --wavelength 660 --output 싱글/결과/gd01_rbc660_known

# Same camera observation, blind; no true wavelength supplied to inverse.
python -m 싱글.코드.pure_gd --sensor 싱글/결과/gd01_camera650_obs/sensor_only.npz --truth 싱글/결과/gd01_camera650_obs/private_truth.npz --mode blind --output 싱글/결과/gd01_camera650_blind
```

Use a new output path for every run. Existing results cannot be overwritten.
Run methods sequentially to limit RAM. Use --methods full150 spectral150 to
restrict a comparison. For real sensor omit --truth; no phase RMSE is then available.
The root comparison.json contains each method status and evaluation; each child
contains recovered.npz, predicted.npz, recovery.json, history.json, and optional
evaluation.json, five_panel.png, convergence.png. Compare sensor_sha256,
wavelength, B17 fingerprints and geometry before interpreting any results.

The first local numerical check used the same descent function on a 16x16
periodic toy objective: 20 accepted updates, objective 260.61860977 to
3.9316371e-18; upward direction rejected. This is NOT B17 validation.
A first test using 1-cos and 50 updates reached numerical cancellation and
Armijo exhaustion; stable chord-squared test avoids cancellation. Real runs
retain strict line search and report failures rather than claim a pass.
Full-grid experiments have not been executed in this session.
