# B17 K=1 single-wavelength phase recovery — Pyramid64 default

This folder contains the **validated K=1 single-wavelength** reference implementation. The optical wavelength is assumed **known**. No blind spectral identification or joint multi-wavelength recovery is performed here.

## Default algorithm

1. Optimize a 64×64 phase parameterization for 35 Armijo-GD iterations.
2. Increase phase parameterization to 128×128 for 35 iterations.
3. Increase to 256×256 for 40 iterations.
4. Increase to native 512×512 for 40 iterations.
5. Run native 512×512 L-BFGS-B for **up to 150 iterations**.

GD steps 1–100 are unregularized; GD steps 101–150 and L-BFGS use periodic circular regularization with weight **0.015**. The hardware is unchanged: original B17 owner's LUT and width-map archive, 3800×3800 propagation grid, 1429×1429 metasurface, native 512×512 phase, 3000×3000 camera.

The interpolation affects only the **optimization variable**, not the physical propagation or sensor.

## Run the validated K=1 examples

From repository root:

```bash
python -m pip install numpy scipy h5py scikit-image matplotlib

# Self-tests: circular-prior finite difference and pyramid interpolation adjoint
python hyperspectral_phase/single_b17_v1/cloud_k1_pyramid64_default.py --self-test

# K=1 independent test (single wavelength, 1% Gaussian sensor noise)
python hyperspectral_phase/single_b17_v1/cloud_k1_pyramid64_default.py --scene moon --wavelength 550 --noise-pct 1

# Noise stress: 3% Gaussian noise at 660 nm
python hyperspectral_phase/single_b17_v1/cloud_k1_pyramid64_default.py --scene camera_rot90 --wavelength 660 --noise-pct 3
```

Supported examples: `moon`, `brick`, `chelsea`, `camera_rot90`; wavelengths **440, 550 or 660 nm**; `--noise-pct 3` is restricted to **660 nm**. The source `capstone_optics/05_NN_metacam-main.zip` must remain at its existing repository location.

Saved outputs: `b17_k1_pyramid64_stress_outputs/<scene>_<wavelength>nm_noise<noise>/`. The output includes `evaluation_circular.json`, `five_panels_circular.png`, and recovered phase files. Ground truth is read during synthetic observation generation and **after inverse fitting** for metric calculation, never passed to the optimizer.

## Source and validation

The K=1 implementation imports the following sibling files: `cloud_k1_astronaut_multires.py` (pyramid), `cloud_k1_endpoint_robust.py` (loss, circular prior, L-BFGS and evaluation), `k1_cloud_batch.py` / `full3800_common.py` (B17 original data and optics), and `../hyperspectral_1_b17_core.py` (forward/adjoint).

GitHub Actions evidence from the experimental branch:

- [16-case independent scenes, rotations, wavelengths and noise stress](https://github.com/ericahanyang/capstone_optics/actions/runs/42843668774): **16/16 pass** (12 at 1% sensor noise; 4 at 3%).
- [Four matched Cameraman/Coins generalization cases](https://github.com/ericahanyang/capstone_optics/actions/runs/42831938165): **4/4 pass**, improved or numerically equivalent phase RMSE.
- [Astronaut 660 nm pyramid comparison](https://github.com/ericahanyang/capstone_optics/actions/runs/42820625037): zero-init 0.204911 rad phase RMSE vs Pyramid64 **0.119159 rad**.

These are **synthetic measurements** using a matched B17 forward/inverse model; they do not establish robustness to hardware calibration errors or guarantee reconstruction of arbitrary input imagery. For scientifically independent validation, freeze algorithm settings and extend to new images/seeds/optical model mismatches.

## Scope and safety

- K=1 only: no changes to `hyperspectral_2_multi.py`, `hyperspectral_3_continuum.py`, the experiments' original scripts, or meta/sensor hardware.
- Keep historical baseline code intact for numerical comparison.
- Do not use any true-phase maps to initialize, tune, or choose between candidate inverse solutions.
- Do not commit generated sensor arrays, LUT/width-map material or experimental truth arrays.
