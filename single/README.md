# Single K=1

Run `python -m single.run single --scene tank --true-wavelength 660 --mode known --wavelength 660 --output single/results/tank660` or choose camera, boat, triple, normal, variable, tilt, double. Blind: `--mode blind` without `--wavelength`. `--geometry-seed` and `--seed` independently control RBC geometry and noise. The inverse receives sensor only and known wavelength if applicable. See docs/REPRODUCTION.md. Full inverse migration validation remains pending.
