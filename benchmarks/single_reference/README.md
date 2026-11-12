# Historical fixed 660 nm RBC benchmark

| Method | Full RMSE rad | ROI RMSE rad | Groups pass |
|---|---:|---:|---:|
| Original bounded | .053314 | .174806 | 24/50 |
| Periodic | .030815 | .084598 | 48/50 |
| Spectral Periodic | .029155 | .077086 | 49/50 |
| Spatial-Spectral (not adopted) | .029246 | .077579 | 49/50 |

Historical full grid 3800/1429/512/3000, fixed sealed GD100, 100 accepted L-BFGS updates, alpha .015, 660 nm. Group 31 remains .329843 rad. Sensor L2 .441514%. Synthetic 100 RBC, 50 groups, noise seed 672906.

The exact archived benchmark ZIP and original B17 ZIP are supplied in the local migration package (not uploaded via the GitHub connector). Put the benchmark ZIP next to this README, original B17 under 데이터/B17. Then:

```bash
python benchmarks/single_reference/replay.py --output 싱글/결과/historical_replay --prepare-only
python benchmarks/single_reference/replay.py --output 싱글/결과/historical_replay_run
```

The adapter verifies both archive SHA-256 values, copies the archived source and sealed reference into a new output directory, changes only file paths, then runs the original spectral L-BFGS and finalizer. It does not repeat GD100. Original archives are not modified. The actual new full replay remains to be run; use separate output paths. This historical record is not a result of the new integrated solver.
