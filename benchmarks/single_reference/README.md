# Historical fixed 660 nm RBC benchmark

| Method | Full RMSE rad | ROI RMSE rad | Groups pass |
|---|---:|---:|---:|
| Original bounded | .053314 | .174806 | 24/50 |
| Periodic | .030815 | .084598 | 48/50 |
| Spectral Periodic | .029155 | .077086 | 49/50 |
| Spatial-Spectral (not adopted) | .029246 | .077579 | 49/50 |

Historical full grid 3800/1429/512/3000, fixed same GD100, 100 accepted L-BFGS, alpha .015, 660 nm. Group 31 remains .329843 rad. Sensor L2 .441514%. Same original B17, synthetic 100 RBC, 50 groups, noise seed 672906. Historical exact ZIP and assets supplied via local migration bundle; this text does not constitute a rerun. New solver must be validated independently. See docs/REPRODUCTION.md.
