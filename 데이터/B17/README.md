# Original B17 calibration

The original mapped width and wavelength LUT are now tracked under:
`05_NN_metacam-main/assets/data/Fab/B17/`. The project reads these files directly; no extra upload or decompression is needed.

Required SHA-256:
- LUT `bayesLUT_MSE_v6.3_nonoverlap.mat`: 02c551f2e24fb01d91416ea21cd17d24ae1c5e6fab0d630bff8e1fb2b4feaeaa
- Width `0.6NA_random_70_1_300_1mm_mapped_width.mat`: b7d799c1e3a5913e7e9034fd2efc31dc2dab98822058ef08e2e57470ac2d68f9

The loader also supports the original ZIP at `데이터/B17/05_NN_metacam-main.zip`, or an explicit B17_SOURCE path to a matching directory/ZIP (legacy B17_ZIP accepted). Both MAT hashes must match regardless of packaging. The original historical ZIP outer SHA-256 was 6264a21cb0433c691cd4511f1f9f428f8f92b160cc4388845b6c9b0586fad716; extracted inputs have no outer archive hash and record a physical source fingerprint instead. Historical exact replay still requires its original ZIP separately.

No optical parameters changed. The uploaded original research software directory is retained, but the new solver imports only 공통/ modules. No new license is asserted.
