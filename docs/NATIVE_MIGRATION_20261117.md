# Public native workspace — 2026-11-17

All active single-spectrum code, inputs, checkpoints and results now belong to this public Hyperspectral_HYU project.

- Runtime: `.venv-native/Scripts/python.exe`
- Code: `싱글/native/코드/`
- Inputs: `싱글/native/데이터/`
- Active experiments: `싱글/native/실험/`
- Resume configuration: `싱글/native/실험/all_photos_660_float64_20261117/resume_command.json`
- Public artifacts: `공개결과/싱글/`; website source: `site/`
- Publication: `tools/public_results.py --watch`, exclusively to OMS-HYU/Hyperspectral_HYU.
- Public machine-readable index: https://oms-hyu.github.io/Hyperspectral_HYU/data/latest_single.json

859 migrated files (1,773,743,159 bytes) were verified before intentional path/document changes. Raw arrays, optimizer states, input ZIPs and the Python environment remain inside this project locally; compact results, figures and code are published. The previous location remains an inactive backup, excluded from Git. No old repository history or publisher clone was imported.

Frozen solver hashes and input hashes are checked before resumption. Native5713 amplitude physics are unchanged. Ordinary photographs use GD150/L-BFGS150 maximum with truth-free convergence checks. Physical RBC cases retain a separate FullGD100 + Spectral Periodic L-BFGS100 route, pending native validation and missing ROI/layer/group metadata. No RBC success is claimed.

New figures contain original image, observed amplitude speckle, true phase, estimated phase and wrapped error. Four earlier baseline previews retain explicit historical-stage labels; they are not represented as newly rendered final five-panel figures.
