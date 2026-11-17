"""Validate migrated solver identity and configure this public workspace."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / '싱글/native'
RUN = TRACK / '실험/all_photos_660_float64_20261117'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    identity = json.loads((RUN/'ordinary_policy_identity.json').read_text(encoding='utf-8'))
    for name, expected in identity['code'].items():
        assert sha(TRACK/'코드'/name) == expected, name
    source = TRACK/'실험/native_batch_20261117/_source/input/05_NN_metacam-main.zip'
    target = TRACK/'데이터/05_NN_metacam-main.zip'
    assert sha(source) == identity['optics_sha256']
    if target.exists() and sha(target) != sha(source):
        backup = target.with_name('05_NN_metacam-previous-variant.zip')
        if not backup.exists(): shutil.copy2(target, backup)
    shutil.copy2(source, target)
    archive = TRACK/'데이터/B17_wavelength_only_39x3_20261115.zip'
    assert sha(archive) == identity['input_sha256']
    config = dict(python=str(ROOT/'.venv-native/Scripts/python.exe'), args=[
        str(TRACK/'코드/ordinary_policy_batch.py'), '--input-archive', str(archive),
        '--optics-archive', str(target), '--output', str(RUN)])
    (RUN/'resume_command.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'docs/NATIVE_MIGRATION_20261117.md').write_text('''# Public native workspace — 2026-11-17

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
''',encoding='utf-8')
    for p in [ROOT/'README.md', TRACK/'README.md']:
        text=p.read_text(encoding='utf-8')
        marker='<!-- public-native-workspace -->'
        if marker not in text:
            p.write_text(marker+'\n현재 작업 위치: `싱글/native/` · 실행 환경: `.venv-native/Scripts/python.exe` · 공개 결과: https://oms-hyu.github.io/Hyperspectral_HYU/ · 이관 기록: `docs/NATIVE_MIGRATION_20261117.md`. 아래 이전 실행 경로는 과거 기록입니다.\n\n'+text,encoding='utf-8')
    print('Validated frozen solver and inputs; configured public workspace.')

if __name__ == '__main__': main()
