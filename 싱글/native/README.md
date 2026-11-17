<!-- public-native-workspace -->
현재 작업 위치: `싱글/native/` · 실행 환경: `.venv-native/Scripts/python.exe` · 공개 결과: https://oms-hyu.github.io/Hyperspectral_HYU/ · 이관 기록: `docs/NATIVE_MIGRATION_20261117.md`. 아래 이전 실행 경로는 과거 기록입니다.

# 싱글 하이퍼스펙 — 최신 B17 K=1

최종 갱신: 2026-11-16. **5713² native amplitude**만 현재 기준으로 사용한다. [최신 결과 및 원본 수치](결과/최신_20261116/README.md) · [연구 규격](연구기록/UNIFIED_SPECTRUM_KO.md) · [진행도](연구기록/SINGLE_PROGRESS.md) · [인수인계](연구기록/SINGLE_HANDOVER.md).

| 이미지 | GD150 circular RMSE | 최종 circular RMSE | 최종 일반 RMSE | Sensor L2 | 정밀 복원 |
|---|---:|---:|---:|---:|---|
| Mandrill | 0.113548 | 0.072388 | 0.096197 | 0.873755% | 60회 + Fourier 20회 |
| Cameraman 90° | 0.079079 | 0.058329 | 0.184236 | 0.877448% | 60회 |
| Barbara | 0.087111 | 0.060671 | 0.062638 | 0.876445% | 60회 |
| Boat | 0.060783 | 0.037773 | 0.066397 | 0.875357% | 60회 |

660 nm Known, 영 위상 → GD150 → 약한 circular prior. Circular RMSE 기준 4/4 통과. 파장-only 273/273 및 224/224 정확; 위상복원과 분리. 아직 Tank fresh, 다른 파장 phase, RBC 및 Blind 통합 미완료.

기존 unified_simulator.py와 과거 single_b17_v1은 역사적 3800² 재현용이며 최신 수치의 실행기가 아니다. 최신 native 알고리즘/실행기 및 재현법은 연구기록을 참조한다. 전체 센서·optimizer 배열은 기존 별도 실험 ZIP에 보존하고, 이 Git 결과 폴더에는 실제 평가 JSON·해시·요약을 둔다. 멀티·연속 독립.

## 최신 실행기 재현

Python 3.13 권장 (검증 3.13.5). SciPy private L-BFGS-B state API 때문에 1.17.0으로 고정한다. 정확한 재현에는 실험 원본 512² phase truth NPY와 seed 및 B17 ZIP을 사용한다.

```powershell
python -m pip install -r "싱글_하이퍼스펙/requirements-native.txt"
python "싱글_하이퍼스펙/run.py" --help
python "싱글_하이퍼스펙/run.py" generate --scene barbara --truth /path/to/private_truth.npy --seed 672910 --wavelength 660 --archive /path/to/05_NN_metacam-main.zip --run "싱글_하이퍼스펙/결과/barbara_native"
python "싱글_하이퍼스펙/run.py" gd --archive /path/to/05_NN_metacam-main.zip --run "싱글_하이퍼스펙/결과/barbara_native" --steps 150
python "싱글_하이퍼스펙/run.py" lbfgs --archive /path/to/05_NN_metacam-main.zip --run "싱글_하이퍼스펙/결과/barbara_native" --alpha 0.0001 --until 60
python "싱글_하이퍼스펙/run.py" evaluate --run "싱글_하이퍼스펙/결과/barbara_native" --truth /path/to/private_truth.npy --phase "싱글_하이퍼스펙/결과/barbara_native/long_alpha_0.0001/phase_060.npy" --output "싱글_하이퍼스펙/결과/barbara_native/evaluation_L60.json"
```

GD --steps는 추가 accepted 횟수, L-BFGS --until은 누적 목표. 중단 후 상태를 읽어 재개한다. 추가 Fourier는 동일 L60 이후 fourier --arm matched --steps 20. 정답은 합성 센서 생성과 사후 평가에서만 읽고 inverse는 관측만 사용한다. 현재 CLI는 Known 복원 전용이며 Blind 통합 검증과 다른 파장 일반화는 미완료다. 원본 전체 배열과 체크포인트는 별도 실험 패키지에 보존한다. 구형 실행기는 --legacy로만 호출한다.

CLI 독립 확인: 실제 Barbara 원본 sensor SHA-256 및 GD1 residual 정확히 일치. 전체 210회 재실행과는 구별한다.



## VS Code / Codex 로컬 실행

[최신 실행 인수인계 및 Drive 데이터 경로](연구기록/SINGLE_CODEX_LOCAL.md) · [Codex AGENTS 지침](AGENTS.md) · [기존 디자인 대시보드](../site/gs-metasurface-guide/index.html). 새 연구는 Tank Known 660 nm부터 시작한다. 완료 4종을 다시 실행하지 않으며, GPU/자동 배치는 아직 검증되지 않았다. 큰 데이터는 로컬과 Drive를 이용한다.

## 자동 배치 실행

`코드/batch_runner.py`는 원본 입력 archive의 `inputs_manifest.json`과
`phase_inputs.npz`를 검증한 뒤, 각 케이스를 순차적으로
`generate → GD150 → L-BFGS60 → evaluate` 순서로 실행한다. 각 실행의
`batch_status.json`과 `batch.log`는 결과 폴더에 남으며, 중단 시 기존
`manifest.json`과 native checkpoint를 재사용한다. 입력 manifest나 phase
배열이 없으면 큐를 추측하지 않고 blocked 상태로 종료한다.

```powershell
& ".venv\Scripts\python.exe" "싱글_하이퍼스펙\코드\batch_runner.py" `
  --input-archive "PATH\B17_wavelength_only_39x3_20261115.zip" `
  --optics-archive "PATH\05_NN_metacam-main.zip" `
  --output "싱글_하이퍼스펙\실험\native_batch_YYYYMMDD" `
  --only tank
```

`--only`를 생략하면 manifest에 실제로 존재하는 39개 케이스를 manifest 순서로
처리한다. `--inspect-only`는 배열을 생성하지 않고 queue를 검증·기록한다.

