# VS Code / Codex 로컬 실행 인수인계 — 2026-11-17

최신 사용자 지시: 작업 전 GPT '싱글스펙' 대화의 최신 개정을 확인한다. 대화 ID와 확인 절차는 [최신 개정](LATEST_SINGLE_CHAT_REVISION.md), 루트/싱글 AGENTS.md 참조. 사이트는 **피겨 형식만** 참고하고 native5713² amplitude 물리조건은 유지한다. 취소된3800² 사이트조건 배치를 실행하지 않는다. 현재 34종 배치는 기존 Brick GD53에서 재개한다.

## 현재 상태

활성 규격 및 수치는 [진행도](SINGLE_PROGRESS.md), [프로토콜](UNIFIED_SPECTRUM_KO.md), [최신 JSON](../결과/최신_20261116/native_results.json)이 기준. 660 nm Known Mandrill/Cameraman 90°/Barbara/Boat 4종 완료. 나머지 35종 full inverse 미검증. 파장-only 273/273 및 robustness 224/224는 full phase inverse가 아니다. Tank가 다음 신규 실행이다. 기존 native CLI는 CPU Known 전용이며 전체 배치/CUDA/Blind는 미구현 또는 미검증.

## 데이터 및 보존

[Google Drive Capstone_Optics](https://drive.google.com/drive/folders/1SpH4D-ZM3AjRXWoX4ziBHUKNJLn5dg0P). 접근 권한은 기존 Drive 설정에 따르며 공개화하지 않는다.

- 01_원본데이터: 05_NN_metacam-main.zip (16,222,487 bytes), B17_wavelength_only_39x3_20261115.zip (26,454,479 bytes; wave/phase_inputs.npz 및 wave/inputs_manifest.json 확인 후 실제 39종 512² truth 추출). LUT/widthmap 해시를 native_results.json과 대조. 원본 ZIP을 다시 다운로드하거나 저장소 데이터/에서 동일한 원본 사용 가능.
- 02_싱글: B17_Single_Native_Fresh_GD150_20261116.zip, B17_Mandrill_Prior_Matched_L20.zip, B17_Cameraman_Native_GD150_20261116.zip, B17_Barbara_Native_GD35_20261116.zip, Mandrill_L60_summary.csv, Boat GD150/L60 및 Cameraman/Barbara 평가 JSON. Barbara GD35는 최종 체크포인트가 아님. Boat/Barbara L60 전체 raw optimizer 및 Cameraman 최종 phase는 백업 완전성 미확인; 평가 JSON/그림과 raw state를 혼동하지 않는다.
- 05_연구기록: 파장-only/robust JSON. 06_최신_HTML: B17_Single_Spectral_Results_Latest.html (10,415,426 bytes; 원본 standalone, 선택창/상세 팝업). Git site와 동일한 데이터/스타일; Git 버전은 큰 WebP를 상대 경로로 분리.
- 대용량 파일은 PC 또는 Drive 동기화 폴더에 보존. Drive 업로드 실패한 파일은 삭제 금지. 새 센서/optimizer/phase의 정확한 재개 자료와 해시를 별도 로컬 저장. 원본 데이터/코드/최신 체크포인트 무단 삭제 금지.

## Windows 설정 및 실행

저장소 루트에서 `git status`, `git pull --ff-only origin main`. 로컬 변경 충돌 시 중단하고 보존. `code capstone_optics.code-workspace`로 열기. Python 3.13.5 검증 환경 및 SciPy 1.17.0 고정. 기존 venv가 Python 3.11이면 별도 .venv-native로 분리한다.

```powershell
py -3.13 -m venv .venv-native
$py = '.\.venv-native\Scripts\python.exe'
& $py -m pip install -r '싱글_하이퍼스펙/requirements-native.txt'
& $py '싱글_하이퍼스펙/run.py' --help
```

Codex에게 CPU/RAM/GPU/VRAM을 확인시키고 실제 입력 경로를 찾도록 한다. Tank truth는 위 manifest에서 정확한 key/shape/범위 확인 후 별도 .npy로 추출하고 해시를 기록한다. 아래 경로는 예시이며 로컬에 실제 파일을 준비한다. Tank 신규 seed 672912를 제안하며 이전 실험 seed 충돌 여부 확인. 새 run 경로가 이미 있으면 generate 재실행 금지.

```powershell
$run = '싱글_하이퍼스펙/결과/tank_native_660'
$archive = '싱글_하이퍼스펙/데이터/05_NN_metacam-main.zip'
$truth = 'local_data/tank_truth.npy'
& $py '싱글_하이퍼스펙/run.py' generate --scene tank --truth $truth --seed 672912 --wavelength 660 --archive $archive --run $run
& $py '싱글_하이퍼스펙/run.py' gd --archive $archive --run $run --steps 150
& $py '싱글_하이퍼스펙/run.py' lbfgs --archive $archive --run $run --alpha 0.0001 --until 60
& $py '싱글_하이퍼스펙/run.py' evaluate --run $run --truth "$run/private_truth.npy" --phase "$run/gd_150.npy" --output "$run/evaluation_GD150.json"
& $py '싱글_하이퍼스펙/run.py' evaluate --run $run --truth "$run/private_truth.npy" --phase "$run/long_alpha_0.0001/phase_060.npy" --output "$run/evaluation_L60.json"
```

각 단계의 실제 accepted count 및 상태를 확인하고 실패 시 다음 단계로 진행하지 않는다. evaluate는 phase RMSE와 해시를 출력하며 sensor L2는 동일 checkpoint의 progress/history에서 읽는다. GD150 및 L60 모두 정확한 sensor/phase hash와 연결한다. 같은 폴더 재개는 --steps 추가 횟수, --until 누적 목표 사용. L60 내부 target=150이므로 complete=false일 수 있으나 accepted=60을 기준으로 판단. 실험 중 다른 solver 버전/물리 조건 변경 금지.

## 자동 실행 요구 (Codex 구현 항목)

1. 실제 39종 manifest 기반 큐, 완료된 동일 protocol scene/wavelength 건너뛰기. Tank fresh 660 → 나머지 일반 이미지 660 → 440/550 → RBC full inverse → Blind 통합. 파장-only 성공을 full inverse로 간주하지 말 것. 순서 변경 시 기록.
2. CPU smoke 확인 후 단일 프로세스로 연속 실행/재개. GPU는 별도 브랜치 또는 별도 비교 구현으로 forward/adjoint/gradient/목적함수 및 수치 일치 검증 후 채택. 무조건 GPU가 빠르다고 주장하지 않는다.
3. 단계별 체크포인트/로그/해시/실패 이유 저장. circular RMSE <=0.15 rad, relative sensor L2 <=0.025 및 GD150/L60 검증 후에만 최종 통과. 일반 RMSE 별도. 실패는 실패, 미실행은 미검증.
4. 통과 결과마다 native_results.json 및 실제 평가 JSON, 진행도, 로그, 사이트의 matrix/dataset/gallery를 동기화. 원래 CSS 및 선택/클릭 동작 변경 금지. 원본 위상/복원/오차 그림은 실제 배열에서 생성. 4종 기존 수치 보존. Cameraman L60 수치는 확보, 그림은 GD150만; Mandrill 추가 Fourier20은 별도 비교.
5. Git site `site/gs-metasurface-guide/index.html`을 로컬 브라우저에서 열어 39종 선택, Known/Unknown, 상세 팝업, 확대, JSON 저장 검증. `python site/gs-metasurface-guide/export_standalone.py --output <경로>`로 독립 HTML 내보내기. main 반영과 Pages 배포 성공은 별도. 대용량 raw arrays Git 자동 커밋 금지.

Codex 시작 문장: "싱글_하이퍼스펙/AGENTS.md와 연구기록/SINGLE_CODEX_LOCAL.md를 먼저 읽고 현재 main에서 Tank 신규 복원부터 실제 실행해. 로컬 환경과 데이터 해시를 검증하고 중단 재개 가능한 순차 배치 실행기를 만들어. 기존 인터랙티브 HTML 디자인을 건드리지 말고 통과 결과만 자동 갱신해. 멀티·연속은 수정 금지."
