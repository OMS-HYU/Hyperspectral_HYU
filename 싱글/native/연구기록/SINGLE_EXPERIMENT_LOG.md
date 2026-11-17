# 최신 싱글 실험 기록 — 2026-11-16

Git 게시 갱신: 이전에는 batch가 로컬 복사만 했고 커밋/푸시는 없었음을 확인하여 사용자에게 정정했다. `publish_verified_results.py --watch`를 구현하고 실제4건의 phase/sensor/truth해시, 유한평가수치를 검증한 JSON/5패널10파일을 전용clone에서 커밋/푸시했다. 원격main `68c9895a49b4a1235e655fd6530c0a5cd8b5ae6c` 일치 확인. 게시경로 `싱글_하이퍼스펙/결과/자동게시_20261117/`. 이후 완료결과30초간격 감지/증분푸시 프로세스 실행. 로컬raw/optimizer·연구체크아웃의미커밋변경은 보존했다. 기존사이트/Pages배포는 변경/성공확인하지 않았다.

최신 정정: 사용자 지시는 사이트의 피겨 형식만 적용하고 물리조건은 GPT 싱글스펙 최신 개정 유지. 실제 해당 대화의 최신3개 turn을 확인하여 native5713²/2857², 6.3mm/1.85µm, amplitude/기존배경, GD150+circularL60를 재확인했다. 3800² Moon GD2 smoke는 취소 실험으로 보존하고 관련 실행경로를 비활성화했다. 기존 native forward/GD/L-BFGS 파일 byte hash 및 Brick GD53 원자적 state 해시 일치 확인. 피겨만 inferno(log amplitude)/viridis/coolwarm±π로 수정하고 완료2종 재생성, 물리배열 불변 확인 후 native34종을 재개했다. 테스트4개 통과. 앞으로 GPT대화 참조 규칙을 루트/싱글 AGENTS.md에 기록했다.

2026-11-17 사용자 스페클 이의에 따른 [감사](../결과/speckle_audit_20261117/README.md): 과거 그림 59건 및 physical_speckles/생성 코드 대조. Moon forward 배경장 분해로 사각형 재현, 센서 해시 및 잡음 잔차 확인. 현재 진폭 제곱 패널을 기존 intensity 관측과 같게 설명한 내용을 정정. 배치 프로세스만 정지, Brick GD53 원자적 state/완료 2종 보존. 원본 모델·solver 변경이나 기존 수치의 intensity 성공 재분류는 하지 않았다.

| 실험 | 실제 완료 | 결과 |
|---|---|---|
| 파장-only 39×7 | 273건 | 273 exact; max error 0 nm |
| 파장-only robustness | 224건 | 224 exact; coarse 220/224 |
| Mandrill 신규 | GD150 + L60 | circular .113548 → .077902 |
| Mandrill matched circular Fourier | L60부터 restart 20회 | circular .072388, 일반 .096197, L2 .873755% |
| Cameraman 90° 신규 | GD150 + L60 | circular .058329, 일반 .184236, L2 .877448% |
| Barbara 신규 | GD150 + L60 | circular .060671, 일반 .062638, L2 .876445% |
| Boat 신규 | GD150 + L60 | circular .037773, 일반 .066397, L2 .875357%; GD150 .060783; 67 evaluations |

모든 위상 실험은 660 nm, native amplitude, 1% noise. 원본 수치·해시는 [최신 결과 JSON](../결과/최신_20261116/native_results.json)과 함께 보관. 5종 prior/edge/Fourier 비교는 실험적 비교군이고 기본 프로토콜은 α=1e-4. 정답 기반 최저 iteration 선택 금지. 과거 2026-11-12 이전 기록은 Git 이력 및 이전연구에 남아 있고 활성 기록에서 제외.

[Boat GD150](../결과/최신_20261116/boat_GD150.json), [Boat L60](../결과/최신_20261116/boat_L60.json): zero start, sensor/truth/phase/checkpoint 해시 확인. L60 기준 완료; 내부 계획 150회 중 60회에서 pause. 다음 Tank. 사용자 지시에 따라 이후 통과 결과는 별도 요청 없이 사이트와 함께 반영.



2026-11-17: 연구 계산 추가 없음. 기존 인터랙티브 사이트 양식 복원, 39개 선택 대상, 4종 최신 실제 결과 및 21개 단계별 복원 그림을 동기화. Google Drive 중요 자료 백업 확인. Tank는 아직 미실행. Codex 로컬 전환 지침은 [여기](SINGLE_CODEX_LOCAL.md).

## 2026-11-17 Tank fresh zero-start

| 항목 | 실제 값 |
|---|---|
| 입력 | `B17_wavelength_only_39x3_20261115.zip`, SHA-256 `28202bebb138151724dac9e10e710597ddaf8b0a29737e67fc49e64aeea0791a` |
| Tank phase | manifest `tank`, SHA-256 `095da2f3e7b996243a02707ef69bbe1af2d8223bdf328fcd965a95ae6843945f` |
| 광학 archive | SHA-256 `6264a21cb0433c691cd4511f1f9f428f8f92b160cc4388845b6c9b0586fad716` |
| 조건 | 660 nm, zero phase, native amplitude, 1% Gaussian noise, seed `672906` |
| 센서 | SHA-256 `0c541a29f4481a46ae719d29fc9811b145cec7d131f449484421eb1ebc3b4d5c` |
| GD150 | 완료; sensor L2 `0.008732069936622294`; phase SHA-256 `a4d7215a81a77c134a065b4c175f66bf589423300e067472b68ac09cfdc64f7f` |
| L-BFGS60 | 실패/미완료; 42 accepted, `task=[4,402]`, `phase_060.npy` 미생성 |
| 판정 | `FAILED/INCOMPLETE`; circular/general RMSE 및 HTML pass 게시 안 함 |

실행 로그와 checkpoint는 로컬 `실험/native_batch_20261117/tank`에 보존한다.

## 2026-11-17 Tank seed 672912 정밀도 복구 실험

- `실험/tank_fresh_672912_20261117/tank`: GD150 완료, 기본 float32 L-BFGS는 37 accepted에서 `task=[4,402]`로 종료. 기존 full-memory checkpoint를 그대로 재개해도 같은 종료를 재현했다. 원본 파일과 기본 solver/model은 보존한다.
- 이는 L-BFGS의 relative function reduction 수렴 종료이며 60 accepted 완료로 취급하지 않는다. float32 목적함수의 수치 정밀도가 원인인지 별도 실험한다.
- `코드/tank_precision_recovery.py`: 원본 보정 transfer coefficient 및 5713² 기하를 유지하며 전파/물체/카메라/목적함수를 float64/complex128로 계산한다. 동일 GD150부터 새 curvature로 60회를 요청하고 `recovery_float64/`에 별도 full-memory state를 저장한다. 원본 float32 L60와 혼합하지 않는다.
- 정답을 사용하지 않은 기울기 중앙차분 검증 상대 오차 `9.731614149565967e-7`. 독립적인 재개 일치 및 수렴 시 iteration 비조작 테스트 2개 통과.
- 사후 평가는 `코드/evaluate_tank_recovery.py`에서만 truth를 읽고 원본 float32 모델과 정밀도 복구 모델 모두의 센서 오차를 산출한다. 통과하더라도 기존 공통 float32 L60 완료 목록과 별도로 기록한다.
- 실제 완료: 배정밀도 60 accepted / 61 evaluations, circular 및 일반 RMSE `0.05727736225102632` rad, 원본 모델 sensor L2 `0.008755804879924063`, 배정밀도 모델 sensor L2 `0.008755923398815276`. 품질 통과. 기존 L37 circular `0.057282852286099493` rad와 거의 같은 복원 품질이다.
- [평가/비교 이미지/설정/해시](../결과/tank_precision_20261117/README.md)를 보존했다. 기존 공식 4종 float32 L60 결과 수는 유지한다. 원격 main과 HEAD 일치 확인; 이번 결과는 로컬에 저장했으며 원격 배포는 수행하지 않았다.

## 2026-11-17 전체 나머지 입력 실행

후속 피겨 지시: 원본/스페클/True phase/Estimated phase/오차의 5패널로 변경. Moon 상하반전 L60 및 평가 완료 후 단계 경계에서 정상 정지, 출력 코드만 변경하고 재개했다. 기존 3패널 보존, 완료 2종 5패널 생성 및 위상 해시 불변 확인. 신규 `render_five_panel.py`, 평가 JSON에 출력 provenance 추가. 광학 및 GD/L-BFGS 코드는 변경하지 않았다. 배치/재개 테스트 4개 통과. 실제 진행 상태는 queue.json을 따른다.

사용자가 다른 사진도 모두 진행하도록 지시했다. 검증된 Tank 정밀도 변형을 사용한 `all_photos_batch.py`를 별도 구현하여 [34개 실제 순차 배치](../결과/all_photos_660_float64_20261117/README.md)를 시작했다. 일반 26개 → RBC 8개, 660 nm만 실행하며 기존 완료 5개를 제외한다. 완료 건 제외/RBC 후순위와 자식 프로세스 실패 후 다음 입력 실행을 포함한 테스트 총 4개 통과. 입력 archive 및 각 위상 해시 검증 완료. Moon 센서 생성 완료와 GD 시작 확인. 최종 결과 수치/실패는 결과 index 및 실행 queue에 누적한다. 이번 전체 입력 지시는 기존 RBC 보류보다 우선하며, RBC를 측정/세포별 검증 성공으로 간주하지 않는다.

