# 싱글 진행도 — 2026-11-16

**후속 진행 정책:** 일반사진 GD≤150/L-BFGS≤150, truth-free 수렴 기준으로 조기 종료 허용. 기존 Moon/Moon상하반전/Brick 3개 최종 통과. 일반사진 큐는26개이며 RBC8개는별도방법 큐로 분리했다. `ordinary_policy_batch.py`와 `rbc_queue.json` 참조. RBC의 과거 FullGD100+SpectralPeriodicL100 결과는 현재 native 검증과 구분한다. 물리조건과 피겨양식 변경 없음.

**최신 사용자 정정(2026-11-17):** [싱글스펙 대화 확인 및 최신 개정](LATEST_SINGLE_CHAT_REVISION.md)을 따른다. 사이트는 피겨만 참고하며 물리조건은 기존5713² native amplitude로 유지한다. 임의로 준비한3800² intensity 경로는 취소/비활성화했다. 완료2종은 실제배열을 유지하고 피겨만 재생성, 기존 Brick GD53 체크포인트에서 34종 native 배치를 재개한다. 아래 스페클 감사와 사이트 적용 기록의 '물리조건 변경/대량 보류'는 이 정정 이전 상태이며 현재 지시로 사용하지 않는다.

후속 사용자 지정: 피겨는 https://oms-hyu.github.io/Hyperspectral_HYU/ 실제 결과를 기준으로 생성한다. [실제 사이트 5패널 재배치 및 출력 기준](../결과/site_reference_figures_20261117/README.md). inferno 관측 스페클, viridis 위상, coolwarm ±π 오차. 다른 모델의 센서를 대체 사용하지 않는다. 새 intensity 복원 완료가 아니며 대량 배치 보류 상태는 유지한다.

**2026-11-17 스페클 감사:** [기존 그림/관측 모델 감사](../결과/speckle_audit_20261117/README.md). 대량 배치는 Brick GD53에서 일시 정지, 완료 2종 및 모든 상태 보존. 현재 native의 배경장만으로 사각형이 재현되며 amplitude 투영 후 제곱은 intensity 투영과 같지 않다. 기존 물리 intensity 스페클과 같은 조건으로 해석하지 않는다. 현재 상태는 queue.json 기준이다.

현재 활성 프로토콜: [native B17](UNIFIED_SPECTRUM_KO.md). [최신 정량 결과](../결과/최신_20261116/README.md) 및 [JSON](../결과/최신_20261116/native_results.json)이 수치의 단일 기준이다.

| 이미지 | GD150 circular RMSE | 최종 circular RMSE | 최종 일반 RMSE | Sensor L2 | 정밀 복원 |
|---|---:|---:|---:|---:|---|
| Mandrill | 0.113548 | 0.072388 | 0.096197 | 0.873755% | 60회 + Fourier 20회 |
| Cameraman 90° | 0.079079 | 0.058329 | 0.184236 | 0.877448% | 60회 |
| Barbara | 0.087111 | 0.060671 | 0.062638 | 0.876445% | 60회 |
| Boat | 0.060783 | 0.037773 | 0.066397 | 0.875357% | 60회 |

## 완료 / 미완료

| 실험 | 상태 |
|---|---|
| 5713² native amplitude 일반 이미지 Known 660 | 4/4 circular 및 sensor 기준 통과 |
| GD150 | Mandrill, Cameraman 90°, Barbara, Boat 완료 |
| L-BFGS60 α=0.0001 | 위 4종 완료; Mandrill 추가 Fourier20 완료 |
| 파장-only 비격자 / 잡음 강건성 | 273/273 / 224/224 정확 |

| Boat 신규 영 위상 복원 | GD150 + L60 완료, fresh 통과 |
| Tank 신규 영 위상 복원 | seed 672912: GD150 + 별도 배정밀도 L60 완료, circular 0.057277 rad, sensor L2 0.875580%; 원본 float32 L37 종료 보존 |

| 일반 이미지 440/550 nm phase inverse | NOT RUN |
| RBC full native phase inverse | 추가 검증 필요; triple/overlap 미해결 |
| Blind wavelength + phase 통합 / 실제 장비 | NOT RUN |

다음: Tank L-BFGS termination 원인 조사 및 재시도. L60 성공 전에는 440/550 nm 및 39종 큐를 시작하지 않는다. 이후 440/550/660 파장별 비교, RBC, Blind 통합. 이전 연구 결과를 성공 값으로 채우지 말 것. 기존 저장된 체크포인트는 외부 실험 패키지에서 계속 사용 가능; 전체 배열은 이 커밋에 미포함.

2026-11-17 갱신: [Tank 배정밀도 복구 실제 결과](../결과/tank_precision_20261117/README.md)는 GD150에서 새 curvature로 L60 완료 및 품질 기준 통과. 기존 float32 L37 결과도 기준 이내였으며 배정밀도에 따른 RMSE 개선은 작다. 정밀도 변형이므로 기존 공통 float32 L60 4종 목록과 분리한다. 440/550 nm 및 39종 후속 큐는 이번 작업에서 실행하지 않았다.

후속 사용자 지시(2026-11-17): 다른 사진들도 모두 진행. [나머지 34개 660 nm 순차 배치](../결과/all_photos_660_float64_20261117/README.md)를 실제 시작했다. 일반 입력 26개 후 RBC 관련 8개, 기존 완료 5개는 제외. `실험/all_photos_660_float64_20261117/queue.json`이 실시간 상태 기준이다. 정밀도 변형 GD150 + float64 L60 및 사후 평가를 입력별 실행하며 실패도 기록한 뒤 다음 입력으로 진행한다. 다른 파장 및 Blind는 실행하지 않는다. 전체 완료로 표시하지 않는다.

## 2026-11-17 Tank fresh 실행 상태

실제 입력 archive `B17_wavelength_only_39x3_20261115.zip`
(SHA-256 `28202bebb138151724dac9e10e710597ddaf8b0a29737e67fc49e64aeea0791a`)의
`inputs_manifest.json`과 `phase_inputs.npz/tank`를 사용했다. Tank phase SHA-256은
`095da2f3e7b996243a02707ef69bbe1af2d8223bdf328fcd965a95ae6843945f`이며,
광학 archive SHA-256은 `6264a21cb0433c691cd4511f1f9f428f8f92b160cc4388845b6c9b0586fad716`,
noise seed는 `672906`이다.

실행 명령은 `batch_runner.py --only tank --wavelength 660 --seed 672906`이며,
실제 GD150은 완료됐다. 센서 SHA-256은
`0c541a29f4481a46ae719d29fc9811b145cec7d131f449484421eb1ebc3b4d5c`,
GD150 센서 L2는 `0.008732069936622294`, GD150 phase SHA-256은
`a4d7215a81a77c134a065b4c175f66bf589423300e067472b68ac09cfdc64f7f`이다.
그러나 L-BFGS는 42 accepted에서 `task=[4,402]`로 종료해 L60 checkpoint가
없다. 따라서 Tank는 **FAILED/INCOMPLETE**, 공식 통과 및 HTML 게시 대상이
아니다. 기존 결과는 삭제하지 않았다. 기존 저장된 체크포인트는 외부 실험 패키지에서 계속 사용 가능; 전체 배열은 이 커밋에 미포함.


## 통과 결과 자동 게시 규칙 (2026-11-16 사용자 지시)

별도 업로드 요청 없이 실험 완료 시 자동 반영한다. 동일 native 5713² amplitude, zero start, GD150 및 L-BFGS60(α=1e-4) 완료와 체크포인트/센서/정답/위상 해시 확인 후 circular RMSE ≤0.15 rad AND sensor L2 ≤0.025이면 최신 최종 통과 목록에 추가한다. 일반 RMSE는 별도 표시한다. GD150만 통과한 경우 중간 성과로 기록하되 L60 완료로 표시하지 않는다. 실패/transfer/저차 기저/파장-only는 각 별도 범주에 남긴다. 해당 실험의 원본 평가 JSON·실제 복원 이미지·native_results.json·사이트 비교표/그래프·진행도/실험 로그를 동일 변경으로 갱신하고 main 반영 및 Pages 배포 상태를 검증한다. Pages 실패 시 배포 성공으로 보고하지 않는다. 멀티/연속 변경 금지.


## 2026-11-17 로컬 Codex 및 결과 사이트 운영

[Codex 실행 인수인계](SINGLE_CODEX_LOCAL.md)를 따른다. 사이트는 [기존 인터랙티브 양식](../../site/gs-metasurface-guide/index.html)을 기준으로 고정하며 39종 선택/Unknown 전환/상세 팝업/단계별 실제 비교 이미지를 유지한다. 통과 결과는 별도 요청 없이 수치·이미지·평가 JSON과 함께 갱신한다. Pages 실패와 main 반영은 구분하고, 오프라인 HTML을 우선 사용한다. 대용량 원본/체크포인트는 로컬·Drive에 저장하고 Git에는 요약과 필요한 이미지 및 코드만 둔다. 현재 새 배치 실행기 및 CUDA 가속 구현/검증은 미완료.
