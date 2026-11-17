# 싱글 인수인계 — native 5713² (2026-11-16)

1. [활성 물리·알고리즘 규격](UNIFIED_SPECTRUM_KO.md)과 [진행도](SINGLE_PROGRESS.md)를 먼저 읽는다.
2. 최신 실제 수치는 [결과 디렉터리](../결과/최신_20261116/README.md), [수치 JSON](../결과/최신_20261116/native_results.json), [실험 로그](SINGLE_EXPERIMENT_LOG.md)에 있다. 최신 합격 수치는 660 nm Known의 4장면뿐이다.
3. 기존 3800²/5700² 실행기와 결과는 역사적 재현 전용이다. 기존 source를 최신 결과의 실행기로 오인하지 말 것. 현재 사용한 native 원본 실험 패키지: B17_Single_Native_Fresh_GD150_20261116.zip, B17_Mandrill_L60_Checkpoint.zip, B17_Mandrill_Prior_Matched_L20.zip, B17_Cameraman_Native_GD150_20261116.zip, Barbara의 GD/optimizer 체크포인트 및 평가 JSON. 실제 센서·위상·optimizer 배열은 별도 보관.
4. 660 nm 일반 이미지 복원 시 전체 광학 해상도 5713²를 유지하고, 초기 영 위상에서 GD150 → α=1e-4 L-BFGS60를 기준으로 한다. 새로운 실행/체크포인트는 서로 분리. 정답을 fitting/stop/selection에 사용 금지.
5. 파장-only 결과는 별도 검증이다. 273건과 224건의 최종 수치를 실제 기록에서 확인. 위상 통합 성공으로 분류 금지.
6. 다음 순서: Tank → 440/550/660 일반화 → RBC full native phase → Blind end-to-end. 모든 실패/미실행 명시. 멀티·연속은 수정 금지.


## 2026-11-17 로컬 Codex 및 결과 사이트 운영

[Codex 실행 인수인계](SINGLE_CODEX_LOCAL.md)를 따른다. 사이트는 [기존 인터랙티브 양식](../../site/gs-metasurface-guide/index.html)을 기준으로 고정하며 39종 선택/Unknown 전환/상세 팝업/단계별 실제 비교 이미지를 유지한다. 통과 결과는 별도 요청 없이 수치·이미지·평가 JSON과 함께 갱신한다. Pages 실패와 main 반영은 구분하고, 오프라인 HTML을 우선 사용한다. 대용량 원본/체크포인트는 로컬·Drive에 저장하고 Git에는 요약과 필요한 이미지 및 코드만 둔다. 현재 새 배치 실행기 및 CUDA 가속 구현/검증은 미완료.
