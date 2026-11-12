# 연구 진행도

- 단파장: 2026-11-13 사용자 결정으로 적혈구 단독 연구. Full GD100 + Spectral Periodic L-BFGS100 기본. 660 nm Known 기준 49/50, ROI .077085576 rad. Pyramid64 전체 비교(250/300회) ROI .083631361/.084559009로 악화, 기본에서 제외. 현재 통합 solver 전체 역복원 재현은 미검증. Blind는 실험 옵션으로 유지하고 Known 660 nm를 우선 검증.
- 멀티: K=2~3, 독립 위상, 파장 식별과 강도 분리. 현재 forward/plan만. 기존 V71 연구는 원래 저장소에 동결; 이관된 새 inverse라고 주장하지 않는다.
- 연속: quadrature 및 intensity forward, 이후 연속 spectral density 및 identifiability 검증.
- 현재 미해결: RBC group31, prior bias; spatial-spectral은 ROI .077579로 악화, 채택 안 함. 타 파장 전처리기 일반화 및 실제 측정 미검증.
- 새 실험은 날짜/commit/seed/광학 조건/accepted step/센서 및 결과 hash/실제 metric/실패 원인 기록. 기존 150-step, reduced/pyramid 프로토콜과 혼합 금지.
