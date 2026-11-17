# K=1 native B17 연구 규격 (2026-11-16)

- 전파 5713², meta aperture 2857², 원본 width map 2856², 물체 512², 센서 3000². dx=350 nm, z1=400 µm, z2=6.3 mm, camera pitch=1.85 µm.
- 원본 B17 LUT/width map의 SHA-256을 [결과 JSON](../결과/최신_20261116/native_results.json)에서 확인. 실제 실험 ZIP 해시도 기록. 원본 MAT의 해시는 반드시 일치시킨다.
- Phase-only object, 저장 위상 [0,2π), 물체 바깥 고정 배경: meta aperture 안쪽 phase 1 rad, 그 바깥 complex unity. ASM/SASM, amplitude bilinear antialias, 1% mean amplitude Gaussian + clip ≥0. 실제 센서 intensity와 동일하다고 주장하지 않는다.
- 영 위상, BB GD + Armijo, 64² 35회 → 128² 35회 → 256² 40회 → 512² 40회. α=0. 피라미드는 변수 해상도만 변경하며 매번 native 전파 격자를 사용.
- 기본 정밀 복원: circular prior α=1e-4, L-BFGS-B 60 accepted updates 기준 비교. 150회 전체 완료라고 표시하지 않는다. Mandrill의 추가 20회 Fourier matched prior는 별도 탐색 비교군.
- Known 파장 660 nm의 4종 검증 완료. 다른 파장, RBC full inverse 및 Blind 통합 미완료. 파장-only 273/273와 강건성 224/224는 위상 복원이 아니다.
- Circular RMSE = sqrt(mean(angle(exp(i*(estimated-truth)))²)), piston 정렬 없음. 일반 RMSE는 별도 산출. Sensor L2 = 상대 amplitude 잔차(최적 scalar gain 포함). 기준 circular≤.15 rad, sensor L2≤.025. 실제 정답/ROI/true wavelength는 inverse/후보 선택에 입력하지 않는다(known wavelength만 예외).
- 매 실험의 센서/정답/초기/최종 위상 SHA-256, accepted iterations, 노이즈 seed, 코드 버전 및 평가를 보존. 최신 결과 외에는 공식 성능으로 노출하지 않는다. 멀티/연속과 역사적 원본 소스는 변경하지 않는다.
