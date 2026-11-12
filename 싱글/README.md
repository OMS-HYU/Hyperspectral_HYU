# 싱글 K=1 — 적혈구 전용 연구

**기본 알고리즘: Full-resolution GD100 + Spectral Periodic L-BFGS100.** 피라미드 사용 안 함. 일반 사진은 멀티/연속 및 역사적 실험용으로 보존.

```bash
python -m 싱글.run single --output 싱글/결과/RBC_660
# defaults: triple / Known 660 nm / spectral / noise seed 672906 / geometry seed 20261127
python -m 싱글.run single --scene normal --true-wavelength 550 --mode known --wavelength 550 --output 싱글/결과/normal550
```

적혈구 모델: normal, variable, tilt, double, triple. --geometry-seed와 --seed는 독립. --method periodic은 비교용, --mode blind는 미지 파장 실험용이며 true wavelength를 inverse에 전달하지 않는다.

코드/solver.py: 실제 512² GD100 및 Spectral Periodic L-BFGS100, 0≤φ<2π, α=.015, 3800² B17, 3000² 센서. 생성/역복원/평가 분리. 기준 ROI .077085576 rad, 49/50은 기존 고정 센서와 GD100에서 검증된 역사적 결과이고 새 생성 장면의 성능으로 주장하지 않는다.

정확한 역사적 실험 재현: [benchmarks/single_reference](../benchmarks/single_reference/README.md). 연구 결정 및 전체 피라미드 비교: [연구기록/개발방향.md](연구기록/개발방향.md).
