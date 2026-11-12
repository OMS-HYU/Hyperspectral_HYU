# Reproduction

1. Install requirements, the original extracted B17 calibration is included; run `python -m tests.test_contract --full-forward`.
2. RBC only: `python -m 싱글.run single --output 싱글/결과/RBC_660` (triple, Known 660 nm, seed 672906, full GD100+Spectral100). Then normal/variable/tilt/double and independent geometry seeds; blind triple 660 remains experimental. Preserve each output separately. For exact historical sensor/checkpoint use the archived benchmark, not the newly generated RBC model.
3. Evaluate only after inverse; each run records history, recovery/evaluation JSON, phase/sensor arrays, five-panel figure. No truth in optimization. Ground truth can be used for synthetic measurement generation and post-hoc metrics only.
4. Historical fixed-GD benchmark has distinct solver details; do not report its numbers as migrated performance. Benchmark archive is retained separately, see benchmarks/single_reference/README.md.
5. Multi/continuum are forward foundations, not inverse. Never claim wavelength recovery from a supplied ground-truth plan.

## 고정 최고 성능 센서 재현 (새 100+100 전체 실행)

기존 원본 실험 ZIP `B17_K1_RBC_0_2pi_Spectral_Periodic_LBFGS100_FULL_20261112.zip`을 로컬에 둔 뒤:

```bash
python -m 싱글.run reference --archive PATH_TO_ORIGINAL_ZIP --output 싱글/결과/RBC_fixed_660
```

원본 센서 SHA-256 일치가 필수이며, 새 GD100과 Spectral100을 처음부터 실행한다. 원래 physical phase는 센서 생성에만 사용하고 wrapped truth는 별도 파일에 격리한다. historical_comparison.json에서 기존 GD/최종 위상 해시 및 ROI/31번/통과 집단을 비교한다. `--generate-only`는 센서 재생성 검증만 수행한다. 이 신규 명령의 전체 역복원 성능은 아직 미검증. 기존 봉인 GD100 재사용 replay.py와는 별도.
