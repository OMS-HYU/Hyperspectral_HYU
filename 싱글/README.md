# 싱글 K=1 — 적혈구 연구 및 일반 이미지 복원

**[🌐 싱글 명령어 생성기 바로 열기](https://htmlpreview.github.io/?https://github.com/OMS-HYU/Hyperspectral_HYU/blob/3f31c78e2d995d9ff30aa81ed65a85df6a846d5c/%EC%8B%B1%EA%B8%80/%EC%BD%94%EB%93%9C%EC%83%9D%EC%84%B1%EA%B8%B0/index.html)** · [오프라인 실행 및 HTML 파일](코드생성기/README.md)


**[상세 한글 실행 가이드북](가이드북.md)**

**기본 알고리즘: Full-resolution GD100 + Spectral Periodic L-BFGS100.** 적혈구에서는 피라미드 사용 안 함. 일반 사진은 Pyramid64 GD150 + L-BFGS-B 최대150을 기본 후보로 선택. `--method auto` (기본), `spectral`, `pyramid`, `periodic`, `compare` 지원. 새 포트 전체 반복 검증은 아직 미완료.

```bash
python -m 싱글.run single --output 싱글/결과/RBC_660
# defaults: triple / Known 660 nm / spectral / noise seed 672906 / geometry seed 20261127
python -m 싱글.run single --scene normal --true-wavelength 550 --mode known --wavelength 550 --output 싱글/결과/normal550
```

적혈구 모델: normal, variable, tilt, double, triple. --geometry-seed와 --seed는 독립. --method periodic은 비교용, --mode blind는 미지 파장 실험용이며 true wavelength를 inverse에 전달하지 않는다.

일반 이미지 예시: `python -m 싱글.run single --scene camera --method pyramid --output 싱글/결과/camera660`.

코드/pyramid.py: 일반 이미지용 기존 Pyramid64 재사용, 광학 입력과 외부 위상은 wrap.

코드/solver.py: 실제 512² GD100 및 Spectral Periodic L-BFGS100, 0≤φ<2π, α=.015, 3800² B17, 3000² 센서. 생성/역복원/평가 분리. 기준 ROI .077085576 rad, 49/50은 기존 고정 센서와 GD100에서 검증된 역사적 결과이고 새 생성 장면의 성능으로 주장하지 않는다.

정확한 역사적 실험 재현: [benchmarks/single_reference](../benchmarks/single_reference/README.md). 연구 결정 및 전체 피라미드 비교: [연구기록/개발방향.md](연구기록/개발방향.md).

## 고정 최고 성능 센서 재현 (새 100+100 전체 실행)

기존 원본 실험 ZIP `B17_K1_RBC_0_2pi_Spectral_Periodic_LBFGS100_FULL_20261112.zip`을 로컬에 둔 뒤:

```bash
python -m 싱글.run reference --archive PATH_TO_ORIGINAL_ZIP --output 싱글/결과/RBC_fixed_660
```

원본 센서 SHA-256 일치가 필수이며, 새 GD100과 Spectral100을 처음부터 실행한다. 원래 physical phase는 센서 생성에만 사용하고 wrapped truth는 별도 파일에 격리한다. historical_comparison.json에서 기존 GD/최종 위상 해시 및 ROI/31번/통과 집단을 비교한다. `--generate-only`는 센서 재생성 검증만 수행한다. 이 신규 명령의 전체 역복원 성능은 아직 미검증. 기존 봉인 GD100 재사용 replay.py와는 별도.
