# 적혈구 실험

- R0: 고정 660 nm 100 RBC / 50 groups, 원래 센서 SHA-256 및 GD100 봉인본의 역사적 49/50 재현.
- R1: normal, variable, tilt, double, triple, Known 660 nm. 동일 센서에서 spectral / periodic 비교.
- R2: 다른 geometry/noise seed; 440, 550, 660 nm 및 중간 파장.
- R3: Blind 후보 파장 탐색 (experimental, Known 검증 후).
- R4: group31 실패 원인 및 개선. ground truth는 생성/평가에서만 사용.

전체 광학 격자와 GD100+L-BFGS100 고정. 0~2π canonical. 피라미드는 기본 경로에서 제외. 성능, 계산량, hash, 실패 원인 기록.

R0 실행: `python -m 싱글.run reference --archive PATH_TO_ORIGINAL_ZIP --output 싱글/결과/RBC_fixed_660`. 전체 재검증은 미실행.


## 신규 알고리즘 비교

[싱글 가이드북](../가이드북.md)을 기준으로 동일 센서의 Spectral/Pyramid를 비교한다.

```bash
python -m 싱글.run single --scene triple --method compare --output 싱글/결과/compare_rbc660
python -m 싱글.run single --scene camera --method compare --output 싱글/결과/compare_camera660
```

기존 benchmark와 신규 실행 결과는 분리한다.
