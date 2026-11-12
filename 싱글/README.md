# 싱글 K=1

`python -m 싱글.run single --scene tank --true-wavelength 660 --mode known --wavelength 660 --output 싱글/결과/tank660`

실제 solver는 코드/solver.py. Known/experimental Blind, GD100+L-BFGS100. Blind에서는 --wavelength 금지. --geometry-seed와 --seed는 독립적으로 제어. 통합 inverse의 전체 재현은 미검증.
