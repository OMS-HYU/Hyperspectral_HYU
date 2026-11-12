# 입력 이미지 및 출처

싱글 일반 이미지 기본값은 이 폴더의 원본 파일을 읽는다. 모든 원본은 로컬에서 이용 가능하다. HTML 생성기는 이미지 표시용 내장 데이터도 함께 포함한다.

| scene | 파일 | 출처/설명 |
|---|---|---|
| camera | camera.png | scikit-image v0.25.2 원본 |
| camera_rot90 | camera.png | 같은 원본을 코드에서 90° 반시계 회전, 미리보기는 미리보기/camera_rot90.jpg |
| chelsea | chelsea.png | scikit-image v0.25.2 |
| checkerboard | checkerboard.png | scikit-image v0.25.2 chessboard_GRAY.png |
| moon | moon.png | scikit-image v0.25.2 |
| brick | brick.png | scikit-image v0.25.2 |
| coins | coins.png | scikit-image v0.25.2 |
| astronaut | astronaut.png | scikit-image v0.25.2 |
| boat | boat.jpg | 기존 저장소 원본 |
| tank | tank.png | 기존 USC-SIPI 7.1.03 무손실 PNG, tank_source.md 참고 |
| pepper | pepper.png | C4L 512×512 grayscale Peppers, USC-SIPI 4.2.07 변환본 |
| custom | 사용자 파일 | CLI --image-file로 지정, HTML에서 직접 파일 미리보기 가능 |

scikit-image 원본: https://github.com/scikit-image/scikit-image/tree/v0.25.2/skimage/data (각 이미지의 별도 출처와 사용 조건은 https://scikit-image.org/docs/stable/api/skimage.data.html 참고).
Peppers: https://github.com/coderforlife/c4l-image-dataset/blob/master/512x512/pepper.png (grayscale 연구용 이미지; USC-SIPI 원본 저작권 상태 미확인: https://sipi.usc.edu/database/copyright.php).
기존 cameraman.png는 이전 소스 저장소의 별도 원본으로 그대로 보존한다. 표준 camera와 동일한 파일이라고 가정하지 않는다. Boat/Tank와 Peppers 등은 연구용 데이터이며 이 README가 별도 재배포 권리를 부여하지 않는다.

## 적혈구 기본 합성 위상맵

RBC/에는 normal, variable, tilt, double, triple 5종의 660 nm / geometry seed 20261127 모델 위상맵이 들어 있다. 실제 `공통/object_models.py`의 rbc_phase를 사용해 만든 512² phase의 128² 표시용 PNG(16단계 twilight palette)다.
이 PNG를 역복원기에 전달하지 않는다. 실제 시뮬레이션은 입력 파장 및 geometry seed에 맞춰 512² phase를 다시 생성한다. 표시용 PNG는 측정 데이터나 과거 봉인 benchmark가 아니다.

원본과 새 합성 위상의 구분을 유지하며, 정답 데이터를 역복원에 사용하지 않는다.
