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

## 추가 표준 벤치마크 (24개 신규 PNG)

### Set12 — 전체 12개

원본: [KAIR/testsets/set12](https://github.com/cszn/KAIR/tree/master/testsets/set12). 숫자로 저장된 원본 12개를 실제 이미지 내용을 확인한 후 의미 있는 이름으로 복사했다. PNG 원본 바이트는 변경하지 않았으며, 기존 cameraman/boat/pepper도 유지한다.

| scene | 저장 파일 | 원본 파일 |
|---|---|---|
| `camera_set12` | `camera_set12.png` | 01.png |
| `house` | `house.png` | 02.png |
| `pepper_set12` | `pepper_set12.png` | 03.png |
| `starfish` | `starfish.png` | 04.png |
| `monarch` | `monarch.png` | 05.png |
| `airplane` | `airplane.png` | 06.png |
| `parrot` | `parrot.png` | 07.png |
| `lena` | `lena.png` | 08.png |
| `barbara` | `barbara.png` | 09.png |
| `boat_set12` | `boat_set12.png` | 10.png |
| `man` | `man.png` | 11.png |
| `couple` | `couple.png` | 12.png |

### C4L 512×512 — 전체 13개 (12개 신규 + 기존 Pepper)

원본: [C4L 512x512](https://github.com/coderforlife/c4l-image-dataset/tree/master/512x512). 원본 바이트를 그대로 보존했다. Mandrill은 Baboon이라고도 부르는 표준 테스트 이미지다.

| scene | 저장 파일 | 원본 파일 |
|---|---|---|
| `aerial` | `aerial.png` | aerial.png |
| `bark` | `bark.png` | bark.png |
| `boat_c4l` | `boat_c4l.png` | boat.png |
| `brick_wall` | `brick_wall.png` | brick wall.png |
| `electron_micrograph` | `electron_micrograph.png` | em.png |
| `mandrill` | `mandrill.png` | mandrill.png |
| `motioncar` | `motioncar.png` | motioncar.png |
| `plane_c4l` | `plane_c4l.png` | plane.png |
| `random_texture` | `random_texture.png` | random.png |
| `sand` | `sand.png` | sand.png |
| `stream` | `stream.png` | stream.png |
| `tank_c4l` | `tank_c4l.png` | tank.png |
| `pepper` | `pepper.png` | pepper.png |

두 데이터셋의 원본 PNG는 `8bbfa05d907972e624f8ae617250cc385bb2ff07` 커밋에 추가되었다. 이름이 유사한 Boat, Peppers, Cameraman, Tank의 서로 다른 데이터셋 버전은 별도 scene으로 유지하므로 기존 실험의 입력 파일을 바꾸지 않는다.

- 라이선스 주의: Set12 및 C4L 저장소에 포함되었다는 사실만으로 개별 사진의 재배포 권리가 확인되는 것은 아니다. 특히 Lena/Lenna 등 역사적 사진의 저작권은 원저작자에게 귀속될 수 있다. 연구용 원본의 출처를 보존하고, 논문/공개 배포 시 개별 권리·출판사의 이미지 사용 기준을 확인할 것.
- PNG 원본은 RGB/회색조 변환과 512×512 리사이즈를 수행하는 공통 image_phase의 입력이다. 역복원에 정답 위상을 전달하지 않는다.
- 이 목록은 완성된 Set12와 C4L 512×512, 기존 입력 모음에 한정하며 전 세계 모든 영상처리 데이터셋의 목록은 아니다.
