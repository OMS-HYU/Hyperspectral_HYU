# 싱글 스펙트럴 명령어 생성기

**[🌐 브라우저에서 바로 열기](https://htmlpreview.github.io/?https://github.com/OMS-HYU/Hyperspectral_HYU/blob/832a3f2ecf7a0faa420e40cd5d11ca78c4a2e805/%EC%8B%B1%EA%B8%80/%EC%BD%94%EB%93%9C%EC%83%9D%EC%84%B1%EA%B8%B0/index.html)**

위 링크는 공개 HTML의 실행 미리보기(외부 HTMLPreview 서비스)다. GitHub 자체의 HTML 파일 보기에서는 JavaScript가 실행되지 않는다. 서비스가 차단되거나 미리보기가 작동하지 않으면 아래 오프라인 방법을 사용한다.

## Windows 오프라인 실행 (권장)
- 탐색기에서 [실행.cmd](실행.cmd)를 더블클릭하거나 [index.html](index.html)을 브라우저로 연다.
- VS Code 터미널에서 저장소 루트 기준:
```powershell
Start-Process ".\싱글\코드생성기\index.html"
```
- 이미지도 HTML에 포함되어 있어 인터넷이나 서버 설치가 필요 없다.

입력 물체 / 440~660 nm / Known·Blind / auto·spectral·pyramid·periodic·compare / 결과 경로 / seed를 설정하고 명령어를 복사한다.
복사한 명령어는 `Hyperspectral_HYU_single`의 VS Code 터미널에서 실행한다. HTML 자체는 Python 계산을 실행하지 않는다.

[싱글 상세 가이드북](../가이드북.md)

## 이미지 미리보기

기존 16개 scene은 HTML에 미리보기가 내장되어 있다. 추가된 24개(Set12 12개 및 C4L 12개)는 로컬 HTML을 열면 `../../데이터/입력이미지/`의 원본 PNG를 직접 읽고, 온라인 HTMLPreview는 동일 저장소의 고정 커밋 `8bbfa05d...`의 PNG를 읽는다. 전체 40개 기본 scene 및 custom을 지원한다. 신규 이미지까지 오프라인에서 보려면 HTML만 다운로드하지 말고 저장소 전체를 최신 상태로 유지한다. Boat/Tank/Peppers는 저장소 원본, RBC 5종은 660 nm/geometry seed 20261127의 합성 위상맵(128×128 표시용)을 사용한다. Custom과 파일 override는 파일 선택으로 브라우저 미리보기 가능. 실제 CLI 경로는 별도로 입력한다. [원본 이미지 및 출처](../../데이터/입력이미지/README.md).

## 복원 알고리즘 — 추천 선택 (HTML)

HTML의 03. 복원 알고리즘 첫 번째 옵션은 **추천 알고리즘**이다. 기존 Auto 옵션을 화면에서 제거했다. 기본 선택이며 사진·파장·Known/Blind·파일 경로 설정에 따라 추천 방법과 검증 근거가 표시된다. 추천을 선택하면 실제 명령어에 spectral/pyramid/compare 중 구체적인 방법이 들어가며, 출력 폴더 이름에도 그 방법을 사용한다. 직접 다른 알고리즘을 선택할 수도 있다.

- Triple: 기존 660 nm 고정 RBC 센서에서 Spectral ROI 0.077086 rad, 49/50 (그룹31 실패). 새 장면과 다른 조건은 별도 검증 필요.
- 나머지 RBC 4종: Spectral 연구 기준, 개별 전체 검증 미완료.
- Moon/Brick/Chelsea/Camera rot90: 과거 16-case Pyramid 근거. Astronaut 660 nm 과거 RMSE 0.119159 rad. Camera/Coins 과거 4-case 일반화 근거.
- Lena 등 신규 일반 이미지: Pyramid 잠정 후보, 미검증.
- Custom 및 Boat/Tank/Pepper 파일 override: Compare 추천.

CLI의 기존 `--method auto`는 이전 명령과의 호환을 위해 그대로 유지한다. HTML의 추천은 그 옵션을 호출하지 않고 구체적 알고리즘을 선택한다. 과거 합성 실험과 신규 통합 코드 검증은 구분한다.
