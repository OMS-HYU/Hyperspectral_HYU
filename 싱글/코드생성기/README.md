# 싱글 스펙트럴 명령어 생성기

**[🌐 브라우저에서 바로 열기](https://htmlpreview.github.io/?https://github.com/OMS-HYU/Hyperspectral_HYU/blob/6d59367bd0e7bad5bc22cc1c7b2c5b90a6f9d712/%EC%8B%B1%EA%B8%80/%EC%BD%94%EB%93%9C%EC%83%9D%EC%84%B1%EA%B8%B0/index.html)**

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

## 이미지별 추천과 Auto의 차이

Auto는 적혈구 5종 spectral, 일반/Custom pyramid를 실행하는 기존 규칙이다. HTML의 추천 패널은 과거 검증 근거와 범위를 표시하며, 실제 선택과 별도로 유지한다. 추천 방법 적용 버튼은 명시 method를 설정한다. Triple은 역사적 고정 660 nm 기준 ROI 0.077086 rad, 49/50(그룹31 실패); 새 합성 장면 및 다른 파장/Blind는 미검증. Moon/Brick/Chelsea/Camera rot90은 과거 16-case Pyramid 검증 근거, Astronaut 660 nm는 과거 RMSE 0.119159 rad, Camera/Coins는 과거 4-case 일반화 근거가 있다. 나머지 일반 장면은 Pyramid 기본 후보(미검증). Custom 및 파일 override는 Compare 추천이나 Auto는 그대로 Pyramid이다. 새 통합 코드의 full-grid 재검증 또는 모든 이미지의 최적 알고리즘 판정으로 해석하지 않는다.
