# 싱글 스펙트럴 명령어 생성기

**[🌐 브라우저에서 바로 열기](https://htmlpreview.github.io/?https://github.com/OMS-HYU/Hyperspectral_HYU/blob/c6f888cbc463dbb5a33a6eeba51d95a073d63bc6/%EC%8B%B1%EA%B8%80/%EC%BD%94%EB%93%9C%EC%83%9D%EC%84%B1%EA%B8%B0/index.html)**

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

모든 16개 기본 scene의 미리보기는 HTML에 내장된다. Boat/Tank/Peppers는 저장소 원본, RBC 5종은 660 nm/geometry seed 20261127의 합성 위상맵(128×128 표시용)을 사용한다. Custom과 파일 override는 파일 선택으로 브라우저 미리보기 가능. 실제 CLI 경로는 별도로 입력한다. [원본 이미지 및 출처](../../데이터/입력이미지/README.md).
