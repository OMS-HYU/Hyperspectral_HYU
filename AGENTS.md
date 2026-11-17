# 사용자 연구 인수인계 지침 — 최신 정정 2026-11-17

## 최신 이관 개정 — 이하 과거 경로 지시보다 우선

- 사용자가 모든 작업을 **현재 Hyperspectral_HYU 프로젝트로 이관**하라고 지시했다. 활성코드 `싱글/native/코드/`, 연구기록 `싱글/native/연구기록/`, 원본/센서/체크포인트 `싱글/native/실험/`, Python `.venv-native/Scripts/python.exe`에서 실행/저장한다. 859파일(1,773,743,159 bytes)의 복사 SHA-256 일치를 확인했다. 이전 폴더는 보존용 백업이며 실행하지 않는다.
- 유일한 원격은 `https://github.com/OMS-HYU/Hyperspectral_HYU.git`(공개). 결과창은 `https://oms-hyu.github.io/Hyperspectral_HYU/`. `공개결과/싱글/`의 JSON/피겨와 `site/`의 공개페이지를 동기화한다. 외부 비공개 이미지URL에 의존하지 않는다. 코드·기록도 이 공개 프로젝트에서 관리한다.
- 큰원본/NPY/NPZ/센서/optimizer/환경/로그는 같은 프로젝트 내에 보존하고 Git에서 제외한다. 다른 저장소의 Git이력·중복clone은 복사하지 않았다. 이관검증은 `실험/이관_20261117/verified_files.json`.
- 실제 GPT 싱글스펙 최신 대화를 매 작업 전에 확인한다. 물리조건은 native5713²/2857²/512²/3000²,350nm/1.85µm,400µm/6.3mm,기존고정배경/amplitude 유지. 사이트는 **피겨 형식만** 참고한다.
- 일반사진은 GD≤150/L≤150, truth-free 수렴 기준으로 조기 종료. 물리RBC FullGD100+SpectralPeriodicL100과 별도로 검증하며 ROI/그룹정보가 없으면 미검증으로 표시한다. 현미경사진형3종은 물리RBC5종과 구분한다.
- 피겨 고정: 원본 / 관측스페클 / True phase / Estimated phase / wrapped오차. amplitude 그대로 inferno log1p(A/mean(A)), 위상viridis0~2π, 오차coolwarm±π. 표시목적으로 제곱/배경제거/센서대체하지 않는다.
- 게시실행기는 `tools/public_results.py`. 로컬저장·Git푸시·Pages배포 성공은 별도 확인한다. 이전 원격 게시실행기는 이관본에서도 비활성화했다. 멀티/연속은 수정하지 않는다.

## 이하: 이관 이전 기록(과거 경로를 활성 실행 지시로 사용하지 않음)

## 최우선: capstone_optics 사용 금지 및 공개 결과 위치

사용자가 명시적으로 `capstone_optics` 사용을 금지했다. 해당 원격 저장소에 fetch/clone/commit/push/배포하지 않는다. 로컬 `capstone_optics/` 경로의 계산 및 자동 게시도 중단했다. 기존 파일은 삭제하지 않고 증거 확인·보존 목적으로만 읽는다. 이 경로의 실행기를 다시 시작하지 않는다. 아래의 과거 경로 참조는 이 금지 지시 이전 기록이다.

사용할 프로젝트/공개 결과 대상은 `OMS-HYU/Hyperspectral_HYU`와 https://oms-hyu.github.io/Hyperspectral_HYU/ 이다. 후속 구현은 이 프로젝트의 적절한 경로로 별도 정리하고, 사용 금지 경로의 프로세스를 임의로 재개하지 않는다. 원본 배열/optimizer를 공개하지 않고 검증된 피겨·평가 JSON을 공개 결과창에 게시한다.

이번 대화에서 잘못된 원격에 올린 것은 커밋 `68c9895a49b4a1235e655fd6530c0a5cd8b5ae6c`의 결과10파일뿐이다. 원격 코드·raw·optimizer 업로드는 없었다. 이 커밋은 삭제/되돌리지 않은 상태이며, 사용 금지 원격에 추가 접근해 수정하지 않는다. 자세한 사실은 docs/PUBLICATION_AUDIT_20261117.md 참조.

싱글 스펙트럼 연구 작업은 ChatGPT **싱글스펙** 대화 ID `6afbf72d-51c4-83ee-85c2-82c0c08e46e2`의 최신 연구 개정부터 확인한다. `list_threads`/`read_thread`로 읽고, 싱글 후속 대화가 명시되면 연결해서 최신 내용을 확인한다. 읽지 않은 대화를 읽었다고 말하지 않는다. 현재 사용자 정정이 과거 대화 요약보다 우선한다.

현재 이 작업의 상세 규격과 실행기는 `capstone_optics/싱글_하이퍼스펙/AGENTS.md`, `연구기록/UNIFIED_SPECTRUM_KO.md`, `연구기록/LATEST_SINGLE_CHAT_REVISION.md`를 따른다. 루트의 과거 3800² 코드/README를 근거로 이 native 실험의 물리조건을 임의 변경하지 않는다.

사용자가 지정한 https://oms-hyu.github.io/Hyperspectral_HYU/ 는 **피겨의 배치·색상·표시 방식만 참고**한다. 물리조건은 GPT 싱글스펙 최신 개정 기준이다. 피겨 변경 요청으로 전파 격자·개구·거리·센서 피치·관측 물리량·배경장·최적화 조건을 변경하지 않는다. 다섯 패널은 원본 사진 / 관측 스페클 / True phase / Estimated phase / 오차다.

멀티/연속은 별도 연구이며 싱글 작업 때문에 수정하지 않는다.
