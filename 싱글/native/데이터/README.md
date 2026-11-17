# 실제 입력 데이터

- 05_NN_metacam-main.zip: 기존 저장소 capstone_optics/05_NN_metacam-main.zip과 동일 Git blob, 내부 LUT/width map 해시는 코드의 check_b17()에서 검증. 원본을 수정하지 않는다.
- 입력이미지/: 기존 저장소의 Cameraman, Boat, Tank와 Tank 출처 문서 실제 파일. 기본 camera 모델은 scikit-image의 camera()를 사용하며, 저장된 이미지를 사용하려면 custom --image-file 경로를 지정한다.
- 합성 RBC의 물리적 입력은 코드/unified_object_models.py에서 생성한다. 센서/정답 대형 배열은 결과 폴더에 생성한다. 실제 광학 측정 데이터라고 주장하지 않는다.
