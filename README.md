# stroke prediction

## 1. 데이터 다운로드
-https://www.kaggle.com/code/docxian/stroke-prediction/input링크 접속 후 데이터 다운로드

## 2. 전처리
- sklearn 모듈이용해 범주형 변수 처리 및 NULL값 제거

## 3. 모델학습
- RandomForestClassifier를 사용하여 뇌졸중 발생 여부를 예측하는 모델 학습

## 4. 예측
- 사용자에게 다음과 같은 정보를 입력받는다.
  성별 (Male 또는 Female)
  나이
  고혈압 여부 (1: 있음, 0: 없음)
  심장 질환 여부 (1: 있음, 0: 없음)
  결혼 여부 (Yes 또는 No)
  직업 유형 (Private, Self-employed, Govt_job, Children, Never_worked 중 하나)
  거주 유형 (Urban 또는 Rural)
  평균 혈당 농도
  체질량 지수 (BMI)
  흡연 상태 (formerly smoked, never smoked, smokes, Unknown)

## 5. 결과
- 모델이 뇌졸중 확률을 화면에 나타낸다
