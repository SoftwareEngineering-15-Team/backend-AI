# backend-AI
뇌졸중 발병확률을 예측할 AI 모델입니다.
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 데이터 불러오기
data = pd.read_csv('/content/healthcare-dataset-stroke-data.csv.xls')

# 결측치 처리 
data = data.dropna()

# 범주형 변수 처리를 위해 LabelEncoder 초기화
le_gender = LabelEncoder()
le_married = LabelEncoder()
le_work_type = LabelEncoder()
le_residence_type = LabelEncoder()
le_smoking_status = LabelEncoder()

# 범주형 변수 라벨 인코딩
data['gender'] = le_gender.fit_transform(data['gender'])
data['ever_married'] = le_married.fit_transform(data['ever_married'])
data['work_type'] = le_work_type.fit_transform(data['work_type'])
data['Residence_type'] = le_residence_type.fit_transform(data['Residence_type'])
data['smoking_status'] = le_smoking_status.fit_transform(data['smoking_status'])

# 입력 변수와 목표 변수 분리
X = data.drop(['id', 'stroke'], axis=1)
y = data['stroke']

# 모델 정의 및 학습
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# 예측 함수 정의
def predict_stroke():
    # 사용자 입력 받기
    gender = input("성별을 입력하세요 (Male 또는 Female): ")
    age = int(input("나이를 입력하세요: "))
    hypertension = int(input("고혈압 여부를 입력하세요 (1: 있음, 0: 없음): "))
    heart_disease = int(input("심장 질환 여부를 입력하세요 (1: 있음, 0: 없음): "))
    ever_married = input("결혼 여부를 입력하세요 (Yes 또는 No): ")
    work_type = input("직업 유형을 입력하세요 (Private, Self-employed, Govt_job, Children, Never_worked 중 하나): ")
    Residence_type = input("거주 유형을 입력하세요 (Urban 또는 Rural): ")
    avg_glucose_level = float(input("평균 혈당 농도를 입력하세요: "))
    bmi = float(input("체질량 지수 (BMI)를 입력하세요: "))
    smoking_status = input("흡연 상태를 입력하세요 (formerly smoked, never smoked, smokes, Unknown): ")

    # LabelEncoder를 사용하여 변환
    gender_encoded = le_gender.transform([gender])[0]
    ever_married_encoded = le_married.transform([ever_married])[0]
    work_type_encoded = le_work_type.transform([work_type])[0]
    Residence_type_encoded = le_residence_type.transform([Residence_type])[0]
    smoking_status_encoded = le_smoking_status.transform([smoking_status])[0]
    
    # 입력 데이터 생성
    input_data = pd.DataFrame({
        'gender': [gender_encoded],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [ever_married_encoded],
        'work_type': [work_type_encoded],
        'Residence_type': [Residence_type_encoded],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_status_encoded]
    })
    
    # 예측
    prediction_proba = model.predict_proba(input_data)[0]
    
    
    return prediction_proba

# 예측 함수 호출
prediction_proba = predict_stroke()

print("뇌졸중 발생 확률 : {:.2f}%".format(prediction_proba[1] * 100))
