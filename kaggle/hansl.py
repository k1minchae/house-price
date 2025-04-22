import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1. 학습 및 테스트 데이터 로드
train = pd.read_csv("./train.csv")  # SalePrice가 포함된 학습 데이터
test = pd.read_csv("./test.csv")  # SalePrice가 없는 테스트 데이터

# 2. 데이터 통합 → 같은 방식으로 전처리하기 위해 train/test를 합침
train["is_train"] = 1  # 학습 데이터임을 표시
test["is_train"] = 0  # 테스트 데이터임을 표시
test["SalePrice"] = np.nan  # 테스트셋에 타겟값(SalePrice) 열을 추가 (값은 NaN)

df_all = pd.concat([train, test], axis=0)  # train과 test 데이터를 세로 방향으로 합침

# 3. 수치형, 범주형 변수 나누기
numeric_features = df_all.select_dtypes(
    include=[np.number]
).columns.tolist()  # 숫자형 변수
categorical_features = df_all.select_dtypes(
    include=["object"]
).columns.tolist()  # 범주형 변수

# 4. 결측값 처리
# 숫자형 → 평균으로 대체 / 범주형 → "-"라는 문자열로 대체
df_all[numeric_features] = df_all[numeric_features].fillna(
    df_all[numeric_features].mean()
)
df_all[categorical_features] = df_all[categorical_features].fillna("-")

# 5. 범주형 변수 더미화 (One-hot encoding, drop_first=True → 첫 카테고리는 제거해서 다중공선성 줄임)
df_all_encoded = pd.get_dummies(df_all, columns=categorical_features, drop_first=True)

# 6. 전처리된 데이터에서 다시 train/test 분리
train_encoded = df_all_encoded[df_all_encoded["is_train"] == 1].drop(
    columns=["is_train"]
)  # 학습 데이터
test_encoded = df_all_encoded[df_all_encoded["is_train"] == 0].drop(
    columns=["is_train", "SalePrice"]
)  # 예측용 테스트 데이터

# 7. 입력 변수(X), 타겟 변수(y) 분리
X_train = train_encoded.drop(columns=["SalePrice"])  # 입력값
y_train = train_encoded["SalePrice"]  # 타겟값

# 8. 파이프라인 구성: 스케일링 + LassoCV (교차검증으로 alpha 자동 탐색)
lasso_pipeline = make_pipeline(
    StandardScaler(),  # 입력 데이터 스케일링
    LassoCV(cv=5, random_state=42),  # LassoCV: 교차검증으로 최적의 alpha를 찾는 Lasso
)

# 모델 학습
lasso_pipeline.fit(X_train, y_train)
lasso_model = lasso_pipeline.named_steps[
    "lassocv"
]  # 파이프라인 안의 LassoCV 객체 꺼내기

# 학습 결과 출력
print(f"최적 alpha: {lasso_model.alpha_}")  # 선택된 alpha (정규화 강도)
print(f"Train R²: {lasso_model.score(X_train, y_train):.4f}")  # 결정계수 R² 출력

# 9. 테스트 데이터 예측
y_pred = lasso_pipeline.predict(test_encoded)

# 10. 제출 파일 생성
submit = pd.read_csv("sample_submission.csv")  # 기본 제출 파일 형식 불러오기
submit["SalePrice"] = y_pred  # 예측 결과로 덮어쓰기
submit.to_csv("lasso_baseline.csv", index=False)  # 제출 파일 저장

print("lasso_baseline.csv 저장 완료!")  # 완료 메시지
