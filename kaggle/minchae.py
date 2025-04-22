import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1. 학습 및 테스트 데이터 로드
train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

# 2. 데이터 합치기 (동일한 전처리를 위해)
train["is_train"] = 1
test["is_train"] = 0
test["SalePrice"] = np.nan  # 타겟 임시 생성

df_all = pd.concat([train, test], axis=0)

# 3. 수치형 & 범주형 분리
numeric_features = df_all.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df_all.select_dtypes(include=['object']).columns.tolist()

# 4. 결측치 처리
df_all[numeric_features] = df_all[numeric_features].fillna(df_all[numeric_features].median())
df_all[categorical_features] = df_all[categorical_features].fillna("-")

# 5. 더미 변수 처리
df_all_encoded = pd.get_dummies(df_all, columns=categorical_features, drop_first=True)

# 6. 학습 / 테스트 분리
train_encoded = df_all_encoded[df_all_encoded["is_train"] == 1].drop(columns=["is_train"])
test_encoded = df_all_encoded[df_all_encoded["is_train"] == 0].drop(columns=["is_train", "SalePrice"])

# 7. 학습용 X, y
X_train = train_encoded.drop(columns=["SalePrice"])
y_train = train_encoded["SalePrice"]

# 8. 라쏘 회귀 학습
lasso_pipeline = make_pipeline(
    StandardScaler(),
    LassoCV(cv=5, random_state=42, max_iter=10000)
)

lasso_pipeline.fit(X_train, y_train)
lasso_model = lasso_pipeline.named_steps['lassocv']

print(f"최적 alpha: {lasso_model.alpha_}")
print(f"Train R²: {lasso_model.score(X_train, y_train):.4f}")

# 9. 예측
y_pred = lasso_pipeline.predict(test_encoded)

# 10. 제출파일 생성
submit = pd.read_csv("sample_submission.csv")
submit["SalePrice"] = y_pred
submit.to_csv("lasso_minchae.csv", index=False)

print("lasso_baseline.csv 저장 완료!")
# 0.14046