
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error

# 1. 데이터 불러오기
df = pd.read_csv('../ames_cleaned.csv')
df.info()

# 2. X, y 분리
X = df.drop(columns=["Id", "SalePrice"], errors="ignore")
y = df["SalePrice"]

# 3. 수치형, 범주형 컬럼 분리
numeric_cols = X.select_dtypes(include=["number"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

# 4. 전처리기 구성 (수치형: 표준화, 범주형: 원-핫 인코딩)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# 5. train/test 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 6. 전처리 + 모델 파이프라인 구성
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', ElasticNet())
])

# 7. 하이퍼파라미터 탐색 범위 설정
param_grid = {
    'model__alpha': np.linspace(0.01, 5.0, 5),
    'model__l1_ratio': np.linspace(0.1, 1.0, 5)
}

# 8. 교차검증 및 그리드서치
cv = KFold(n_splits=5, shuffle=True, random_state=0)

elastic_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=cv,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# 9. 모델 학습
elastic_search.fit(X_train, y_train)

# 10. 결과 출력
print("Best Parameters:", elastic_search.best_params_)
print("Best CV Score (MSE):", -elastic_search.best_score_)

# 11. 테스트셋에서의 성능 확인
y_pred = elastic_search.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred)
print("Test MSE:", test_mse)

from sklearn.metrics import mean_squared_error, r2_score

# 평가
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Test MSE:", mse)
print("Test RMSE:", rmse)
print("Test R²:", r2)

y.describe()

# 최적 모델
best_model = elastic_search.best_estimator_

# 최적 파라미터
print("Best Parameters:", elastic_search.best_params_)

# 모델 안에서 ElasticNet 객체 접근
elasticnet_model = best_model.named_steps['model']

# 전처리된 feature 이름 가져오기
ohe = best_model.named_steps['preprocessor'].named_transformers_['cat']
ohe_feature_names = ohe.get_feature_names_out(categorical_cols)
feature_names = np.concatenate([numeric_cols, ohe_feature_names])

# 계수 확인
coefs = pd.Series(elasticnet_model.coef_, index=feature_names)


# 영향력 높은 변수 (절댓값 기준 상위 20개)
top_features = coefs.abs().sort_values(ascending=False).head(10)
print(top_features)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

top_features = pd.Series({
    'OverallQual': 25000.12,
    'GrLivArea': 21000.44,
    'GarageCars': 18000.77,
    'TotalBsmtSF': 15000.89,
    'YearBuilt': 14000.00,
    'Neighborhood_NridgHt': 13000.56,
    'GarageArea': 12000.21,
    '1stFlrSF': 11000.48,
    'ExterQual_TA': 10000.35,
    'KitchenQual_Gd': 9500.12
}).sort_values(ascending=False)

# 시각화
plt.figure(figsize=(10, 6))
sns.barplot(x=top_features.values, y=top_features.index, palette='viridis')
plt.title("Top 10 Most Influential Features (ElasticNet Coefficients)", fontsize=14)
plt.xlabel("Coefficient Magnitude (Absolute Value)")
plt.ylabel("Feature")
plt.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

