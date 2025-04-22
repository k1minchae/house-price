import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import statsmodels.api as sm

# 1. 데이터 불러오기
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. 숫자형/범주형 분리
numeric_df = train_df.select_dtypes(include=["number"])
categorical_df = train_df.select_dtypes(include=["object", "category"])

# 3. 범주형 변수 더미 인코딩
categorical_df = pd.get_dummies(categorical_df, drop_first=True).astype(int)

# 4. 합치고 결측치 제거
df = pd.concat([numeric_df, categorical_df], axis=1).dropna()

# 5. X, y 분리
y = df["SalePrice"]
X = df.drop(columns=["SalePrice", "Id"])

from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 표준화 + LassoCV 파이프라인 구성
lasso_cv = make_pipeline(
    StandardScaler(),                  # 스케일링 (Lasso는 스케일에 민감함)
    LassoCV(cv=5, random_state=42)     # 교차검증으로 최적 alpha 탐색
)

# 모델 학습
lasso_cv.fit(X, y)

# 최적 alpha 확인
best_alpha = lasso_cv.named_steps['lassocv'].alpha_
print(f"최적 alpha: {best_alpha}")


# 예측
y_pred = lasso.predict(X)

# 결과 저장
submit = pd.read_csv('sample_submission.csv')
submit["SalePrice"] = y_pred
submit.to_csv('lasso_baseline.csv', index=False)


#####################################################

import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

# 1. 데이터 불러오기
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. 숫자형/범주형 분리
X_train_num = train_df.select_dtypes(include=["number"]).drop(columns=["Id", "SalePrice"])
X_test_num = test_df.select_dtypes(include=["number"]).drop(columns=["Id"])
y_train = train_df["SalePrice"]

X_train_cat = pd.get_dummies(train_df.select_dtypes(include=["object", "category"]), drop_first=True).astype(int)
X_test_cat = pd.get_dummies(test_df.select_dtypes(include=["object", "category"]), drop_first=True).astype(int)

# 3. 컬럼 정렬 맞추기
X_train_cat, X_test_cat = X_train_cat.align(X_test_cat, join='left', axis=1, fill_value=0)

# 4. 수치 + 범주형 합치기
X_train = pd.concat([X_train_num, X_train_cat], axis=1)
X_test = pd.concat([X_test_num, X_test_cat], axis=1)

# 5. 결측치 처리 (평균으로)
imputer = SimpleImputer(strategy="mean")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# 6. LassoCV 파이프라인 구성 및 학습
lasso = make_pipeline(
    StandardScaler(),
    LassoCV(cv=5, random_state=42)
)
lasso.fit(X_train, y_train)

# 최적 alpha 확인
best_alpha = lasso_cv.named_steps['lassocv'].alpha_
print(f"최적 alpha: {best_alpha}")

# 7. 예측
y_pred = lasso.predict(X_test)

# 8. 결과 저장
submit = pd.read_csv('sample_submission.csv')
submit["SalePrice"] = y_pred
submit.to_csv('lasso_baseline.csv', index=False)
