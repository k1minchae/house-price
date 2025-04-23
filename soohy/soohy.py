#####################################################
##########Lasso####################
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

# 1. 데이터 불러오기
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
train_df.info()

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
best_alpha = lasso.named_steps['lassocv'].alpha_
print(f"최적 alpha: {best_alpha}")

# 7. 예측
y_pred = lasso.predict(X_test)

# 8. 결과 저장
# submit = pd.read_csv('sample_submission.csv')
# submit["SalePrice"] = y_pred
# submit.to_csv('lasso_baseline.csv', index=False)

# 9. 가장 영향력 있는 변수 추출

# LassoCV 계수 추출
lasso_model = lasso.named_steps['lassocv']
coefficients = lasso_model.coef_

# 변수명과 계수를 데이터프레임으로 정리
coef_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": coefficients
})

# 절대값 기준으로 정렬 후 상위 10개 출력
top_features_Lasso = coef_df.reindex(coef_df.Coefficient.abs().sort_values(ascending=False).index).head(10)
print("집값에 가장 영향을 많이 주는 변수 Top 10:")
print(top_features_Lasso)

##########################################
#shap#####

import shap
from sklearn.ensemble import RandomForestRegressor

# 모델 훈련 (예시로 RandomForest 사용)
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# SHAP 값 계산
explainer = shap.Explainer(rf, X_train)
shap_values = explainer(X_train)

# 요약 플롯 (변수 중요도 시각화)
shap.plots.bar(shap_values)

# 각 변수별로 SHAP 값의 절댓값 평균 계산
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

# 변수 이름과 평균 SHAP 값을 데이터프레임으로 정리
shap_importance = pd.DataFrame({
    'feature': X_train.columns,
    'mean_abs_shap': mean_abs_shap
})

# 중요도 기준으로 정렬 후 상위 10개 추출
top_10_features_shap = shap_importance.sort_values(by='mean_abs_shap', ascending=False).head(10)

print(top_10_features_shap)


########################################################################
#SFS
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold

# -------------------------------------------------
# 1. 데이터 로드 (주석 해제 시 직접 로드)
# train_df = pd.read_csv('train.csv')
# -------------------------------------------------

# 2. 전처리
X_num = train_df.select_dtypes(include=['number']).drop(columns=['Id', 'SalePrice'])
X_cat = pd.get_dummies(train_df.select_dtypes(include=['object', 'category']),drop_first=True).astype(int)

X = pd.concat([X_num, X_cat], axis=1)
y = train_df['SalePrice']

len(X_num.columns)

# 3. 결측치
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_num = pd.DataFrame(imputer.fit_transform(X_num), columns=X_num.columns)

# 4. Train/Valid 분할  (SFS 내부 cross‑val 과는 별개)
X_train, X_val, y_train, y_val = train_test_split(
        X_num, y, test_size=0.2, random_state=42)

# 5. AIC 스코어 함수 (mlxtend용 → est, X, y 받아야 함)
def aic_scorer(estimator, X_data, y_data):
    # 예측값 필요 없이 OLS로 직접 적합해야 하므로 statsmodels 사용
    X_df = pd.DataFrame(X_data, columns=X_train.columns)
    X_sm = sm.add_constant(X_df)
    model = sm.OLS(y_data, X_sm).fit()
    # mlxtend는 '클수록 좋음' 스코어를 기대 → 음수로 바꿔서 최소화 ↔ 최대화 변환
    return -model.aic

# 6. SFS 설정
base_lr = LinearRegression()

sfs = SFS(
    estimator=base_lr,
    k_features=(1, X_train.shape[1]),   # <-- 여기 수정
    forward=True,
    floating=False,
    scoring=aic_scorer,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1
)

sfs.fit(X_train.values, y_train.values)  # 이제 오류 없이 동작

# 7. 학습 (numpy array 전달)
sfs = sfs.fit(X_train.values, y_train.values)

# 8. 선택된 피처
selected_idx = list(sfs.k_feature_idx_)
selected_cols = X_train.columns[selected_idx]
print("선택된 변수:", list(selected_cols))

# 9. 최종 OLS 적합 및 요약
X_sel_const = sm.add_constant(X[selected_cols])
final_model = sm.OLS(y, X_sel_const).fit()
print(final_model.summary())

##############################################
###permutation_importance###

from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance

# 1. 최종 모델 학습
final_model = LinearRegression()
final_model.fit(X_train[selected_cols], y_train)

# 2. Permutation Importance 계산
perm_importance = permutation_importance(final_model, X_val[selected_cols], y_val, n_repeats=10, random_state=42)

# 3. 중요도 출력
perm_importance_df = pd.DataFrame({
    'Feature': selected_cols,
    'Importance': perm_importance.importances_mean
}).sort_values(by='Importance', ascending=False)

print(perm_importance_df)

# 상위 10개 변수만 출력
top_10_features = perm_importance_df.head(10)
print(top_10_features)

############################################################
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import statsmodels.api as sm

# 1. 데이터 불러오기
train_df = pd.read_csv('train.csv')

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

# 6. SFS로 AIC 기반 feature 선택
def aic_score(estimator, X, y):
    X_ = sm.add_constant(X)
    model = sm.OLS(y, X_).fit()
    return -model.aic  

model = LinearRegression()
sfs = SFS(
    model,
    k_features="best",
    forward=True,
    scoring=aic_score,
    cv=0,
)
sfs = sfs.fit(X, y)

# 7. 선택된 피처로 OLS 학습
selected_features = list(sfs.k_feature_names_)
X_selected = sm.add_constant(df[selected_features])
ols_model = sm.OLS(y, X_selected).fit()
print(ols_model.summary())
final_model = LinearRegression()
final_model.fit(X[selected_features], y)

len(selected_features)


###########test##########
# 4. test 데이터 불러오기 및 전처리
test_df = pd.read_csv('test.csv')

test_numeric = test_df.select_dtypes(include=["number"])
test_categorical = test_df.select_dtypes(include=["object", "category"])
test_categorical = pd.get_dummies(test_categorical, drop_first=True).astype(int)

X_test_full = pd.concat([test_numeric, test_categorical], axis=1)
X_test_full = X_test_full.fillna(X_test_full.mean())

# 5. train에 있었던 컬럼 기준으로 누락된 컬럼은 train 평균으로 채우고
for col in X.columns:
    if col not in X_test_full.columns:
        X_test_full[col] = X[col].mean()

# test에만 있는 컬럼은 제거
extra_cols = set(X_test_full.columns) - set(X.columns)
X_test_full.drop(columns=extra_cols, inplace=True)

X_test_full = X_test_full[X.columns]

# 6. 선택된 피처만 추출
X_test_selected = X_test_full[selected_features]

# 7. 예측 및 제출 파일 저장
y_pred = final_model.predict(X_test_selected)
y_pred = np.where(y_pred < 0, 0, y_pred)  # 음수 방지 조치

submit = pd.read_csv('./data/house/sample_submission.csv')
submit["SalePrice"] = y_pred
submit.to_csv('./data/house/baseline.csv', index=False)