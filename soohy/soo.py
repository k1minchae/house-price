
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error

########
df = pd.read_csv('../ames_cleaned.csv')

df.columns

def add_features(df):
    df["Total_sqr_footage"] = (
        df["BsmtFinSF1"] + df["BsmtFinSF2"] + df["1stFlrSF"] + df["2ndFlrSF"]
    )
    df["Total_Bathrooms"] = (
        df["FullBath"]
        + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"]
        + 0.5 * df["BsmtHalfBath"]
    )
    df["Total_porch_sf"] = (
        df["OpenPorchSF"]
        + df["3SsnPorch"]
        + df["EnclosedPorch"]
        + df["ScreenPorch"]
        + df["WoodDeckSF"]
    )
    df["TotalHouse"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalArea"] = (
        df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"] + df["GarageArea"]
    )
    return df


def drop_redundant_columns(df):
    cols_to_drop = [
        "BsmtFinSF1",
        "BsmtFinSF2",
        "1stFlrSF",
        "2ndFlrSF",
        "FullBath",
        "HalfBath",
        "BsmtFullBath",
        "BsmtHalfBath",
        "OpenPorchSF",
        "3SsnPorch",
        "EnclosedPorch",
        "ScreenPorch",
        "WoodDeckSF",
        "TotalBsmtSF",
        "GarageArea",
    ]
    return df.drop(columns=cols_to_drop)



df=add_features(df)
df=drop_redundant_columns(df)
df

# 시각화 히트맵으로 EDA? 과정

# SalePrice와의 상관계수 계산
corr_matrix = df.corr(numeric_only=True)
top_corr = corr_matrix['SalePrice'].abs().sort_values(ascending=False).head(11)  # 본인 포함 상위 11개 (SalePrice 포함됨)

# 상관계수 높은 상위 10개 변수 추출 (SalePrice 제외)
top_features = top_corr.index[1:]  # SalePrice 본인은 제외
print("SalePrice와 상관계수가 높은 상위 10개 변수:")
print(top_corr[1:])


# 히트맵 그리기
plt.figure(figsize=(10, 8))
sns.heatmap(df[top_features.tolist() + ['SalePrice']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Top 10 Correlated Features with SalePrice')
plt.show()

######################################

# # 1. 데이터 불러오기
# df = pd.read_csv('../ames_cleaned.csv')
# df.info()

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

# 영향력 높은 변수 (절댓값 기준 상위 10개)
top_features_ela = coefs.abs().sort_values(ascending=False).head(10)
print(top_features_ela)

#################elastic net 변수 시각화#############
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

#######################shap##########################

import shap

# 샘플링 (계산 시간 절약)
X_sample = X_train.sample(n=100, random_state=42)

# 전처리 및 모델 추출
preprocessor = best_model.named_steps['preprocessor']
model = best_model.named_steps['model']

# 전처리된 입력 데이터
X_processed = preprocessor.transform(X_sample)

# feature 이름 가져오기
ohe = preprocessor.named_transformers_['cat']
ohe_feature_names = ohe.get_feature_names_out(categorical_cols)
feature_names = np.concatenate([numeric_cols, ohe_feature_names])

# 전처리된 데이터에 feature 이름 붙이기
X_processed_df = pd.DataFrame(X_processed.toarray() if hasattr(X_processed, 'toarray') else X_processed,
                              columns=feature_names)

# SHAP Explainer 생성 및 계산
explainer = shap.Explainer(model, X_processed_df)
shap_values = explainer(X_processed_df)

# bar plot 시각화
shap.plots.bar(shap_values, max_display=10)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# 1. SHAP 값 평균 절댓값 계산
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

# 2. 상위 10개 feature 추출
top_idx = np.argsort(mean_abs_shap)[-10:][::-1]
top_features = np.array(feature_names)[top_idx]
top_importances = mean_abs_shap[top_idx]

# 3. 시리즈로 정리
top_shap_series = pd.Series(top_importances, index=top_features)

# 4. 시각화
plt.figure(figsize=(10, 6))
sns.barplot(x=top_shap_series.values, y=top_shap_series.index, palette='viridis')
plt.title("Top 10 SHAP Feature Importances", fontsize=14)
plt.xlabel("Mean |SHAP Value|")
plt.ylabel("Feature")
plt.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

top_features_ela
top_shap_series

##############################################

# 공통 feature 추출
common_features = top_features_ela.index.intersection(top_shap_series.index)

# 공통된 변수 중요도만 시리즈로 저장
elastic_common = top_features_ela.loc[common_features]
shap_common = top_shap_series.loc[common_features]

print("공통 변수 목록:", list(common_features))
print("\n[ElasticNet 기반 중요도]")
print(elastic_common)

print("\n[SHAP 기반 중요도]")
print(shap_common)

######################OLS########################
import statsmodels.api as sm

# 공통 중요 변수 리스트
common_features = ['GrLivArea', '2ndFlrSF', 'OverallQual', 'YearBuilt']

# X, y 설정
X_ols = df[common_features]
y_ols = df['SalePrice']

# 상수항 추가 (intercept 포함)
X_ols_const = sm.add_constant(X_ols)

# OLS 모델 적합
ols_model = sm.OLS(y_ols, X_ols_const).fit()

# 회귀 결과 요약 출력
print(ols_model.summary())

# 모델 성능을 개선시켜보자! 
# 이상치 제거 위해 박스플롯 그려보기

######################박스 플롯##########################

import matplotlib.pyplot as plt
import seaborn as sns

# 공통 변수들
common_features = ['GrLivArea', '2ndFlrSF', 'OverallQual', 'YearBuilt']

# 데이터 준비 (SalePrice와 비교할 때 boxplot)
plt.figure(figsize=(15, 10))

# 각 변수에 대해 박스플롯 그리기
for i, feature in enumerate(common_features, 1):
    plt.subplot(2, 2, i)  # 2x2 그리드
    sns.boxplot(x=df[feature], color='lightblue')
    plt.title(f'Boxplot of {feature}')
    plt.tight_layout()

plt.show()

####################박스 플롯#######################
import matplotlib.pyplot as plt
import seaborn as sns

# 공통 변수들
common_features = ['GrLivArea', '2ndFlrSF', 'OverallQual', 'YearBuilt']

# 데이터 준비 (SalePrice와 비교할 때 boxplot)
plt.figure(figsize=(10, 12))

# 각 변수에 대해 박스플롯 그리기
for i, feature in enumerate(common_features, 1):
    plt.subplot(2, 2, i)  # 2x2 그리드
    sns.boxplot(y=df[feature], color='lightblue')
    plt.title(f'Boxplot of {feature}')
    plt.tight_layout()

plt.show()
#################이상치 제거####################
# 이상치 제거 함수 정의
def remove_outliers(df, features):
    for feature in features:
        Q1 = df[feature].quantile(0.25)  # 1사분위수
        Q3 = df[feature].quantile(0.75)  # 3사분위수
        IQR = Q3 - Q1  # IQR
        lower_bound = Q1 - 1.5 * IQR  # 하한선
        upper_bound = Q3 + 1.5 * IQR  # 상한선
        df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]  # 이상치 제거
    return df

# 공통 변수들
common_features = ['GrLivArea', '2ndFlrSF', 'OverallQual', 'YearBuilt']

# 이상치 제거
df_cleaned = remove_outliers(df, common_features)

# 이상치 제거 후 데이터 확인
print(f"Before removing outliers: {df.shape[0]} rows")
print(f"After removing outliers: {df_cleaned.shape[0]} rows")

###############이상치 제거해서 박스플롯#######
import matplotlib.pyplot as plt
import seaborn as sns

# 공통 변수들
common_features = ['GrLivArea', '2ndFlrSF', 'OverallQual', 'YearBuilt']

# 이상치 제거
df_cleaned = remove_outliers(df, common_features)

# 데이터 준비 (이상치 제거 후 boxplot)
plt.figure(figsize=(10, 12))

# 각 변수에 대해 박스플롯 그리기
for i, feature in enumerate(common_features, 1):
    plt.subplot(2, 2, i)  # 2x2 그리드
    sns.boxplot(y=df_cleaned[feature], color='lightblue')
    plt.title(f'Boxplot of {feature} (After Removing Outliers)')
    plt.tight_layout()

plt.show()

########################다시 선형 회귀 모델 돌리기######
import statsmodels.api as sm
import pandas as pd

# 독립 변수와 종속 변수 정의
X = df_cleaned[['GrLivArea', '2ndFlrSF', 'OverallQual', 'YearBuilt']]
y = df_cleaned['SalePrice']

# 상수항 추가
X = sm.add_constant(X)

# OLS 선형 회귀 모델 학습
model = sm.OLS(y, X).fit()

# 결과 출력
print(model.summary())


###########################로그 변환?###########################
from sklearn.linear_model import LinearRegression

# 이상치 제거된 데이터로 로그 변환
df_cleaned['LogSalePrice'] = np.log(df_cleaned['SalePrice'])

# X, y 재설정 (SalePrice -> LogSalePrice로 변경)
X_cleaned = df_cleaned[['GrLivArea', '2ndFlrSF', 'OverallQual', 'YearBuilt']]
y_cleaned = df_cleaned['LogSalePrice']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.3, random_state=42)

# 선형 회귀 모델 학습
model_cleaned = LinearRegression()
model_cleaned.fit(X_train, y_train)

# 예측
y_pred_cleaned = model_cleaned.predict(X_test)

# 성능 평가
mse_cleaned = mean_squared_error(y_test, y_pred_cleaned)
rmse_cleaned = np.sqrt(mse_cleaned)
r2_cleaned = r2_score(y_test, y_pred_cleaned)

print("Test MSE:", mse_cleaned)
print("Test RMSE:", rmse_cleaned)
print("Test R²:", r2_cleaned)

# 회귀 계수 출력
print("\nCoefficients:")
for feature, coef in zip(X_cleaned.columns, model_cleaned.coef_):
    print(f"{feature}: {coef}")

##############################

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. 데이터 로딩 (이미 df_cleaned에 이상치 제거된 데이터가 있다고 가정)
# df_cleaned = pd.read_csv("your_cleaned_data.csv")  # 데이터 로딩

# 2. SalePrice에 로그 변환을 적용
df_cleaned['LogSalePrice'] = np.log(df_cleaned['SalePrice'])

# 3. X, y 데이터 준비 (SalePrice -> LogSalePrice로 변경)
X_cleaned = df_cleaned[['GrLivArea', '2ndFlrSF', 'OverallQual', 'YearBuilt']]  # 독립 변수
y_cleaned = df_cleaned['LogSalePrice']  # 종속 변수 (로그 변환된 SalePrice)

# 4. Train/Test Split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.3, random_state=42)

# 5. 선형 회귀 모델 학습
model_cleaned = LinearRegression()
model_cleaned.fit(X_train, y_train)

# 6. 예측
y_pred_cleaned = model_cleaned.predict(X_test)

# 7. 성능 평가
mse_cleaned = mean_squared_error(y_test, y_pred_cleaned)
rmse_cleaned = np.sqrt(mse_cleaned)
r2_cleaned = r2_score(y_test, y_pred_cleaned)

print("Test MSE:", mse_cleaned)
print("Test RMSE:", rmse_cleaned)
print("Test R²:", r2_cleaned)

# 8. 예측된 값 복원 (로그 변환된 예측값을 지수 변환하여 실제 SalePrice로 복원)
y_pred_actual = np.exp(y_pred_cleaned)

# 9. 회귀 계수 출력
print("\nCoefficients:")
for feature, coef in zip(X_cleaned.columns, model_cleaned.coef_):
    print(f"{feature}: {coef}")

# 10. 실제 SalePrice와 예측값 비교 (지수 변환 후 실제 값 복원)
y_test_actual = np.exp(y_test)
comparison = pd.DataFrame({
    'Actual SalePrice': y_test_actual,
    'Predicted SalePrice': y_pred_actual
})

# 11. 예측 결과 출력
print("\nActual vs Predicted SalePrice (Log transformed values):")
print(comparison.head())



#####################################################
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. 로그 변환된 종속 변수 생성
df_cleaned['LogSalePrice'] = np.log(df_cleaned['SalePrice'])

# 2. 독립 변수 및 종속 변수 정의
X_cleaned = df_cleaned[['GrLivArea', '2ndFlrSF', 'OverallQual', 'YearBuilt']]
y_cleaned = df_cleaned['LogSalePrice']

# 3. 데이터 분할 (70% 학습, 30% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.3, random_state=42)

# 4. ElasticNet 모델 정의 및 학습 (적당한 alpha/l1_ratio 설정)
elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elastic_model.fit(X_train, y_train)

# 5. 예측
y_pred = elastic_model.predict(X_test)

# 6. 성능 평가
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Test MSE:", mse)
print("Test RMSE:", rmse)
print("Test R²:", r2)

# 7. 로그 예측값을 지수 변환하여 실제 가격으로 복원
y_pred_actual = np.exp(y_pred)
y_test_actual = np.exp(y_test)

# 8. 회귀 계수 출력
print("\nElasticNet Coefficients:")
for feature, coef in zip(X_cleaned.columns, elastic_model.coef_):
    print(f"{feature}: {coef}")

# 9. 예측 결과 비교
comparison = pd.DataFrame({
    'Actual SalePrice': y_test_actual,
    'Predicted SalePrice': y_pred_actual
})

print("\nActual vs Predicted SalePrice (Restored from Log):")
print(comparison.head())

########################################################

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. 로그 변환된 종속 변수 생성
df_cleaned['LogSalePrice'] = np.log(df_cleaned['SalePrice'])

# 2. 독립 변수 및 종속 변수 정의
X_cleaned = df_cleaned[['GrLivArea', '2ndFlrSF', 'OverallQual', 'YearBuilt']]
y_cleaned = df_cleaned['LogSalePrice']

# 3. 데이터 분할 (70% 학습, 30% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.3, random_state=42)

# 4. ElasticNet 모델 정의 및 학습 (적당한 alpha/l1_ratio 설정)
elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elastic_model.fit(X_train, y_train)

# 5. 예측
y_pred = elastic_model.predict(X_test)

# 6. 성능 평가
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Test MSE:", mse)
print("Test RMSE:", rmse)
print("Test R²:", r2)

# 7. 로그 예측값을 지수 변환하여 실제 가격으로 복원
y_pred_actual = np.exp(y_pred)
y_test_actual = np.exp(y_test)

# 8. 회귀 계수 출력
print("\nElasticNet Coefficients:")
for feature, coef in zip(X_cleaned.columns, elastic_model.coef_):
    print(f"{feature}: {coef}")

# 9. 예측 결과 비교
comparison = pd.DataFrame({
    'Actual SalePrice': y_test_actual,
    'Predicted SalePrice': y_pred_actual
})

print("\nActual vs Predicted SalePrice (Restored from Log):")
print(comparison.head())

