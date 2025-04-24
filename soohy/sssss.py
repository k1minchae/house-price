
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

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

# 예: 이상치 제거된 데이터를 저장하고 싶다면
df.to_csv("df_cleaned.csv", index=False)


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

#####################################
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

######################################################

# 상관계수 기반 top 10 변수
heatmap_top_features = set(top_features)

# 모델 기반 top 10 변수
elastic_top_features = set(top_features_ela.index)

# 공통된 변수
common_features = heatmap_top_features & elastic_top_features
print("공통된 변수:")
print(common_features)


#######################################################

# 이상치 제거용 함수 정의 (IQR 방식)
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    cleaned_df = df[(df[column] >= lower) & (df[column] <= upper)]
    return cleaned_df

# 공통 변수 목록
common_features = ['TotalArea', 'YearBuilt', 'GrLivArea', 'OverallQual']

# 모든 이상치 제거 적용
df_cleaned = df.copy()
for col in common_features:
    df_cleaned = remove_outliers_iqr(df_cleaned, col)

print(f"이상치 제거 후 데이터 크기: {df_cleaned.shape}")

##########################################

import matplotlib.pyplot as plt
import seaborn as sns

# 공통 변수 리스트 (4개)
common_features = ['TotalArea', 'YearBuilt', 'GrLivArea', 'OverallQual',"Total_sqr_footage"]

# 시각화
plt.figure(figsize=(12, 10))
for i, feature in enumerate(common_features):
    plt.subplot(2, 2, i + 1)  # 2행 2열
    sns.regplot(
        x=df_cleaned[feature],
        y=df_cleaned['SalePrice'],
        scatter_kws={'alpha': 0.5},
        line_kws={'color': 'red'}
    )
    plt.title(f'{feature} vs SalePrice')
    plt.xlabel(feature)
    plt.ylabel('SalePrice')

plt.tight_layout()
# plt.suptitle("공통 영향 변수들과 SalePrice의 관계", fontsize=16, y=1.02)
plt.show()

df_cleaned[['GrLivArea', 'Total_sqr_footage', 'TotalArea']].corr()