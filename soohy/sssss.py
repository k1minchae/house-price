
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

# 변수 합친거 저장
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
common_features = ['TotalArea', 'YearBuilt', 'GrLivArea', 'OverallQual', 'Total_sqr_footage']

# 모든 이상치 제거 적용
df_cleaned = df.copy()
for col in common_features:
    df_cleaned = remove_outliers_iqr(df_cleaned, col)

print(f"이상치 제거 후 데이터 크기: {df_cleaned.shape}")


#########################################

# 시각화# 드롭다운
import plotly.graph_objects as go
import numpy as np

# 공통 변수 리스트
common_features = ['TotalArea', 'YearBuilt', 'GrLivArea', 'OverallQual', 'Total_sqr_footage']
x_var = common_features[0]  # 기본 X축

# 초기 회귀선 계산
x = df_cleaned[x_var]
y = df_cleaned['SalePrice']
coef = np.polyfit(x, y, 1)  # 1차 회귀
reg_line = coef[0] * x + coef[1]

# Figure 생성
fig = go.Figure()

# 산점도
fig.add_trace(
    go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(color='steelblue', size=8, line=dict(width=1, color='DarkSlateGrey')),
        name='Data'
    )
)

# 회귀선
fig.add_trace(
    go.Scatter(
        x=x,
        y=reg_line,
        mode='lines',
        line=dict(color='red', width=2),
        name='Regression Line'
    )
)

# 드롭다운 버튼 설정
buttons = []
for feature in common_features:
    x = df_cleaned[feature]
    y = df_cleaned['SalePrice']
    coef = np.polyfit(x, y, 1)
    reg_line = coef[0] * x + coef[1]
    buttons.append(
        dict(
            label=feature,
            method='update',
            args=[
                {'x': [x, x], 'y': [y, reg_line]},
                {'xaxis.title': feature,
                 'title': f'SalePrice vs {feature}'}  # 제목을 동적으로 업데이트
            ]
        )
    )

# 레이아웃 설정
fig.update_layout(
    title=f'SalePrice vs {x_var}',  # 초기 제목 설정
    template='plotly_white',
    width=800,
    height=600,
    xaxis_title=x_var,
    yaxis_title='SalePrice',
    updatemenus=[dict(
        buttons=buttons,
        direction='down',
        showactive=True,
        x=0.6,  # 드롭다운을 오른쪽으로 살짝 이동
        xanchor='center',
        y=1.15,
        yanchor='top'
    )]
)

fig.show()
########################################
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 공통 변수 리스트
common_features = ['TotalArea', 'YearBuilt', 'GrLivArea', 'OverallQual', 'Total_sqr_footage']

# 데이터 준비
X = df_cleaned[common_features]
y = df_cleaned['SalePrice']

# 데이터 분할 (학습 데이터와 테스트 데이터)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 데이터 스케일링 (ElasticNet은 스케일에 민감하므로 반드시 필요)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ElasticNet 모델 생성 및 학습
elastic_net = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)
elastic_net.fit(X_train_scaled, y_train)

# 회귀 계수 출력
coefficients = elastic_net.coef_

# 변수와 회귀 계수 결합
coefficients_df = pd.DataFrame({
    'Feature': common_features,
    'Coefficient': coefficients
})

# 각 특성의 변화에 따른 집값 변화 (단위 변화시)
coefficients_df['Price Change per Unit'] = coefficients_df['Coefficient']

# 결과 출력
print(coefficients_df)
