import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
# 데이터 불러오기
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.shape, test.shape

# 결합 후 범주형, 수치형 분리
df = pd.concat([train, test])
df.shape
numeric_df = df.select_dtypes(include=['number'])
categorical_df = df.select_dtypes(include=['object', 'category'])

# 더미 변수 처리
categorical_df = pd.get_dummies(categorical_df, drop_first=True).astype(int)
df = pd.concat([numeric_df, categorical_df], axis=1)
df.shape
# 결측치 많은 컬럼 제거
null_col = df.isnull().sum().sort_values(ascending=False)[1:3].index.tolist()
df.drop(null_col, axis=1, inplace=True)
df.isnull().sum().sort_values(ascending=False)

# train, test 다시 나누기

train_df = df[:len(train)]
test_df = df[len(train):]

# train_test_split 하기 

tr_x = train_df.drop(columns=['SalePrice'])
tr_y = train_df['SalePrice']
test_df



from sklearn.model_selection import train_test_split
train_x, valid_x, train_y, valid_y = train_test_split(tr_x,tr_y,test_size=0.2, random_state=42)

# # train 결측치 제거

# train_df.dropna(inplace=True)
# test_df.fillna(0, inplace=True)  # 테스트 데이터는 예측만 하므로 임시로 0으로 대체 가능


from sklearn.linear_model import LassoCV, RidgeCV
import matplotlib.pyplot as plt
alphas = np.logspace(-3, 2, 100)

lassocv_model = LassoCV(alphas=alphas,cv=5,max_iter=10000)
lassocv_model.fit(train_x, train_y)
lassocv_model.predict(train_x)
lassocv_model.score(train_x, train_y)
lassocv_model.alpha_ # 최적의 alpha

plt.figure(figsize=(10, 5))
plt.plot(np.log10(lassocv_model.alphas_), lassocv_model.mse_path_.mean(axis=1), marker='o', label='Train MSE')
plt.legend()
plt.grid() 
plt.tight_layout()



### 
ames = pd.read_csv('ames.csv')
ames.head()
ames.shape

# 파생 변수 만들기
ames['price_area_ratio'] = ames['SalePrice'] / ames['GrLivArea']  # 평당 가격
ames['price_qual_ratio'] = ames['SalePrice'] / ames['OverallQual']  # 품질당 가격
ames['price_yrblt_ratio'] = ames['SalePrice'] / (2025 - ames['YearBuilt'])  # 연식 보정 가격

from scipy.stats import zscore


ames['z_price_sqft'] = zscore(ames['price_area_ratio'])
ames['z_qual_price'] = zscore(ames['price_qual_ratio'])
ames['z_year_adj'] = zscore(ames['price_yrblt_ratio'])

ames['ValueScore'] = -1 * (ames['z_price_sqft'] + ames['z_qual_price'] + ames['z_year_adj']) / 3

from sklearn.cluster import KMeans

cluster_features = ames[['price_area_ratio', 'price_qual_ratio', 'price_yrblt_ratio']]
kmeans = KMeans(n_clusters=3, random_state=42)
ames['Cluster'] = kmeans.fit_predict(cluster_features)

import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x='Cluster', y='ValueScore', data=ames)
plt.title("Cluster별 ValueScore 분포")
plt.show()

ames['Latitude']
ames['Longitude']

ames['Condition1'].unique()
import plotly.express as px

# 지도 스타일은 open-street-map을 사용하면 토큰 없이 사용 가능
fig = px.scatter_mapbox(
    df,
    lat="Latitude",
    lon="Longitude",
    color="Cluster",
    hover_data=["SalePrice", "GrLivArea", "OverallQual", "YearBuilt"],
    color_continuous_scale=px.colors.qualitative.Set2,  # 또는 discrete_color_sequence
    zoom=11,
    height=600
)

fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
fig.update_layout(title="Ames Housing 군집 분석 결과 (지도 시각화)")
fig.show()


# 토지 용도 구분 데이터 개수 확인 
zoning_val = ['C (all)','FV','I (all)','A (agr)', 'C (all)']
ames[ames['MSZoning'].isin(zoning_val)]
ames['MSZoning'].value_counts()

a = ames[ames['MSZoning'].isin(['FV'])].iloc[1,:]
a['Latitude'], a['Longitude']
import folium

map_fv = folium.Map(location=[a['Latitude'], a['Longitude']], zoom_start=12,tiles='OpenStreetMap')
folium.Marker(location=[a['Latitude'], a['Longitude']],tooltip='환영합니다',
                       icon=folium.Icon(icon='home', color='red')).add_to(map_fv)
map_fv


'''
# 리모델링 연도 확인 
'''

a = ames['YearRemodAdd'].unique().tolist()
a.sort()
a
a = ames['YearRemodAdd'].unique().tolist()
a.sort()
a

ames['HeatingQC'].value_counts()
ames['Electrical'].value_counts()
ames['Functional'].value_counts()

ames[ames['Electrical'].isin(['FuseP'])]['YearBuilt']
ames['YearBuilt'].sort_values(ascending=False)

a = ames[ames['YearBuilt']==1872]

# 차고 


a['']

map = folium.Map(location=[a['Latitude'], a['Longitude']], zoom_start=12,tiles='OpenStreetMap')
folium.Marker(location=[a['Latitude'], a['Longitude']],tooltip='환영합니다',
                       icon=folium.Icon(icon='home', color='red')).add_to(map)
map

sns.barplot(x = ames['YearBuilt'],y = ames['YearBuilt'].value_counts())


importance_col = ['MSZoning','OverallQual','YearBuilt','YearRemodAdd','HeatingQC','CentralAir','GrLivArea','GarageCars']

sns.barplot(x=ames['GarageCars'],y=ames['SalePrice'])

sns.barplot(ames.groupby('GarageCars')['SalePrice'].mean())

garage_group = ames.groupby('GarageCars')['SalePrice']
garage_group.describe().T

sns.boxplot(x=ames['GarageCars'],y=ames['SalePrice'])