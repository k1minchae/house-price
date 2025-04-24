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
ames = pd.read_csv('../ames_cleaned.csv')
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


map = folium.Map(location=[a['Latitude'], a['Longitude']], zoom_start=12,tiles='OpenStreetMap')
folium.Marker(location=[a['Latitude'], a['Longitude']],tooltip='환영합니다',
                       icon=folium.Icon(icon='home', color='red')).add_to(map)
map

sns.barplot(x = ames['YearBuilt'],y = ames['YearBuilt'].value_counts())


importance_col = ['MSZoning','LotArea','MSZoning','OverallCond','YearBuilt','YearRemodAdd','HeatingQC','CentralAir','GrLivArea','TotRmsAbvGrd','GarageCars']

sns.barplot(ames.groupby('GarageCars')['SalePrice'].mean())

garage_group = ames.groupby('GarageCars')['SalePrice']
garage_group.describe().T

sns.boxplot(x=ames['GarageCars'],y=ames['SalePrice'])

# 샤피로 검정 
from scipy import stats
from scipy.stats import shapiro

shapiro(ames['SalePrice'])
shapiro(ames['GrLivArea'])

# 등분산성 검증
from scipy.stats import levene
levene(ames['SalePrice'], ames['GrLivArea'])

# ols 통한 anova 검정
import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols('SalePrice ~ GarageCars', data=ames).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table

# kruskal 검정l;jhkl;sdkwpeito0mp;lrgt5;l/klyjhmkltg/
from scipy.stats import kruskal
kruskal(ames['GarageCars'],ames['SalePrice'])

# 사후검정
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey = pairwise_tukeyhsd(endog=ames['SalePrice'], groups=ames['GarageCars'], alpha=0.05)
tukey.summary()

'''
# 데이터 확인 
'''

# 차고 없으면 연식이 적을까
sns.boxplot(x=ames['GarageCars'],y=ames['YearBuilt'])

# 면적에 따른 가격 차이 
plt.scatter(ames['GrLivArea'],ames['SalePrice'])

pearson_corr = ames[['GrLivArea','SalePrice']].corr(method='pearson')
# 0.72 상관계수 


# 1층만 있는 집이 있을까?

ames['2ndFlrSF'].isna().sum()
ames['2ndFlrSF'].value_counts()

sns.barplot(ames['TotRmsAbvGrd'].value_counts())
sns.histplot(ames['SalePrice'], bins=30, kde=True)
ames['YearRemodAdd'].value_counts(dropna=False)

ames['YearRemodAdd'].isna().sum()
ames[ames['YearBuilt']==2010]['YearRemodAdd']

sns.boxplot(x=ames['TotRmsAbvGrd'],y=ames['SalePrice'])

# 1층만 있는 집에서 방 개수
ames[ames['2ndFlrSF']==0]['TotRmsAbvGrd'].value_counts()

sns.barplot(x = ames['TotRmsAbvGrd'],y = ames['SalePrice'])
sns.boxplot(x=ames['TotRmsAbvGrd'],y=ames['SalePrice'])
ames['TotRmsAbvGrd'].value_counts()
ames['GarageCars'].value_counts()

ames[ames['OverallCond']==9]
ames['YardArea'] = ames['LotArea'] - ames['1stFlrSF']

sns.boxplot(x=ames['BldgType'],y=ames['SalePrice'])

ames['LivingLotRatio'] = ames['1stFlrSF'] / ames['LotArea']

sns.histplot(ames['LivingLotRatio'], bins=30, kde=True)

ames[ames['HeatingQC']=='TA']['Heating'].value_counts()

ames 






'''
########################################################################
########################################################################
### 여기서 부터 
########################################################################
########################################################################
'''


''' 
# 면적에 따른 집 값 계산 'LotArea', 'GrLivArea', '1stFlrSF', '2ndFlrSF'
'''
import seaborn as sns 
import matplotlib.pyplot as plt


ames = pd.read_csv('../ames_cleaned.csv')
ames = ames[ames['CentralAir']=='Y']

ames.head()

ames['YardArea'] = (ames['LotArea'] - ames['1stFlrSF']).clip(lower=0)

# 1인 가구: 마당 가중치 낮음
ames['WeightedArea_single'] = ames['GrLivArea'] + 0.1 * ames['YardArea']

# 2인 가구: 마당 가중치 중간
ames['WeightedArea_couple'] = ames['GrLivArea'] + 0.25 * ames['YardArea']

# 자녀가 있는 가족: 마당 가중치 높음 
ames['WeightedArea_family'] = ames['GrLivArea'] + 0.5 * ames['YardArea']

ames['PricePerWeighted_single'] = ames['SalePrice'] / ames['WeightedArea_single']
ames['PricePerWeighted_couple'] = ames['SalePrice'] / ames['WeightedArea_couple']
ames['PricePerWeighted_family'] = ames['SalePrice'] / ames['WeightedArea_family']


sns.histplot(ames['PricePerWeighted_single'])
sns.histplot(ames['PricePerWeighted_couple'])
sns.histplot(ames['PricePerWeighted_family'])

# 분위수로 4분할

def score_area(df, col='Price_Per_Area'):
    q20 = df[col].quantile(0.2)
    q40 = df[col].quantile(0.4)
    q60 = df[col].quantile(0.6)
    q80 = df[col].quantile(0.8)
    
    df[f'score_area_{col[-6:]}'] = np.where(df[col] <= q20, 5,
                    np.where(df[col] <= q40, 4,
                    np.where(df[col] <= q60, 3,
                    np.where(df[col] <= q80, 2, 1))))
    return df[f'score_area_{col[-6:]}']

score_area(ames, col='PricePerWeighted_single')
score_area(ames, col='PricePerWeighted_couple')
score_area(ames, col='PricePerWeighted_family')

sns.barplot(x= ames['score_area_single'],y = ames['SalePrice'])
sns.barplot(x= ames['score_area_couple'],y = ames['SalePrice'])
sns.barplot(x= ames['score_area_family'],y = ames['SalePrice'])

# single를 위한 집 추천 지도 시각화
import folium 

# GrLivArea 지도 표시
from folium import Marker
from folium.plugins import MarkerCluster

# 기본 지도를 생성합니다.
m = folium.Map(location=[ames['Latitude'].mean(),
                            ames['Longitude'].mean()], 
                            zoom_start=12,
                            tiles="cartodbpositron")

# 마커 클러스터 생성
marker_cluster = MarkerCluster().add_to(m)

# 점수에 따른 색상 맵핑
score_color_mapping = {
    1: 'red',
    2: 'orange',
    3: 'yellow',
    4: 'green',
    5: 'blue'
}

# house1015 데이터프레임을 반복하여 마커 추가
for _, row in ames.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        radius=5,
        color=score_color_mapping[row['score_area_single']],
        fill=True,
        fill_color=score_color_mapping[row['score_area_single']],
        fill_opacity=0.7,
        popup=f"Score: {row['score_area_single']}<br>Price: ${row['SalePrice']:,.0f}<br>Area: {row['GrLivArea']} sqft"
    ).add_to(marker_cluster)  # 생성된 마커를 바로 클러스터에 추가
m

'''
# 이거는 마커 버전 
'''
# 점수 기반 색상 함수 (예시: 1~5점 → 색깔)
def get_color(score):
    if score == 5:
        return 'darkgreen'
    elif score == 4:
        return 'green'
    elif score == 3:
        return 'orange'
    elif score == 2:
        return 'red'
    else:
        return 'darkred'


for _, row in ames.iterrows():
    score = row['score_area_single']  # ← 여기에 본인의 점수 컬럼 이름 사용
    color = get_color(score)

    popup_text = f"""<b>Price:</b> ${row['SalePrice']:,}<br>
                     <b>면적당 점수:</b> {score}점"""
    
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        tooltip='클릭해서 정보 보기',
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(icon='home', color=color)
    ).add_to(map)

map

'''
# 주택 전체 등급으로 점수 매기기 'OverallCond'
'''
def cond_score(x):
    if x >= 9:
        return 5
    elif x >= 7:
        return 4
    elif x >= 5:
        return 3
    elif x >= 3:
        return 2
    else:
        return 1

# 새 컬럼 생성
ames['Cond_Score'] = ames['OverallCond'].apply(cond_score)




'''
# 연식 & 리모델링에 따른 집 값 계산 'YearBuilt', 'YearRemodAdd'
'''
import seaborn as sns 
import matplotlib.pyplot as plt


ames = pd.read_csv('../ames_cleaned.csv')
ames.head()

ames['recent_built'] = ames[['YearBuilt', 'YearRemodAdd']].max(axis=1)
ames['recent_built']

sns.barplot(ames['recent_built'].value_counts())

ames['recent_built'].describe()

# 년도를 기준으로 집의 가치를 평가하는 새로운 변수를 생성

# 같은 간격인 12년 단위로 나누기
# 1950 , 1962, 1974, 1986, 1998, 2010 간격으로 5점 만점으로 나누기

bins = [0, 1950, 1962, 1974, 1986, float('inf')]
labels = [1, 2, 3, 4, 5]

ames['Year_Score'] = pd.cut(ames['recent_built'], bins=bins, labels=labels).astype(int)

sns.barplot(x= ames['Year_Score'],y = ames['SalePrice'])

def get_color(score):
    if score == 5:
        return 'darkgreen'
    elif score == 4:
        return 'green'
    elif score == 3:
        return 'orange'
    elif score == 2:
        return 'red'
    else:
        return 'darkred'


for _, row in ames.iterrows():
    score = row['Year_Score']  # ← 여기에 본인의 점수 컬럼 이름 사용
    color = get_color(score)

    popup_text = f"""<b>Price:</b> ${row['SalePrice']:,}<br>
                     <b>연식 당 점수:</b> {score}점"""
    
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        tooltip='클릭해서 정보 보기',
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(icon='home', color=color)
    ).add_to(map)

map

# 호버해서 나오는 것도 확실하게 보이게 하기 
# 단위도 같이 나오게끔 보여주기 
# 


'''
# 냉난방 여부 점수 매기기 
'''

ames['CentralAir'].value_counts()

# ames = ames[ames['CentralAir']=='Y'] # 위에서 했음

ames['HeatingQC'].value_counts()

# Ex : 5점 , TA : 4점 , Gd : 3점 , Fa : 2점 , Po : 1점

# 냉난방 여부에 따라 점수 매기기
def score_heating(df, col='HeatingQC'):
    df['score_heating'] = np.where(df[col] == 'Ex', 5,
                    np.where(df[col] == 'TA', 4,
                    np.where(df[col] == 'Gd', 3,
                    np.where(df[col] == 'Fa', 2, 1))))
    return df['score_heating']
score_heating(ames, col='HeatingQC')

ames['score_heating'].value_counts()

sns.barplot(x = ames['score_heating'], y = ames['SalePrice'])


m_heat = folium.Map(location=[ames['Latitude'].mean(),
                            ames['Longitude'].mean()], 
                            zoom_start=12,
                            tiles="cartodbpositron")

def get_color(score):
    if score == 5:
        return 'darkgreen'
    elif score == 4:
        return 'green'
    elif score == 3:
        return 'orange'
    elif score == 2:
        return 'red'
    else:
        return 'darkred'


for _, row in ames.iterrows():
    score = row['score_heating']  # ← 여기에 본인의 점수 컬럼 이름 사용
    color = get_color(score)

    popup_text = f"""<b>Price:</b> ${row['SalePrice']:,}<br>
                     <b>냉난방 당 점수:</b> {score}점"""
    
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        tooltip='클릭해서 정보 보기',
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(icon='home', color=color)
    ).add_to(m_heat)

m_heat


'''
# 차고 차량 수용 능력 점수 매기기 'GarageCars'
'''
# 1인 가구는 없거나 0 ~ 1대면 5점, 2인 가구는 1~2대, 가족은 1~2대 
# 그외 벗어난 경우는 차등 점수
sns.barplot(ames['GarageCars'].value_counts())

def garage_score_single(x):
    if x in [0, 1]:
        return 5
    elif x == 2:
        return 4
    elif x == 3:
        return 3
    elif x >= 4:
        return 2
    else:
        return 1

def garage_score_couple(x):
    if x in [1, 2]:
        return 5
    elif x == 3:
        return 4
    elif x >= 4:
        return 3
    elif x == 0:
        return 2
    else:
        return 1

def garage_score_family(x):
    if x in [1,2]:
        return 5
    elif x == 3:
        return 4
    elif x >= 4:
        return 3
    elif x == 0:
        return 2
    else:
        return 1

# 점수 부여
ames['GarageScore_single'] = ames['GarageCars'].apply(garage_score_single)
ames['GarageScore_couple'] = ames['GarageCars'].apply(garage_score_couple)
ames['GarageScore_family'] = ames['GarageCars'].apply(garage_score_family)

'''
# 방 개수 'TotRmsAbvGrd'
'''

# 방 개수와 면적의 상관관계
# sns.boxplot(x=ames['TotRmsAbvGrd'], y=ames['GrLivArea'])

# sns.barplot(ames['TotRmsAbvGrd'].value_counts())

# 욕실 수 4개짜리 방에 집도 욕실 2개 가진 경우 있음 
# ames[ames['FullBath']==2]['TotRmsAbvGrd'].value_counts()

# ames['TotRmsAbvGrd'].value_counts()

def room_score_single(x):
    if x in [3,4]:
        return 5
    elif x == 5:
        return 4
    elif x == 6:
        return 3
    elif x == 7:
        return 2
    else:
        return 1

def room_score_couple(x):
    if x in [5,6]:
        return 5
    elif x in [4,7]:
        return 4
    elif x in [3,8]:
        return 3
    elif x == 9:
        return 2
    else:
        return 1

def room_score_family(x):
    if x in [7,8]:
        return 5
    elif x in [6,9]:
        return 4
    elif x in [5,10]:
        return 3
    elif x in [4,11]:
        return 2
    else:
        return 1

# 점수 부여
ames['RoomScore_single'] = ames['TotRmsAbvGrd'].apply(room_score_single)
ames['RoomScore_couple'] = ames['TotRmsAbvGrd'].apply(room_score_couple)
ames['RoomScore_family'] = ames['TotRmsAbvGrd'].apply(room_score_family)



'''
# 전체 점수 산출해서 시각화 
'''

# 1인 가구 점수
ames['TotalScore_single'] = (ames['Year_Score']
                             + ames['score_area_single']
                             + ames['Cond_Score']
                             + ames['score_heating']
                             + ames['GarageScore_single'] 
                             + ames['RoomScore_single'])/6

# 가중치를 더 준 1인 가구 점수 
ames['TotalScore_single1'] = (
    ames['Year_Score'] +
    2 * ames['score_area_single'] +
    ames['Cond_Score'] +
    ames['score_heating'] +
    ames['GarageScore_single'] +
    ames['RoomScore_single']
) / 7

# 2인 가구 점수 

ames['TotalScore_couple'] = (ames['Year_Score'] 
                             + ames['score_area_couple']
                             + ames['Cond_Score']
                             + ames['score_heating']
                             + ames['GarageScore_couple'] 
                             + ames['RoomScore_couple'])/6

# 가중치를 더 준 1인 가구 점수 
ames['TotalScore_couple1'] = (
    ames['Year_Score'] +
    2 * ames['score_area_couple'] +
    ames['Cond_Score'] +
    ames['score_heating'] +
    ames['GarageScore_couple'] +
    ames['RoomScore_couple']
) / 7


# 자녀 있는 가족 점수 

ames['TotalScore_family'] = (+ ames['Year_Score']
                             + ames['score_area_family']
                             + ames['Cond_Score']
                             + ames['score_heating']
                             + ames['GarageScore_family'] 
                             + ames['RoomScore_family'])/6

ames['TotalScore_family1'] = (
    ames['Year_Score'] +
    2 * ames['score_area_family'] +
    ames['Cond_Score'] +
    ames['score_heating'] +
    ames['GarageScore_family'] +
    ames['RoomScore_family']
) / 7

ames['TotalScore_single']
ames['TotalScore_couple']
ames['TotalScore_family']

'''
# 종합 점수 추출한 집 지도 시각화
'''

def get_color(score):
    if score > 4:
        return 'darkgreen'
    elif score > 3:
        return 'green'
    elif score > 2:
        return 'orange'
    elif score > 1:
        return 'red'
    else:
        return 'darkred'


# 
map_total1 = folium.Map(location=[ames['Latitude'].mean(),
                            ames['Longitude'].mean()], 
                            zoom_start=12,
                            tiles="cartodbpositron",
                            )

for _, row in ames.iterrows():
    score = row['TotalScore_single1']  # ← 여기에 본인의 점수 컬럼 이름 사용
    # ames['TotalScore_single1']
    color = get_color(score)

    popup_text = f"""<b>위도:</b> {row['Latitude']:,}<br>
                     <b>경도:</b> {row['Longitude']:,}<br>
                     <b>Price:</b> ${row['SalePrice']:,}<br>
                     <b>면적 점수:</b> {row['score_area_single']:,}점<br>
                     <b>건축 연도 점수:</b> {row['Year_Score']:,}점<br>
                     <b>냉난방 점수:</b> {row['score_heating']:,}점<br>
                     <b>차고 수용 능력 점수:</b> {row['GarageScore_single']:,}점<br>
                     <b>방 개수 점수:</b> {row['RoomScore_single']:,}점<br>
                     <b>종합 점수:</b> {score:.2f}점"""

    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        tooltip='클릭해서 정보 보기',
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(icon='home', color=color)
    ).add_to(map_total1)
    folium.TileLayer("Stamen Watercolor").add_to(m)

map_total1


'''
# 지도 시각화에서 팝업에 사진 넣고 싶을 때 

import base64

pic = base64.b64encode(open('/content/drive/MyDrive/시각화/그림1.png', 'rb').read()).decode()
image_tag = '<img src="data:image/jpeg;base64,{}">'.format(pic)
iframe = folium.IFrame(image_tag, width=150, height=150)
popup = folium.Popup(iframe, max_width=450)
'''


'''
# 1인 가구 top9 가성비 집 시각화  # single1은 면적 당 집 값 가중치 더 준거
'''
ames_single_top9_index = ames['TotalScore_single1'].sort_values(ascending=False)[:9].index
ames_single_top9 = ames.loc[ames_single_top9_index]

# sns.histplot(ames['TotalScore_single'],bins=10)

map_single_top9 = folium.Map(location=[ames_single_top9['Latitude'].mean(),
                            ames_single_top9['Longitude'].mean()], 
                            zoom_start=12,
                            tiles='OpenStreetMap',
                            #tiles="cartodbpositron",
                            )

for _, row in ames_single_top9.iterrows():
    score = row['TotalScore_single']  # ← 여기에 본인의 점수 컬럼 이름 사용
    color = get_color(score)

    popup_text = f"""<b>위도:</b> {row['Latitude']:,}<br>
                     <b>경도:</b> {row['Longitude']:,}<br>
                     <b>Price:</b> ${row['SalePrice']:,}<br>
                     <b>면적 점수:</b> {row['score_area_single']:,}점<br>
                     <b>건축 연도 점수:</b> {row['Year_Score']:,}점<br>
                     <b>냉난방 점수:</b> {row['score_heating']:,}점<br>
                     <b>차고 수용 능력 점수:</b> {row['GarageScore_single']:,}점<br>
                     <b>방 개수 점수:</b> {row['RoomScore_single']:,}점<br>
                     <b>종합 점수:</b> {score:.2f}점"""

    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        tooltip='클릭해서 정보 보기',
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(icon='home', color=color)
    ).add_to(map_single_top9)
    # folium.TileLayer("Stamen Watercolor").add_to(map_top12)

map_single_top9



'''
# 2인 가구 top14 가성비 집 시각화  # couple1은 면적 당 집 값 가중치 더 준거
'''
ames_couple_top9_index = ames['TotalScore_couple1'].sort_values(ascending=False)[:14].index
ames_couple_top9 = ames.loc[ames_couple_top9_index]

# sns.histplot(ames['TotalScore_single'],bins=10)

map_couple_top9 = folium.Map(location=[ames_couple_top9['Latitude'].mean(),
                            ames_couple_top9['Longitude'].mean()], 
                            zoom_start=12,
                            tiles='OpenStreetMap',
                            #tiles="cartodbpositron",
                            )

for _, row in ames_couple_top9.iterrows():
    score = row['TotalScore_couple1']  # ← 여기에 본인의 점수 컬럼 이름 사용
    color = get_color(score)

    popup_text = f"""<b>위도:</b> {row['Latitude']:,}<br>
                     <b>경도:</b> {row['Longitude']:,}<br>
                     <b>Price:</b> ${row['SalePrice']:,}<br>
                     <b>면적 점수:</b> {row['score_area_single']:,}점<br>
                     <b>건축 연도 점수:</b> {row['Year_Score']:,}점<br>
                     <b>냉난방 점수:</b> {row['score_heating']:,}점<br>
                     <b>차고 수용 능력 점수:</b> {row['GarageScore_single']:,}점<br>
                     <b>방 개수 점수:</b> {row['RoomScore_single']:,}점<br>
                     <b>종합 점수:</b> {score:.2f}점"""

    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        tooltip='클릭해서 정보 보기',
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(icon='home', color=color)
    ).add_to(map_couple_top9)
    # folium.TileLayer("Stamen Watercolor").add_to(map_top12)

map_couple_top9