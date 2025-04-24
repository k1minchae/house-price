# 데이터 전처리하는 코드
import pandas as pd

# 데이터 불러오기
load_df = pd.read_csv('../kaggle/ames.csv')
print(load_df.info())
load_df.info(max_cols=10)


# Nan 이 없음 을 나타내는 컬럼들
nan_columns = ['BsmtCond', 'BsmtFinType2', 'BsmtExposure',
                  'BsmtFinType1', 'FireplaceQu', 'GarageType', 
                  'GarageFinish', 'GarageQual', 'GarageCond',
                  'PoolQC', 'MasVnrType', 'Fence', 'MiscFeature',
                    'Alley', 'BsmtQual']

load_df['LotFrontage'].fillna(0, inplace=True)
load_df['GarageYrBlt'].fillna(0, inplace=True)

load_df[nan_columns] = load_df[nan_columns].fillna("없음")


# 결측치 재확인
load_df.isna().sum()[load_df.isna().sum() > 0]

'''
MasVnrArea      14
BsmtFinSF1       1
BsmtFinSF2       1
BsmtUnfSF        1
TotalBsmtSF      1
Electrical       1
BsmtFullBath     2
BsmtHalfBath     2
GarageCars       1
GarageArea       1
GeoRefNo        20
Prop_Addr       20
Latitude        97
Longitude       97
'''


# 주소는 있는데 위도 경도가 없는 경우
filtered = load_df.loc[(load_df['Latitude'].isna()) & (~load_df['Prop_Addr'].isna()), :]

# 주소도 없는 경우: 20개
len(load_df.loc[(load_df['Latitude'].isna()) & (load_df['Prop_Addr'].isna()), :])


from geopy.geocoders import Nominatim
import time
geolocator = Nominatim(user_agent="ames_geocoder")

# 위도/경도 컬럼 생성
filtered['위도_보정'] = None
filtered['경도_보정'] = None

# 주소 기반 위도/경도 검색
find = 0
not_find = 0
for idx, row in filtered.iterrows():
    try:
        # 주소 + 도시 이름으로 검색 정확도 향상
        full_address = f"{row['Prop_Addr']}, Ames, Iowa"
        location = geolocator.geocode(full_address)

        # 주소 찾은 경우
        if location:
            filtered.at[idx, '위도_보정'] = location.latitude
            filtered.at[idx, '경도_보정'] = location.longitude
            print(f"[O] 주소 찾음: {full_address} → 위도: {location.latitude}, 경도: {location.longitude}")
            find += 1

        # 주소 못 찾은 경우
        else:
            print(f"[X] 주소 찾을 수 없음: {full_address}")
            not_find += 1

    except Exception as e:
        print(f"[!] 오류 발생 at {row['주소']} → {e}")
        not_find += 1
    time.sleep(1)  # API 과부하 방지

# 기존 df에 위도/경도 덮어쓰기
load_df.loc[filtered.index, 'Latitude'] = filtered['위도_보정']
load_df.loc[filtered.index, 'Longitude'] = filtered['경도_보정']

# 주소 찾은 개수: 42, 찾지 못한 개수: 35
print(f"주소 찾은 개수: {find}, 찾지 못한 개수: {not_find}")    

load_df.isna().sum()[load_df.isna().sum() > 0]
'''
MasVnrArea      14
BsmtFinSF1       1
BsmtFinSF2       1
BsmtUnfSF        1
TotalBsmtSF      1
Electrical       1
BsmtFullBath     2
BsmtHalfBath     2
GarageCars       1
GarageArea       1
GeoRefNo        20
Prop_Addr       20
Latitude        55
Longitude       55

'''

# 처리한 데이터 저장
csv_path = "../ames_cleaned.csv"
load_df = load_df.dropna()
load_df.to_csv(csv_path, index=False)

df = pd.read_csv(csv_path)
len(df.loc[df['SaleCondition'] == 'Partial'])
len(df.loc[df['SaleCondition'] >= 3])
df.loc[df['FullBath'] > 2]
sorted(df['SaleCondition'].unique())
df['TotalArea']


# 난방 품질
import matplotlib.pyplot as plt

# 원하는 순서대로 등급 정의
order = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
labels = ['매우우수', '좋음', '보통', '나쁨', '매우나쁨', '없음']

# NA 값 대체
df['HeatingQC'] = df['HeatingQC'].fillna('NA')

# 값 개수 정렬
counts = df['HeatingQC'].value_counts().reindex(order, fill_value=0)

# 시각화
plt.figure(figsize=(8, 5))
plt.bar(labels, counts[order], color='skyblue')
plt.xlabel('난방 품질')
plt.ylabel('건수')
plt.title('난방 품질 분포 (내림차순)')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


df['RoofMatl'].unique()
plt.hist(df['RoofMatl'], bins=20, color='skyblue')
plt.xlabel('지붕 자재')
plt.ylabel('건수')
plt.title('지붕 자재 분포')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

df['TotalArea']




df_cleaned = pd.read_csv("../soohy/df_cleaned.csv")



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
fig = fig.add_trace(
    go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(color='steelblue', size=8, line=dict(width=1, color='DarkSlateGrey')),
        name='Data'
    )
)

# 회귀선
fig = fig.add_trace(
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
                 'title': f'{feature} vs SalePrice'}
            ],
        )
    )

# 레이아웃 설정
fig = fig.update_layout(
    title=f'{x_var} vs SalePrice',
    template='plotly_white',
    width=1000,
    height=600,
    xaxis_title=x_var,
    yaxis_title='SalePrice',
    updatemenus=[
        dict(
            buttons=buttons,
            direction='down',
            showactive=True,
            x=0.5,
            xanchor='center',
            y=1.15,
            yanchor='top'
        )
    ]
)

fig.show()