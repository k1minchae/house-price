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
