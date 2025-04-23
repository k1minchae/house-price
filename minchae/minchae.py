import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 한글 깨짐 방지
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
load_df = pd.read_csv('../kaggle/ames.csv')
load_df.isna().sum()[load_df.isna().sum() > 0]

# 컬럼명 한국어로 변경
column_dict = {
    "SalePrice": "주택 판매 가격 (예측 대상)",
    "MSSubClass": "건물 클래스",
    "MSZoning": "일반 용도 지역 분류",
    "LotFrontage": "도로와 접한 길이 (feet)",
    "LotArea": "대지 면적 (제곱피트)",
    "Street": "도로 접근 유형",
    "Alley": "골목 접근 유형",
    "LotShape": "대지 형태",
    "LandContour": "대지의 평탄도",
    "Utilities": "사용 가능한 공공설비 종류",
    "LotConfig": "대지 구성 방식",
    "LandSlope": "대지 경사도",
    "Neighborhood": "Ames 시 내의 물리적 위치",
    "Condition1": "주요 도로 또는 철도와의 근접도",
    "Condition2": "추가적인 주요 도로 또는 철도와의 근접도",
    "BldgType": "주택 유형",
    "HouseStyle": "주택 스타일",
    "OverallQual": "전체 자재 및 마감 품질",
    "OverallCond": "전체 상태 평가",
    "YearBuilt": "최초 건축 연도",
    "YearRemodAdd": "리모델링 연도",
    "RoofStyle": "지붕 스타일",
    "RoofMatl": "지붕 재질",
    "Exterior1st": "외벽 마감재 (첫 번째)",
    "Exterior2nd": "외벽 마감재 (두 번째)",
    "MasVnrType": "벽돌 베니어 유형",
    "MasVnrArea": "벽돌 베니어 면적 (제곱피트)",
    "ExterQual": "외벽 마감재 품질",
    "ExterCond": "외벽 현재 상태",
    "Foundation": "기초 형태",
    "BsmtQual": "지하실 높이",
    "BsmtCond": "지하실 전반 상태",
    "BsmtExposure": "지하실 외부 노출 여부",
    "BsmtFinType1": "지하실 마감 공간 유형 1",
    "BsmtFinSF1": "마감된 지하 공간 면적 1",
    "BsmtFinType2": "지하실 마감 공간 유형 2",
    "BsmtFinSF2": "마감된 지하 공간 면적 2",
    "BsmtUnfSF": "미마감 지하 공간 면적",
    "TotalBsmtSF": "전체 지하 공간 면적",
    "Heating": "난방 방식",
    "HeatingQC": "난방 품질 및 상태",
    "CentralAir": "중앙 냉방 장치 유무",
    "Electrical": "전기 시스템 종류",
    "1stFlrSF": "1층 면적",
    "2ndFlrSF": "2층 면적",
    "LowQualFinSF": "낮은 품질 마감 면적",
    "GrLivArea": "지상 생활 면적 (제곱피트)",
    "BsmtFullBath": "지하 전체 욕실 수",
    "BsmtHalfBath": "지하 반 욕실 수",
    "FullBath": "지상 전체 욕실 수",
    "HalfBath": "지상 반 욕실 수",
    "Bedroom": "침실 수 (지하 제외)",
    "Kitchen": "주방 수",
    "KitchenQual": "주방 품질",
    "TotRmsAbvGrd": "총 방 수 (욕실 제외, 지상 기준)",
    "Functional": "주택 기능성 등급",
    "Fireplaces": "벽난로 수",
    "FireplaceQu": "벽난로 품질",
    "GarageType": "차고 위치",
    "GarageYrBlt": "차고 건축 연도",
    "GarageFinish": "차고 내부 마감 상태",
    "GarageCars": "차고 수용 차량 수",
    "GarageArea": "차고 면적",
    "GarageQual": "차고 품질",
    "GarageCond": "차고 상태",
    "PavedDrive": "포장 진입로 여부",
    "WoodDeckSF": "목재 데크 면적",
    "OpenPorchSF": "개방형 현관 면적",
    "EnclosedPorch": "폐쇄형 현관 면적",
    "3SsnPorch": "3계절용 현관 면적",
    "ScreenPorch": "방충망 있는 현관 면적",
    "PoolArea": "수영장 면적",
    "PoolQC": "수영장 품질",
    "Fence": "울타리 품질",
    "MiscFeature": "기타 부대시설",
    "MiscVal": "기타 부대시설의 금전적 가치",
    "MoSold": "판매 월",
    "YrSold": "판매 연도",
    "SaleType": "판매 유형",
    "SaleCondition": "판매 조건",
    "PID": "고유 식별 번호",
    "BedroomAbvGr": "지상 침실 수",
    "KitchenAbvGr": "지상 주방 수",
    "GeoRefNo": "지리 참조 번호",
    "Prop_Addr": "주소",
    "Latitude": "위도",
    "Longitude": "경도"
}

df_renamed = load_df.rename(columns=column_dict)
df_renamed.columns
df_renamed


csv_path = "../ames_kor.csv"
df_renamed.to_csv(csv_path, index=False)
csv_path

df = df_renamed
df.isna().sum()[df.isna().sum() > 0]

load_df.isna().sum()[load_df.isna().sum() > 0]

# Nan 이 없음 을 나타내는 컬럼들
nan_columns = ['지하실 전반 상태', 
               '지하실 마감 공간 유형 2', '지하실 외부 노출 여부', 
               '지하실 마감 공간 유형 1', '벽난로 품질',
               '차고 위치', '차고 내부 마감 상태',
               '차고 품질', '차고 상태', '수영장 품질', '벽돌 베니어 유형',
               '울타리 품질', '기타 부대시설', '골목 접근 유형']

# 결측치 처리
df['지하실 높이'].fillna(0, inplace=True)
df['도로와 접한 길이 (feet)'].fillna(0, inplace=True)
df['차고 건축 연도'].fillna(0, inplace=True)
df[nan_columns] = df[nan_columns].fillna("없음")

df.isna().sum()[df.isna().sum() > 0]

print(df.loc[(df['위도'].isna()) & (~(df['주소'].isna())), :])

# 주소는 있는데 위도 경도가 없는 경우
filtered = df.loc[(df['위도'].isna()) & (~df['주소'].isna()), :]

# 주소도 없는 경우: 20개
len(df.loc[(df['위도'].isna()) & (df['주소'].isna()), :])

# 주소를 찾아서 위도경도 찾기
filtered.iloc[:20, -3]
filtered.iloc[20:40, -3]
filtered.iloc[40:60, -3]
filtered.iloc[60:, -3]


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
        full_address = f"{row['주소']}, Ames, Iowa"
        location = geolocator.geocode(full_address)

        if location:
            filtered.at[idx, '위도_보정'] = location.latitude
            filtered.at[idx, '경도_보정'] = location.longitude
            print(f"[O] 주소 찾음: {full_address} → 위도: {location.latitude}, 경도: {location.longitude}")
            find += 1
        else:
            print(f"[X] 주소 찾을 수 없음: {full_address}")
            not_find += 1
    except Exception as e:
        print(f"[!] 오류 발생 at {row['주소']} → {e}")
        not_find += 1
    time.sleep(1)  # API 과부하 방지

# 기존 df에 위도/경도 덮어쓰기
df.loc[filtered.index, '위도'] = filtered['위도_보정']
df.loc[filtered.index, '경도'] = filtered['경도_보정']

# 주소 찾은 개수: 41, 찾지 못한 개수: 36
print(f"주소 찾은 개수: {find}, 찾지 못한 개수: {not_find}")    

print(df.isna().sum())
# 벽돌 베니어 면적 (제곱피트)    14
# 마감된 지하 공간 면적 1       1
# 마감된 지하 공간 면적 2       1
# 미마감 지하 공간 면적         1
# 전체 지하 공간 면적          1
# 전기 시스템 종류            1
# 지하 전체 욕실 수           2
# 지하 반 욕실 수            2
# 차고 수용 차량 수           1
# 차고 면적                1
# 지리 참조 번호            20
# 주소                  20
# 위도                  56
# 경도                  56

df = df.dropna()
df
df.to_csv(csv_path, index=False)

