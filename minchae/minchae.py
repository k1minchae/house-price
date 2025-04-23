import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 한글 깨짐 방지
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
load_df = pd.read_csv('../kaggle/ames.csv')
load_df.isna().sum()[load_df.isna().sum() > 0]

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