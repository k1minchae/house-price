# Ames House Price

**LS 빅데이터 스쿨 4기 팀 프로젝트**

Ames House Price Data를 활용하여 대시보드를 제작하는 팀 프로젝트입니다.

[>프로젝트 대시보드 바로가기 링크]("https://k1minchae.github.io/house-price/")

<br>

## 팀원 소개

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/k1minchae">
        <img src="https://github.com/k1minchae.png" width="100px;" alt="k1minchae"/>
        <br />
        <sub><b>김민채</b></sub>
      </a>
      <br />
      리모델링 영향 분석<br />
      Qmd/Git 관리
    </td>
    <td align="center">
      <a href="https://github.com/soohyhy">
        <img src="https://github.com/soohyhy.png" width="100px;" alt="soohyhy"/>
        <br />
        <sub><b>박수현</b></sub>
      </a>
      <br />
      집값 영향 요인 분석<br />
      회귀분석 진행
    </td>
    <td align="center">
      <a href="https://github.com/ParkHansl">
        <img src="https://github.com/ParkHansl.png" width="100px;" alt="ParkHansl"/>
        <br />
        <sub><b>박한슬</b></sub>
      </a>
      <br />
      동네별 선호하는 주택 분석<br />
      발표
    </td>
    <td align="center">
      <a href="https://github.com/leechanghyuk">
        <img src="https://github.com/leechanghyuk.png" width="100px;" alt="leechanghyuk"/>
        <br />
        <sub><b>이창혁</b></sub>
      </a>
      <br />
      가성비 좋은 주택 분석<br />
      조장/발표<br />
    </td>
  </tr>
</table>

<br />

## 데이터 설명

| 컬럼명 (영문) | 설명 (한글)                             |
| ------------- | --------------------------------------- |
| SalePrice     | 주택 판매 가격 (예측 대상)              |
| MSSubClass    | 건물 클래스                             |
| MSZoning      | 일반 용도 지역 분류                     |
| LotFrontage   | 도로와 접한 길이 (feet)                 |
| LotArea       | 대지 면적 (제곱피트)                    |
| Street        | 도로 접근 유형                          |
| Alley         | 골목 접근 유형 (NA: 없음)               |
| LotShape      | 대지 형태                               |
| LandContour   | 대지의 평탄도                           |
| Utilities     | 사용 가능한 공공설비 종류               |
| LotConfig     | 대지 구성 방식                          |
| LandSlope     | 대지 경사도                             |
| Neighborhood  | Ames 시 내의 물리적 위치                |
| Condition1    | 주요 도로 또는 철도와의 근접도          |
| Condition2    | 추가적인 주요 도로 또는 철도와의 근접도 |
| BldgType      | 주택 유형                               |
| HouseStyle    | 주택 스타일                             |
| OverallQual   | 전체 자재 및 마감 품질                  |
| OverallCond   | 전체 상태 평가                          |
| YearBuilt     | 최초 건축 연도                          |
| YearRemodAdd  | 리모델링 연도                           |
| RoofStyle     | 지붕 스타일                             |
| RoofMatl      | 지붕 재질                               |
| Exterior1st   | 외벽 마감재 (첫 번째)                   |
| Exterior2nd   | 외벽 마감재 (두 번째)                   |
| MasVnrType    | 벽돌 베니어 유형 (NA: 없음)             |
| MasVnrArea    | 벽돌 베니어 면적 (제곱피트)             |
| ExterQual     | 외벽 마감재 품질                        |
| ExterCond     | 외벽 현재 상태                          |
| Foundation    | 기초 형태                               |
| BsmtQual      | 지하실 높이 (NA: 없음)                  |
| BsmtCond      | 지하실 전반 상태 (NA: 없음)             |
| BsmtExposure  | 지하실 외부 노출 여부 (NA: 없음)        |
| BsmtFinType1  | 지하실 마감 공간 유형 1 (NA: 없음)      |
| BsmtFinSF1    | 마감된 지하 공간 면적 1                 |
| BsmtFinType2  | 지하실 마감 공간 유형 2 (NA: 없음)      |
| BsmtFinSF2    | 마감된 지하 공간 면적 2                 |
| BsmtUnfSF     | 미마감 지하 공간 면적                   |
| TotalBsmtSF   | 전체 지하 공간 면적                     |
| Heating       | 난방 방식                               |
| HeatingQC     | 난방 품질 및 상태                       |
| CentralAir    | 중앙 냉방 장치 유무                     |
| Electrical    | 전기 시스템 종류                        |
| 1stFlrSF      | 1층 면적                                |
| 2ndFlrSF      | 2층 면적                                |
| LowQualFinSF  | 낮은 품질 마감 면적                     |
| GrLivArea     | 지상 생활 면적 (제곱피트)               |
| BsmtFullBath  | 지하 전체 욕실 수                       |
| BsmtHalfBath  | 지하 반 욕실 수                         |
| FullBath      | 지상 전체 욕실 수                       |
| HalfBath      | 지상 반 욕실 수                         |
| Bedroom       | 침실 수 (지하 제외)                     |
| Kitchen       | 주방 수                                 |
| KitchenQual   | 주방 품질                               |
| TotRmsAbvGrd  | 총 방 수 (욕실 제외, 지상 기준)         |
| Functional    | 주택 기능성 등급                        |
| Fireplaces    | 벽난로 수                               |
| FireplaceQu   | 벽난로 품질 (NA: 없음)                  |
| GarageType    | 차고 위치 (NA: 없음)                    |
| GarageYrBlt   | 차고 건축 연도                          |
| GarageFinish  | 차고 내부 마감 상태 (NA: 없음)          |
| GarageCars    | 차고 수용 차량 수                       |
| GarageArea    | 차고 면적                               |
| GarageQual    | 차고 품질 (NA: 없음)                    |
| GarageCond    | 차고 상태 (NA: 없음)                    |
| PavedDrive    | 포장 진입로 여부                        |
| WoodDeckSF    | 목재 데크 면적                          |
| OpenPorchSF   | 개방형 현관 면적                        |
| EnclosedPorch | 폐쇄형 현관 면적                        |
| 3SsnPorch     | 3계절용 현관 면적                       |
| ScreenPorch   | 방충망 있는 현관 면적                   |
| PoolArea      | 수영장 면적                             |
| PoolQC        | 수영장 품질 (NA: 없음)                  |
| Fence         | 울타리 품질 (NA: 없음)                  |
| MiscFeature   | 기타 부대시설 (NA: 없음)                |
| MiscVal       | 기타 부대시설의 금전적 가치             |
| MoSold        | 판매 월                                 |
| YrSold        | 판매 연도                               |
| SaleType      | 판매 유형                               |
| SaleCondition | 판매 조건                               |

<br />

## 프로젝트 요구사항

**활용 데이터**: Ames 데이터(lon, lat 버전)

1. 데이터 탐색(EDA) 및 전처리 결과 시각화

   - 주요 변수 분포, 결측치 처리, 이상치 탐지 등

2. 지도 기반 시각화
   - 예: Folium, Plotly 등 사용 가능
3. 인터랙티브 요소

   - 예: Plotly 등

4. 모델 학습 페이지
   - 회귀모델 훈련 과정과 결과 시각화
   - 페널티 회귀 모델 필수 사용
5. 스토리텔링 구성
   - 전체 대시보드가 하나의 분석 흐름으로 자연스럽게 이어질 것
   - 꼭 집값 예측이 아니어도 됨!
6. 전체 분량
   - 4-5페이지로 구성

<br />

## 프로젝트 개요

### 프로젝트 명

**“부동산 사무소 신입을 위한 부동산 인사이트 대시보드”**

<br />

### 프로젝트 배경

우리는 중소형 부동산 사무소를 운영하고 있습니다. 신입사원이 입사할 때마다 반복적으로 업무 설명을 해줘야 하는 상황입니다.

신입사원이 부동산 업무 전반을 빠르게 이해하고 실제 상담에 활용할 수 있도록 돕는 **교육용 대시보드**를 제작하기로 했습니다.

<br />

### 프로젝트 목표

• 경험 의존 대신 데이터로 신뢰도 강화

• 인턴용 Ames Housing 대시보드 제작

• 손님 대응 매뉴얼도 겸함

<br />

### 세부 사항

### 1. 가성비 좋은 주택의 조건은 무엇인가?

**🧩 Story**

> "예산은 한정되어 있다. 같은 돈으로 더 나은 조건의 집을 사기 위해선 어떤 기준을 봐야 할까?"

**🎯 주제**

- 손님 유형에 따라, 어떤 주택 조합이 '가성비'가 좋은 집인가를 분석

**🔍 방법**

- 1인가구, 2인가구, 자녀있는 가구로 구분
- 각각의 가구 유형에 맞는 주택 조합을 도출하기 위해, 여러 변수를 종합하여 점수 계산 후 Top 10 집 선정

---

### 2. 리모델링이 집값에 미치는 영향

**🧩 Story**

> "집을 산 뒤 리모델링하면 과연 그만한 가치가 있을까?"

**🎯 주제**

- `YearRemodAdd`와 `YearBuilt` 간 차이, 리모델링 유무에 따른 가격 차이 분석

**🔍 방법**

- 리모델링 여부 (`YearRemodAdd != YearBuilt`)에 따라 그룹 나누기
- 각종 검정/시각화 방법을 통해 리모델링 유무에 따른 가격 차이 분석

---

### 3. 집값에 가장 영향을 많이 주는 요인은 무엇인가?

**🧩 Story**

> "집을 팔거나 살 때, 어떤 요소가 가격을 좌우하는 가장 핵심 요인일까?"

**🎯 주제**

- 회귀 모델 또는 Lasso, Ridge 기반 Elastic Net 모델로 주요 변수 탐색

**🔍 방법**

- 상관관계를 기반으로 변수선택 + Elastic Net으로 변수 선택
- 결과를 바탕으로 “Top 영향 요인” 도출

---

### 4. 동네에 따라 선호되는 집의 스타일이 다를까?

**🧩 Story**

> "같은 평수여도, 어떤 동네는 현대적인 스타일을 더 선호하고, 어떤 동네는 전통적인 구조를 더 선호하지 않을까?"

**🎯 주제**

- `Neighborhood`별로 주택 스타일(`HouseStyle`, `RoofStyle`, `Exterior`)과 가격 사이의 관계 분석

**🔍 방법**

- `Neighborhood`별 주택 스타일 시각화
- 지도 시각화로 인사이트 전달
