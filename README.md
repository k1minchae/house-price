# Ames House Price

**LS 빅데이터 스쿨 4기 팀 프로젝트**

Ames House Price Data를 활용하여 대시보드를 제작하는 팀 프로젝트입니다.

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
      역할2<br />
      역할3
    </td>
    <td align="center">
      <a href="https://github.com/soohyhy">
        <img src="https://github.com/soohyhy.png" width="100px;" alt="soohyhy"/>
        <br />
        <sub><b>박수현</b></sub>
      </a>
      <br />
      집값 요인 분석<br />
      역할2<br />
      역할3
    </td>
    <td align="center">
      <a href="https://github.com/ParkHansl">
        <img src="https://github.com/ParkHansl.png" width="100px;" alt="ParkHansl"/>
        <br />
        <sub><b>박한슬</b></sub>
      </a>
      <br />
      <strong>발표</strong><br />
      동네별 선호하는 주택 분석<br />
      역할3
    </td>
    <td align="center">
      <a href="https://github.com/leechanghyuk">
        <img src="https://github.com/leechanghyuk.png" width="100px;" alt="leechanghyuk"/>
        <br />
        <sub><b>이창혁</b></sub>
      </a>
      <br />
      <strong>조장/발표</strong><br />
      가성비 좋은 주택 분석<br />
      역할3
    </td>
  </tr>
</table>

<br />

## 프로젝트 요구사항

**데이터 분석 및 시각화 대시보드를 제작합니다.**

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

Ames 부동산 인수인계용 대시보드 제작

<br />

### 프로젝트 배경

중소형 부동산 사무소에서는 보통 직원들이 그동안 쌓은 경험을 바탕으로 손님에게 집을 추천하거나 가격 상담을 해줍니다. 그런데 숫자와 자료, 즉 '데이터'를 함께 사용하면 더 정확하고 믿을 수 있는 상담이 가능합니다.

그래서 이번에는 새로 온 인턴이 부동산 일을 빠르게 배우고 잘 적응할 수 있도록 Ames Housing이라는 집 관련 데이터를 활용해, 중요한 내용을 보기 쉽게 정리한 대시보드를 만들려고 합니다.

이 대시보드는 손님이 왔을 때 어떤 집을 찾는지 쉽게 파악하고, 그에 맞는 매물을 빠르고 정확하게 추천할 수 있도록 도와주는 인턴용 안내서 역할도 함께 하도록 설계했습니다.

<br />

### 프로젝트 목표

• 경험 의존 대신 데이터로 신뢰도 강화

• 인턴용 Ames Housing 대시보드 제작

• 손님 대응 매뉴얼도 겸함

<br />

### 세부 사항

## 1. 가성비 좋은 주택의 조건은 무엇인가?

**🧩 Story**

> "예산은 한정되어 있다. 같은 돈으로 더 나은 조건의 집을 사기 위해선 어떤 기준을 봐야 할까?"

**🎯 주제**

- 같은 가격대에서 면적, 위치, 연식 등을 고려했을 때, 어떤 조합이 '가성비'가 좋은 집인가를 분석

**🔍 방법**

- `SalePrice` 대비 `GrLivArea`, `OverallQual`, `YearBuilt` 등의 비율 파생변수 생성
- 클러스터링을 통해 "가성비 그룹"을 나누고 조건 분석

---

## 2. 리모델링이 집값에 미치는 영향

**🧩 Story**

> "집을 산 뒤 리모델링하면 과연 그만한 가치가 있을까?"

**🎯 주제**

- `YearRemodAdd`와 `YearBuilt` 간 차이, 리모델링 유무에 따른 가격 차이 분석

**🔍 방법**

- 리모델링 여부 (`YearRemodAdd != YearBuilt`)에 따라 그룹 나누기
- 동일한 조건(`Area`, `Quality`)일 때 가격 차이 분석
- 회귀모형에 interaction term 추가 (`Remod × Quality`)

---

## 3. 집값에 가장 영향을 많이 주는 요인은 무엇인가?

**🧩 Story**

> "집을 팔거나 살 때, 어떤 요소가 가격을 좌우하는 가장 핵심 요인일까?"

**🎯 주제**

- 회귀 모델 또는 Lasso, Tree 기반 모델로 주요 변수 탐색

**🔍 방법**

- AIC/forward selection 또는 Lasso로 변수 선택
- SHAP 또는 permutation importance로 변수 중요도 해석
- 결과를 바탕으로 “Top 5 영향 요인” 도출

---

## 4. 동네에 따라 선호되는 집의 스타일이 다를까?

**🧩 Story**

> "같은 평수여도, 어떤 동네는 현대적인 스타일을 더 선호하고, 어떤 동네는 전통적인 구조를 더 선호하지 않을까?"

**🎯 주제**

- `Neighborhood`별로 주택 스타일(`HouseStyle`, `RoofStyle`, `Exterior`)과 가격 사이의 관계 분석

**🔍 방법**

- `Neighborhood`별 주택 스타일 비율 시각화
- 특정 스타일이 가격 상승에 기여하는지 회귀 분석
- 지도 시각화로 인사이트 전달

---

## 5. 아이를 키우기 좋은 집은 어떤 조건일까?

**🧩 Story**

> "아이를 키우기 위한 주택 조건을 찾는 부모님을 위한 분석"

**🎯 주제**

- `TotRmsAbvGrd`, `BedroomAbvGr`, `GarageCars`, `YrSold` 등을 종합하여 ‘가족친화적’ 주택 조건 정의

**🔍 방법**

- 가족친화 점수 지표 생성
- 해당 점수와 가격 간 관계 분석
- “가족친화도 높은데 가격은 합리적인” 동네 추천
- 마케팅이나 정책 제안으로 확장 가능

---

## 6. 시장 타이밍: 언제 집을 사면 가장 유리할까?

**🧩 Story**

> "주택 시장에도 타이밍이 있다면? 월별 또는 연도별 가격 추이를 보고 전략을 짜보자"

**🎯 주제**

- `MoSold`, `YrSold` 기반으로 계절성과 시장 흐름 분석

**🔍 방법**

- 월별 평균 가격 시계열 분석
- 연도별 경기 흐름 반영 → 경기침체기 or 호황기 도출
- “12월엔 저렴하게 살 수 있다” 같은 실용적 인사이트
