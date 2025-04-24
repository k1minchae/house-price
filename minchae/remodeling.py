# 리모델링이 집값에 영향을 미치는 지 분석

# 데이터 불러오기
import pandas as pd
df = pd.read_csv('../ames_cleaned.csv')

# 결측치 제거
# 리모델링 연도가 지어진 연도보다 빠른 경우 결측치로 판단
df[df['YearRemodAdd'] < df['YearBuilt']] # 1개
df = df[df['YearRemodAdd'] >= df['YearBuilt']]

# 리모델링 안한 집 vs 한 집
# 5년 이하는 하자보수 기간으로 판단
df['diff'] = df['YearRemodAdd'] - df['YearBuilt'] # 리모델링 연도 - 지어진 연도
not_remodel = df[df['diff'] <= 5] # 1735 개
remodel = df[df['diff'] > 5] # 771 개


# 시각화
import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# 하자보수 기간
# 5년이하는 시공 하자보수 기간으로 판단
plt.figure(figsize=(5, 4))
apt_refactor = not_remodel.loc[(not_remodel['diff'] <= 5) & (not_remodel['diff'] > 0), :]
plt.hist(apt_refactor['diff'], bins=9, color='blue', alpha=0.7)
plt.title('리모델링 연도 - 지어진 연도')
plt.xlabel('리모델링 연도 - 지어진 연도')
plt.xticks([1, 2, 3, 4, 5])
plt.ylabel('빈도수')
plt.show()
# 대부분 시공 직후에 하자보수를 하는 경우가 많음



# 히스토그램 (리모델링 연도)
len_1950 = len(remodel.loc[remodel['YearRemodAdd'] == 1950])
plt.hist(remodel['YearRemodAdd'], bins=50, color='blue', alpha=0.7)
plt.title('리모델링 연도')
plt.text(1950, len_1950, f'{len_1950}', fontsize=12, color='red')
plt.xlabel('리모델링 연도')
plt.ylabel('빈도수')
plt.show()


# 히스토그램 (1950년도 제거)
not_1950 = remodel.loc[remodel['YearRemodAdd'] != 1950, :]
plt.hist(not_1950['YearRemodAdd'], bins=30, color='blue', alpha=0.7)
plt.title('리모델링 연도 (1950년도 제거)')
plt.axvline(not_1950['YearRemodAdd'].mean(), color='red', linestyle='dashed', linewidth=2, label=f"평균: {not_1950['YearRemodAdd'].mean():.0f}")
plt.axvline(not_1950['YearRemodAdd'].median(), color='green', linestyle='dashed', linewidth=2, label=f"중앙값: {not_1950['YearRemodAdd'].median():.0f}")
plt.xlabel('리모델링 연도')
plt.ylabel('빈도수')
plt.legend()
plt.show()
# 왼쪽 꼬리 분포



# 히스토그램 (리모델링 연도 - 지어진 연도)
plt.hist(remodel['diff'], bins=40, color='blue', alpha=0.7)
plt.axvline(remodel['diff'].mean(), color='red', linestyle='dashed', linewidth=2, label=f"평균: {remodel['diff'].mean():.0f}")
plt.axvline(remodel['diff'].median(), color='green', linestyle='dashed', linewidth=2, label=f"중앙값: {remodel['diff'].median():.0f}")
plt.title('리모델링 연도 - 지어진 연도')
plt.xlabel('리모델링 연도 - 지어진 연도')
plt.ylabel('빈도수')
plt.legend()
plt.show()
# 오른쪽 꼬리 분포





# 리모델링 여부에 따른 주택 판매 가격 비교 (박스플롯)
import seaborn as sns
# 두 그룹에 구분 컬럼 추가
remodel['리모델링'] = 'O'
not_remodel['리모델링'] = 'X'

# 데이터 합치기
combined = pd.concat([remodel, not_remodel])

# 이상치 보기
# 리모델링 여부 그룹 설정
grouped = combined.groupby('리모델링')

# 이상치 판별 결과 저장용 리스트
outliers_list = []

# 각 그룹별로 이상치 추출
for label, group in grouped:
    q1 = group['SalePrice'].quantile(0.25)
    q3 = group['SalePrice'].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = group[(group['SalePrice'] < lower_bound) | (group['SalePrice'] > upper_bound)]
    outliers_list.append(outliers)

# 결과 통합
outliers_df = pd.concat(outliers_list)
outliers_df.groupby('리모델링')['SalePrice'].describe()

# 이상치 개수
outliner_cnt_X = outliers_df['리모델링'].value_counts()['X']
outliner_ratio_X = (outliner_cnt_X / len(not_remodel)) * 100
outliner_cnt_O = outliers_df['리모델링'].value_counts()['O']
outliner_ratio_O = (outliner_cnt_O / len(remodel)) * 100

# 박스플롯 그리기
plt.figure(figsize=(8, 6))
sns.boxplot(data=combined, x='리모델링', y='SalePrice', palette='pastel')
plt.title("리모델링 여부에 따른 주택 판매 가격 비교")
plt.xlabel("리모델링 여부")
plt.text(0, 430000, f'이상치: {outliner_cnt_X}개 {outliner_ratio_X:.2f}%', fontsize=12, color='red')
plt.text(1, 500000, f'이상치: {outliner_cnt_O}개 {outliner_ratio_O:.2f}%', fontsize=12, color='red')
plt.ylabel("주택 가격")
plt.grid(True, alpha=0.3)
plt.show()


# 이상치 제거한 데이터프레임 생성
# 신입사원이라 비싼 집은 안맡음
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

remod_cleaned = remove_outliers_iqr(remodel, 'SalePrice')
not_remodel_cleaned = remove_outliers_iqr(not_remodel, 'SalePrice')

# 다시 하나의 데이터프레임으로 합치기
df_cleaned = pd.concat([remod_cleaned, not_remodel_cleaned])

plt.figure(figsize=(8, 6))
sns.boxplot(data=df_cleaned, x='리모델링', y='SalePrice', palette='pastel')
plt.title("리모델링 여부에 따른 주택 판매 가격 비교")
plt.xlabel("리모델링 여부")
plt.ylabel("주택 가격")
plt.grid(True, alpha=0.3)
plt.show()




# 정규성 검정
import matplotlib.pyplot as plt
import scipy.stats as stats

plt.figure(figsize=(12, 5))

# 리모델링 O
plt.subplot(1, 2, 1)
stats.probplot(remod_cleaned['SalePrice'], dist="norm", plot=plt)
plt.title("Q-Q Plot: 리모델링 O")

# 리모델링 X
plt.subplot(1, 2, 2)
stats.probplot(not_remodel_cleaned['SalePrice'], dist="norm", plot=plt)
plt.title("Q-Q Plot: 리모델링 X")

plt.tight_layout()
plt.show()

# anderson-darling 검정
from scipy.stats import anderson
result_remod = anderson(remod_cleaned['SalePrice'])
result_not_remod = anderson(not_remodel_cleaned['SalePrice'])

print("[리모델링 O - Anderson-Darling 검정 결과]")
print(f"Statistic: {result_remod.statistic:.4f}")
for sl, cv in zip(result_remod.significance_level, result_remod.critical_values):
    if result_remod.statistic < cv:
        print(f"정규성 만족 (유의수준 {sl}%)")
    else:
        print(f"정규성 불만족 (유의수준 {sl}%)")

print("[리모델링 X - Anderson-Darling 검정 결과]")
print(f"Statistic: {result_not_remod.statistic:.4f}")
for sl, cv in zip(result_not_remod.significance_level, result_not_remod.critical_values):
    if result_not_remod.statistic < cv:
        print(f"정규성 만족 (유의수준 {sl}%)")
    else:
        print(f"정규성 불만족 (유의수준 {sl}%)")


# 비모수 검정
from scipy.stats import mannwhitneyu

# 리모델링 여부에 따른 주택 가격 비교 (비모수 검정)
stat, p_value = mannwhitneyu(remod_cleaned['SalePrice'], not_remodel_cleaned['SalePrice'], alternative='two-sided')
print(f"Mann-Whitney U 검정 통계량: {stat:.4f}, p-value: {p_value:.4f}")
# Mann-Whitney U 검정 통계량: 273388.5000, p-value: 0.0000
# 두 집단의 중앙값이 다르다.



# 리모델링 한 집값이 더 낮네?
# 왜그럴까?
# 오래된 집이라서 리모델링을 많이 하는 걸까?






# 지어진 연도 비교
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_cleaned, x='리모델링', y='YearBuilt', palette='Set2')
plt.title("리모델링 여부에 따른 지어진 연도 비교")
plt.xlabel("리모델링 여부")
plt.ylabel("지어진 연도")
plt.grid(True, alpha=0.3)
plt.show()

# 지어진 연도 정규성 시각화
plt.figure(figsize=(12, 5))

# 리모델링 O
plt.subplot(1, 2, 1)
stats.probplot(remod_cleaned['YearBuilt'], dist="norm", plot=plt)
plt.title("Q-Q Plot: 리모델링 O")

# 리모델링 X
plt.subplot(1, 2, 2)
stats.probplot(not_remodel_cleaned['YearBuilt'], dist="norm", plot=plt)
plt.title("Q-Q Plot: 리모델링 X")

plt.tight_layout()
plt.show()

# 리모델링 안 한집은 명확하게 정규성을 따르지 않음.
# 리모델링 한 집은 정규성을 따를수도? 검정 ㄱㄱ


# 통계 검정 (리모델링 O)
from scipy.stats import shapiro
stat, pval = shapiro(remod_cleaned['YearBuilt'])
print(f"Shapiro-Wilk 검정 통계량: {stat:.4f}, p-value: {pval:.4f}")

result_remod_year = anderson(remod_cleaned['YearBuilt'])

print("[리모델링 O - Anderson-Darling 검정]")
print(f"Statistic: {result_remod_year.statistic:.4f}")
for sl, cv in zip(result_remod_year.significance_level, result_remod_year.critical_values):
    if result_remod_year.statistic < cv:
        print(f"정규성 만족 (유의수준 {sl}%)")
    else:
        print(f"정규성 불만족 (유의수준 {sl}%)")

# 엥 전부 불만족이네
# 리모델링 한 집도 정규성을 따르지 않음.


# 분포 그려보기
import numpy as np
from scipy.stats import norm

# 데이터 준비
data = remod_cleaned['YearBuilt']

# 정규분포를 위한 평균, 표준편차
mu, std = data.mean(), data.std(ddof=1)

# x값 범위 지정
xmin, xmax = data.min(), data.max()
x = np.linspace(xmin, xmax, 100)

# 정규분포 PDF 계산
p = norm.pdf(x, mu, std)

# 히스토그램 (밀도로 정규화)
plt.hist(data, bins=50, color='blue', alpha=0.7, density=True)

# 정규분포 곡선 추가
plt.plot(x, p, 'r', linewidth=2, label=f'정규분포 PDF')

# 라벨/타이틀
plt.title('지어진 연도')
plt.xlabel('지어진 연도')
plt.ylabel('밀도')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 분포를 그려보니까 안따르는게맞네



# 맨 휘트니 검정
from scipy.stats import mannwhitneyu

# 리모델링 여부에 따른 지어진 연도 비교 (비모수 검정)
stat, p_value = mannwhitneyu(remod_cleaned['YearBuilt'], not_remodel_cleaned['YearBuilt'], alternative='two-sided')
print(f"Mann-Whitney U 검정 통계량: {stat:.4f}, p-value: {p_value:.4f}")
# Mann-Whitney U 검정 통계량: 75080.5000, p-value: 0.0000
# 두 집단의 중앙값이 다르다.
# 오래된 집이 리모델링을 많이 한 것으로 보인다.





# 지어진 연도에 따른 가격 차이 비교
# 연도별 평균 가격 계산
yearly_avg = df_cleaned.groupby('YearBuilt')['SalePrice'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(data=yearly_avg, x='YearBuilt', y='SalePrice')
sns.lineplot()
plt.axvline(x=1950, color='red', linestyle='--', label='1950년')
plt.title('지어진 연도별 평균 주택 가격')
plt.xlabel('지어진 연도')
plt.ylabel('평균 가격')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# 1950년도 전이 변동폭이 크다.
# 그렇다면 오래된 주택은 리모델링 전후 가격차이가 있을까?



old_houses = df_cleaned.loc[df_cleaned['YearBuilt'] <= 1950, :]
old_houses.groupby('리모델링')['SalePrice'].describe()

# 이상치 보기
# 리모델링 O 그룹만 추출
remod_old = old_houses[old_houses['리모델링'] == 'O']

# 이상치 기준 계산 (IQR)
q1 = remod_old['SalePrice'].quantile(0.25)
q3 = remod_old['SalePrice'].quantile(0.75)
iqr = q3 - q1
upper_bound = q3 + 1.5 * iqr

# 이상치 추출
outliers = remod_old[remod_old['SalePrice'] > upper_bound]
n_outliers = len(outliers)
n_total = len(remod_old)
outlier_ratio = (n_outliers / n_total) * 100

# 시각화 (박스플롯)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.boxplot(data=old_houses, x='리모델링', y='SalePrice', palette='pastel')
plt.title("오래된 주택 리모델링 여부에 따른 가격 비교")
plt.text(
    x=0, y=remod_old['SalePrice'].max()*0.95, 
    s=f"이상치: {n_outliers}개 ({outlier_ratio:.2f}%)", 
    color='red', fontsize=12
)
plt.ylabel("주택 판매 가격")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()





# 정규성 검정
# 리모델링 여부 그룹 나누기
old_remod = old_houses[old_houses['리모델링'] == 'O']['SalePrice']
old_not_remod = old_houses[old_houses['리모델링'] == 'X']['SalePrice']

# Q-Q Plot
plt.figure(figsize=(12, 5))

# 리모델링 O
plt.subplot(1, 2, 1)
stats.probplot(old_remod, dist="norm", plot=plt)
plt.title("Q-Q Plot: 리모델링 O")

# 리모델링 X
plt.subplot(1, 2, 2)
stats.probplot(old_not_remod, dist="norm", plot=plt)
plt.title("Q-Q Plot: 리모델링 X")

plt.tight_layout()
plt.show()


result_remod = anderson(old_remod)

print("[오래된 주택 - 리모델링 O]")
print(f"Statistic: {result_remod.statistic:.4f}")
for sl, cv in zip(result_remod.significance_level, result_remod.critical_values):
    if result_remod.statistic < cv:
        print(f"✔ 정규성 만족 (유의수준 {sl}%)")
    else:
        print(f"✘ 정규성 불만족 (유의수준 {sl}%)")


result_not_remod = anderson(old_not_remod)

print("\n[오래된 주택 - 리모델링 X]")
print(f"Statistic: {result_not_remod.statistic:.4f}")
for sl, cv in zip(result_not_remod.significance_level, result_not_remod.critical_values):
    if result_not_remod.statistic < cv:
        print(f"✔ 정규성 만족 (유의수준 {sl}%)")
    else:
        print(f"✘ 정규성 불만족 (유의수준 {sl}%)")
stat, pval = shapiro(old_not_remod)
print(f"Shapiro-Wilk 검정 통계량: {stat:.4f}, p-value: {pval:.4f}")
# Shapiro-Wilk 검정 통계량: 0.9769, p-value: 0.2633
# 리모델링 안 한 주택은 정규성 만족

# 맨휘트니 검정으로 단측검정 (리모델링 한 집이 더 비싼가?)
stat, p = mannwhitneyu(old_remod, old_not_remod, alternative='greater')
print(f"Mann-Whitney U 검정 통계량: {stat:.4f}, p-value: {p:.4f}")
# Mann-Whitney U 검정 통계량: 18992.5000, p-value: 0.0075
# 리모델링 한 집이 더 비쌈


# 집 연식에 따라 리모델링을 언제 제일 많이 하는지?
# 최빈값 계산 (가장 많이 리모델링한 연차)
mode_value = remod_cleaned['diff'].mode()[0]
plt.hist(remod_cleaned['diff'], bins=40, color='blue', alpha=0.7)
plt.axvline(mode_value, color='red', linestyle='dashed', linewidth=2, label=f"최빈값: {mode_value}")
plt.title('리모델링 많이하는 집 연식')
plt.xlabel('집 연식')
plt.ylabel('빈도수')
plt.legend()
plt.show()
# 30년차에 리모델링을 많이 함







remodel_1950 = remod_cleaned.loc[(remod_cleaned['YearRemodAdd'] == 1950), :]
plt.hist(remodel_1950['YearBuilt'], bins=30, color='blue', alpha=0.7)


temp = remodel.loc[remodel['diff'] < 10, 'diff']
remodel = remodel.loc[remodel['diff'] > 1, :]