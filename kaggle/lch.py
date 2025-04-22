import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
import pandas as pd
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