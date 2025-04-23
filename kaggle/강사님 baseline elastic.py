# 머신러닝 기본 베이스라인 - ElasticNet

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 집가격 데이터 불러오세요!
train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')

# 독립변수(X)와 종속변수(y) 분리
X_train = train_df.drop(columns='SalePrice')
y_train = train_df['SalePrice']
X_test = test_df
X_train = X_train.drop(columns=['Alley', 'Id'])
X_test = X_test.drop(columns=['Alley', 'Id'])

# 칼럼 선택
num_columns = X_train.select_dtypes(include=['number']).columns
cat_columns = X_train.select_dtypes(include=['object']).columns


from sklearn.impute import SimpleImputer

freq_impute = SimpleImputer(strategy='most_frequent')
mean_impute = SimpleImputer(strategy='mean')

# fit transform을 통해 결측치 대체
# test 데이터는 train 데이터로 fit한 imputer를 사용
X_train[cat_columns] = freq_impute.fit_transform(X_train[cat_columns])
X_test[cat_columns] = freq_impute.fit_transform(X_test[cat_columns])

# 수치형 변수 결측치를 평균으로 대체
X_train[num_columns] = mean_impute.fit_transform(X_train[num_columns])
X_test[num_columns] = mean_impute.fit_transform(X_test[num_columns])


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

onehot = OneHotEncoder(handle_unknown='ignore', 
                       sparse_output=False)

X_train_cat = onehot.fit_transform(X_train[cat_columns])
X_test_cat = onehot.transform(X_test[cat_columns])

std_scaler = StandardScaler()

X_train_num = std_scaler.fit_transform(X_train[num_columns])
X_test_num = std_scaler.transform(X_test[num_columns])
X_train_all = np.concatenate([X_train_num, X_train_cat], axis = 1)
X_test_all = np.concatenate([X_test_num, X_test_cat], axis = 1)

from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet()
elastic_params = {'alpha' : np.arange(0.1, 1, 0.1),
                  'l1_ratio': np.linspace(0, 1, 5)}

# 파라미터 확인 
ElasticNet().get_params()

# 교차검증
from sklearn.model_selection import KFold, GridSearchCV
cv = KFold(n_splits=5, shuffle=True, random_state=0)

# 그리드서치
elastic_search = GridSearchCV(estimator=elasticnet, 
                              param_grid=elastic_params, 
                              cv = cv, 
                              scoring='neg_mean_squared_error')

elastic_search.fit(X_train_all, y_train)


# 그리드서치 파라미터 성능 확인
print(pd.DataFrame(elastic_search.cv_results_))


# best prameter
print(elastic_search.best_params_)

# 교차검증 best score 
print(-elastic_search.best_score_)

# 최종 예측 
y_pred = elastic_search.predict(X_test_all)
submit = pd.read_csv('.sample_submission.csv')
submit["SalePrice"]=y_pred

# CSV로 저장
submit.to_csv('.minchae_result.csv', index=False)