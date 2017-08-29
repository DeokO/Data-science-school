# -*- coding: utf-8 -*-
'''------------------------------------------------------------------------
 Deokseong Seo
 M.S. Student in Industrial Management Engineering
 Korea University, Seoul, Republic of Korea
 Mobile Phone +82 10 2461 5207
 E-mail    heyhi16@korea.ac.kr
 Data Science and Business Analytics Lab
 Lab Homepage http://dsba.korea.ac.kr
------------------------------------------------------------------------'''

#==============================================================================
# 선형 회귀분석의 가정
# 1. 오차의 등분산성
# 2. 오차의 정규성
# 3. 오차의 독립성
# 4. 선형성
# 추가) X(독립변수) 행렬이 full rank여야 역행렬을 구하고, 회귀계수를 정확하게 구할 수 있다.
#==============================================================================

#==============================================================================
# 선형 회귀 진단
# 1. 잔차의 정규성 검정
# 2. 잔차의 자기상관계수 검정
# 3. 독립변수에 대한 condition number 계산
#==============================================================================

from sklearn.datasets import make_regression
from statsmodels.regression import linear_model as sm
import pandas as pd

X0, y, coef = make_regression(n_samples=100, n_features=1, noise=20, coef=True, random_state=0) #noise : standard deviation of the gaussian noise applied to the output.
dfX0 = pd.DataFrame(X0, columns=['X1'])
dfX = sm.add_constant(dfX0) # 독립변수 앞에 1을 넣어서 절편용 열을 하나 만들어준다.
dfy = pd.DataFrame(y, columns=['y'])

model = sm.OLS(dfy, dfX)
result = model.fit()
print(result.summary())


# 다중공선성(독립변수중 어떤 변수는 다른 변수들로 설명 가능한 경우, full rank가 안되게 되어 회귀계수 추정에 문제가 발생함)을 해결하는 방법
# 1. 변수 선택법으로 의존적인 변수 제거
# 2. PCA로 새로운 변수를 추출
# 3. regulize 방법론 적용 (Lasso, Ridge, ElasticNet 등)

from statsmodels.datasets.longley import load_pandas
import seaborn as sns
import matplotlib.pyplot as plt

y = load_pandas().endog
X = load_pandas().exog
sns.pairplot(X)
plt.show()

X = sm.add_constant(X)
ols_model = sm.OLS(y, X)
ols_results = ols_model.fit()
print(ols_results.summary())




#sklearn을 이용한 다항회귀
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np

def polyreg(degree, seed=0, plot=True):
    polynomial_features = PolynomialFeatures(degree = degree)
    linear_regression = LinearRegression()
    model = Pipeline([('polynomial_features', polynomial_features),
                      ('linear_regression', linear_regression)]) #polynomial_features에는 predict 함수가 없음. 이걸 사용하고자 linear를 pipeline으로 끼워넣어줌
    
    np.random.seed(seed)
    n_samples = 30
    X = np.sort(np.random.rand(n_samples))
    y = np.cos(1.5 * np.pi * X) + np.random.randn(n_samples) * 0.1
    X = X[:, np.newaxis]

    model.fit(X, y)
    
    if plot:
        plt.scatter(X, y)
        xx = np.linspace(0, 1, 1000)
        plt.plot(xx, model.predict(xx[:, np.newaxis]))
        plt.ylim(-2, 2)
        plt.show()
        
    reg = model.named_steps['linear_regression']
    return(reg.coef_, reg.intercept_)

polyreg(1)
polyreg(2)
polyreg(3)
polyreg(50) # 과적합

#데이터수에 비해 추정할 모수가 그리 많지 않음. 과적합이 되지 않는다. training data에 크게 종속적으로 변하지 않는다.
polyreg(2, 0)
polyreg(2, 1)
polyreg(2, 2)

#데이터수에 비해 추정할 모수가 그리 많음. 과적합이 된다. training data에 크게 종속적으로 변한다.
polyreg(50, 0)
polyreg(50, 1)
polyreg(50, 2)




