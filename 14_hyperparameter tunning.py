# -*- coding: utf-8 -*-
'''------------------------------------------------------------------------
 Deokseong Seo
 M.S. Student in Industrial Management Engineering
 Korea University, Seoul, Republic of Korea
 E-mail    heyhi16@korea.ac.kr
 Data Science and Business Analytics Lab
 Lab Homepage http://dsba.korea.ac.kr
------------------------------------------------------------------------'''

#모형이 완성된 후에는 하이퍼파라미터 탐색을 통해 예측성능을 향상시킨다.
#scikit-learn에서의 튜닝도구 : 
    #validation_curve : 단일 하이퍼 파라미터 최적화
    #GridSearchCV : 그리드를 사용한 복수 하이퍼 파라미터 최적화
    #ParameterGrid : 복수 파라미터 최적화용 그리드

#validation_curve : 최적화 할 파라미터의 이름, 범위, 성능기준을 param_name, param_range, scoring 인수로 받아 모든 경우에 대해 성능 기준을 계산함
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.learning_curve import validation_curve
import numpy as np
import matplotlib.pyplot as plt

digits = load_digits()
X, y = digits.data, digits.target
param_range = np.logspace(-6, -1, 10) #10**-6부터 10**-1까지를 10등분함

#SVC를 X, y에 대해 적합하고, gamma라는 이름의 파라미터에 대해 위 파라미터마다 돌며 성능은 cross validation으로 accuracy를 구함. n_jobs=1은 코어를 1개 사용한다는 의미
# %%time #이거로 블록 실행시키면 얼마나 걸리는지 시간을 알 수 있음
train_scores, test_scores = validation_curve(SVC(), X, y,
                                             param_name='gamma', param_range=param_range,
                                             cv=10, scoring='accuracy', n_jobs=1) #10x10의 matrix 두개를 얻음.
    
train_scores_mean = np.mean(train_scores, axis=1) #각 행(변화하는 parameter)마다 얻은 성능을 평균냄
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title('Validation Curve with SVM')
plt.xlabel('$\gamma$')
plt.ylabel('Score')
plt.ylim(0.0, 1.1)
plt.semilogx(param_range, train_scores_mean, label="Training score", color='r')
plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                 alpha=0.2, color='r')
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color='g')
plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                 alpha=0.2, color='g')
plt.legend(loc='best')
plt.show()


#GridSearchCV : validation_curve함수와 달리 모형 래퍼(Wrapper)성격의 클래스이다. fit을 이용하여 복수개의 모형에 대해 실행해줘야 한다.
#grid_scores_ : param_grid의 모든 조합에 대한 성능 결과.
#best_score_ : 최고 성능의 지표 값
#best_params_ : 최고 성능을 보이는 파라미터
#best_estimator_ : 최고 성능을 보이는 파라미터를 가진 모형
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

#pipe line 안에는 리스트 형태로 한번에 엮어서 진행할 절차를 넣어준다.
pipe_svc = Pipeline([('scl', StandardScaler()), ('clf', SVC(random_state=1))])
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10., 100., 1000.]
param_grid = [{'clf__C':param_range, 'clf__kernel':['linear']},
              {'clf__C':param_range, 'clf__gamma':param_range, 'clf__kernel':['rbf']}]

#원래는 param_grid에다가 dict, list를 parameter 이름에 맞춰서(C, gamma, kernel) 이렇게만 해줘도 되지만, 지금은 pipeline이어서 clf__를 앞에 써준듯 하다.
gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=1)
# %time
gs = gs.fit(X, y)

print(gs.best_score_)
print(gs.best_params_)
gs.grid_scores_



#ParameterGrid : 파라미터를 조합하여 탐색 그리드를 생성해 주는 명령어로, iterator 역할을 한다.
#이거로 조합들을 만든다음 for문을 돌려서 진행하는 방식으로 탐색할 수 있음
from sklearn.grid_search import ParameterGrid
param_grid = {'a':[1, 2], 'b':[True, False]}
list(ParameterGrid(param_grid))
param_grid= [{'kernel':['linear']}, {'kernel':['rbf'], 'gamma':[1, 10]}]
list(ParameterGrid(param_grid))




