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

#sigmoid function 류 정리
import matplotlib.pyplot as plt
import numpy as np
xx = np.linspace(-10, 10, 1000)
plt.plot(xx, (1/(1+np.exp(-xx)))*2-1, label='logistic (scaled)')
plt.plot(xx, np.tanh(xx), label='tanh')
plt.legend(loc=2)
plt.show()


#sklearn logistic regression
from sklearn.datasets import make_classification
import statsmodels.regression.linear_model as sm

X0, y = make_classification(n_features=1, n_redundant=0, n_informative=1, n_clusters_per_class = 1, random_state=4)
X = sm.add_constant(X0)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(X0, y)

import matplotlib as mpl
xx = np.linspace(-3, 3, 100)
sigm = 1.0/(1+np.exp(-model.coef_[0][0]*xx - model.intercept_[0]))
plt.plot(xx, sigm)
plt.scatter(X0, y, marker='o', c=y, s=100) #(X0,y) 점들을 찍고, 형태는 o이고, 색은 y값에 따라, 사이즈는 100
plt.scatter(X0, model.predict(X0), marker='x', c=y, s=200, lw=2, alpha=0.5, cmap=mpl.cm.jet)
plt.xlim(-3, 3)
plt.show()



