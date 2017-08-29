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

# 중심 k개 선정 -> 거리 계산 -> 중심 선택(할당) 및 중심 재 계산 -> 반복
import numpy as np
import matplotlib.pylab as plt

X = np.array([[7, 5],[5, 7],[7, 7],[4, 4],[4, 6],[1, 4],[0, 0],[2, 2],[8, 7],[6, 8],[5, 5],[3, 7]], dtype=float)
plt.scatter(X[:,0], X[:,1], s=100)
plt.show()


#KMeans class
from sklearn.cluster import KMeans
model = KMeans(n_clusters=2, init='random', n_init=1, max_iter=1, random_state=1).fit(X)
c0, c1 = model.cluster_centers_
plt.scatter(X[model.labels_==0, 0], X[model.labels_==0, 1], s=100, marker='v', c='r')
plt.scatter(X[model.labels_==1, 0], X[model.labels_==1, 1], s=100, marker='^', c='b')
plt.scatter(c0[0], c0[1], s=100, c='r')
plt.scatter(c1[0], c1[1], s=100, c='b')
plt.show()


import pandas as pd
def kmeans_df(c0, c1):
    #원래 좌표, c0로부터의 거리인 d0, c1으로부터의 거리인 d1, 모델에 의한 class인 c를 저장
    df = pd.DataFrame(np.hstack([X,
                                 np.linalg.norm(X - c0, axis=1)[:, np.newaxis],
                                 np.linalg.norm(X - c1, axis=1)[:, np.newaxis],
                                 model.labels_[:, np.newaxis]]),
                                columns=['x0', 'x1', 'd0', 'd1', 'c'])
    return(df)

kmeans_df(c0, c1)
print(X[model.labels_ == 0, 0].mean(), X[model.labels_ == 0, 1].mean())
print(X[model.labels_ == 1, 0].mean(), X[model.labels_ == 1, 1].mean())

# 이것은 왜 필요한지 모르겠다. 사용하지 않아도 될 듯 하다.
model.score(X)


model = KMeans(n_cluster=2, init='random', n_init=1, max_iter=2, random_state=0).fit(X)
c0, c1 = model.cluster_centers_
print(c0, c1)
plt.scatter(X[model.labels_==0, 0], X[model.labels_==0, 1], s=100, marker='v', c='r')
plt.scatter(X[model.labels_==1, 0], X[model.labels_==1, 1], s=100, marker='^', c='b')
plt.scatter(c0[0], c0[1], s=100, c='r')
plt.scatter(c1[0], c1[1], s=100, c='b')
kmeans_df(c0, c1)



# 예 : iris data
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib as mpl

np.random.seed(5)
centers = [[1, 1], [-1, -1], [1, -1]]
iris = datasets.load_iris()
X = iris.data
y = iris.target

estimators = {'k_means_iris_3' : KMeans(n_clusters=3),
              'k_means_iris_8' : KMeans(n_clusters=8)}

fignum=1
for name, est in estimators.items():
    fig = plt.figure(fignum)
    plt.clf()
    ax = Axes3D(fig, rect=[0,0,.95,1], elev=58, azim=134)
    plt.cla()
    est.fit(X)
    labels = est.labels_
    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float), s=100, cmap=mpl.cm.jet)
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    plt.title(name)
    fignum = fignum+1
    
plt.show()
    
    
    

