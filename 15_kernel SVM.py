# -*- coding: utf-8 -*-
'''------------------------------------------------------------------------
 Deokseong Seo
 M.S. Student in Industrial Management Engineering
 Korea University, Seoul, Republic of Korea
 E-mail    heyhi16@korea.ac.kr
 Data Science and Business Analytics Lab
 Lab Homepage http://dsba.korea.ac.kr
------------------------------------------------------------------------'''

#XOR 문제를 풀지 못하는 선형분류기
import numpy as np
import matplotlib.pyplot as plt
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:,0] > 0, X_xor[:,1] > 0) #TT, FF이면 F를 반환, 나머지 경우에 T 반환. 두개의 논리 조건이 들어감
y_xor = np.where(y_xor, 1, -1) #T인것은 1을, 아니면 -1을 
plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1], c='b', marker='o', label='1', s=100)
plt.scatter(X_xor[y_xor==-1, 0], X_xor[y_xor==-1, 1], c='r', marker='s', label='-1', s=100)
plt.ylim(-3.0)
plt.legend()
plt.title("XOR problem")
plt.show()


import matplotlib as mpl
def plot_xor(X, y, model, title, xmin=-3, xmax=3, ymin=-3, ymax=3):
    XX, YY = np.meshgrid(np.arange(xmin, xmax, (xmax-xmin)/1000), np.arange(ymin, ymax, (ymax-ymin)/1000))
    ZZ = np.reshape(model.predict(np.array([XX.ravel(), YY.ravel()]).T), XX.shape)
    plt.contourf(XX, YY, ZZ, cmap=mpl.cm.Paired_r, alpha=0.5)
    plt.scatter(X[y== 1, 0], X[y== 1, 1], c='b', marker='o', label='+1', s=100)
    plt.scatter(X[y==-1, 0], X[y==-1, 1], c='r', marker='s', label='-1', s=100)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.title(title)
    plt.show()
from sklearn.svm import SVC
svc = SVC(kernel='linear').fit(X_xor, y_xor)
plot_xor(X_xor, y_xor, svc, "Linear SVC")


#기저 함수를 사용한 비선형 판별 모형
from sklearn.preprocessing import FunctionTransformer
def basis(X):
    return np.vstack([X[:, 0]**2, np.sqrt(2)*X[:, 0]*X[:, 1], X[:, 1]**2]).T

X = np.arange(8).reshape(4, 2)
X

FunctionTransformer(basis).fit_transform(X)
#basis(X) 이랑 같은 역할을 함

X_xor2 = FunctionTransformer(basis).fit_transform(X_xor)
plt.scatter(X_xor2[y_xor==1, 0], X_xor2[y_xor==1, 1], c='b', s=50)
plt.scatter(X_xor2[y_xor==-1, 0], X_xor2[y_xor==-1, 1], c='r', s=50)
plt.show()

from sklearn.pipeline import Pipeline
#basis를 이용해서 고차원으로 각 점을 맵핑
#그 차원에서 SVC를 이용, 선형 분류기를 적합하고, predict value 얻음
#다시 원래 차원으로 돌아와서 contour를 그림
basismodel = Pipeline([('basis', FunctionTransformer(basis)), ('svc', SVC(kernel='linear'))]).fit(X_xor, y_xor)
plot_xor(X_xor, y_xor, basismodel, "Basis-Fuction SVC")

#polynomial kernel 이용한 경우, degree는 차수, gamma는 앞에 곱해지는 상수, coef0는 뒤에 더해지는 상수
polysvc = SVC(kernel='poly', degree=2, gamma=1, coef0=0).fit(X_xor, y_xor)
plot_xor(X_xor, y_xor, polysvc, "Polynomial SVC")

#RBF kernel 이용한 경우,
rbfsvc = SVC(kernel='rbf').fit(X_xor, y_xor)
plot_xor(X_xor, y_xor, rbfsvc, "RBF SVC")

#sigmoid kernel 이용한 경우, gamma는 앞에 곱해지는 상수, coef0는 뒤에 더해지는 수
sigmoidsvc = SVC(kernel='sigmoid', gamma=2, coef0=2).fit(X_xor, y_xor)
plot_xor(X_xor, y_xor, sigmoidsvc, "Sigmoid SVC")

#gamma가 높아질수록(=정규분포의 분산이 작아질수록) overfitting이 되어가는 모습을 볼 수 있다.
#이는 데이터간의 분포를 정규분포로 가정하고, 정규분포 기반의 거리를 구하는 것과 같은 의미를 가짐
#만약 정규분포의 분산을 작게 잡으면, 데이터간의 거리가 조금만 차이가 나더라도 크게 부풀려지며, 이는 overfitting의 효과를 가지게 된다.
#반대로, 정규분포의 분산을 크게 잡으면, 데이터간의 거리가 어느정도 차이가 나더라도 그 효과를 무시하게되며, 이는 underfitting의 효과를 가지게 된다.
plot_xor(X_xor, y_xor, SVC(kernel='rbf', gamma=2).fit(X_xor, y_xor), "RBF SVC (gamma=2)")
plot_xor(X_xor, y_xor, SVC(kernel='rbf', gamma=10).fit(X_xor, y_xor), "RBF SVC (gamma=10)")
plot_xor(X_xor, y_xor, SVC(kernel='rbf', gamma=50).fit(X_xor, y_xor), "RBF SVC (gamma=50)")
plot_xor(X_xor, y_xor, SVC(kernel='rbf', gamma=100).fit(X_xor, y_xor), "RBF SVC (gamma=100)")


#예 : iris
from sklearn.datasets import load_iris
#from sklearn.cross_validation import train_test_split 여기서 이제 안되고, 아래의 모듈로 들어가야 사용 가능하다.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#데이터 로딩
iris = load_iris()
X = iris.data[:, [2,3]]
y = iris.target
#테스트셋(30%), 트레이닝셋(70%)로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#정규화
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_combined_std = np.vstack((X_train_std, X_test_std)) #vertically stack. cbind 역할을 함
y_combined = np.hstack((y_train, y_test)) #horizentally stack. rbind 역할을 함
#ploting
def plot_iris(X, y, model, title, xmin=-2.5, xmax=2.5, ymin=-2.5, ymax=2.5):
    XX, YY = np.meshgrid(np.arange(xmin, xmax, (xmax-xmin)/1000), np.arange(ymin, ymax, (ymax-ymin)/1000))
    ZZ = np.reshape(model.predict(np.array([XX.ravel(), YY.ravel()]).T), XX.shape)
    plt.contourf(XX, YY, ZZ, cmap=mpl.cm.Paired_r, alpha=0.5)
    plt.scatter(X[y==0, 0], X[y==0, 1], c='r', marker='^', label='0', s=100)
    plt.scatter(X[y==1, 0], X[y==1, 1], c='g', marker='o', label='1', s=100)
    plt.scatter(X[y==2, 0], X[y==2, 1], c='b', marker='s', label='2', s=100)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.title(title)
    plt.show()
model = SVC(kernel='linear').fit(X_train_std, y_train)
plot_iris(X_test_std, y_test, model, "Linear SVC")
model = SVC(kernel='poly', random_state=0, gamma=10, C=1.0).fit(X_train_std, y_train)
plot_iris(X_test_std, y_test, model, "Polynomial SVC (gamma=10, C=1)")
model = SVC(kernel='rbf', random_state=0, gamma=1, C=1.0).fit(X_train_std, y_train)
plot_iris(X_test_std, y_test, model, "RBF SVC (gamma=1, C=1)")