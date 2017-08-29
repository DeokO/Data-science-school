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

#target값이 0, 2인 것들만 가지고와서 분류 perceptron을 적합해보고자 함
iris = load_iris()
idx = np.in1d(iris.target, [0,2]) #iris.target 중에서 0, 2를 갖는 값이 있는 위치를 줌(boolean 형)
X = iris.data[idx, 0:2]
y = iris.target[idx]
plt.scatter(X[:,0], X[:,1], c=y, s=100)
plt.show()


#Perceptron class이용
from sklearn.linear_model import Perceptron
import matplotlib as mpl
import seaborn as sns
def plot_perceptron(n):
    model = Perceptron(n_iter=n, eta0=0.1, random_state=1).fit(X, y)
    XX_min = X[:,0].min()-1; XX_max = X[:,0].max()+1;
    YY_min = X[:,1].min()-1; YY_max = X[:,1].max()+1;
    XX, YY = np.meshgrid(np.linspace(XX_min, XX_max, 1000), np.linspace(YY_min, YY_max, 1000))
    ZZ = model.predict(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape) #np.c_는 R에서 cbind와 같은 역할. r_은 rbind
    cmap = mpl.colors.ListedColormap(sns.color_palette('Set2')) #palette
    plt.contourf(XX, YY, ZZ, cmap=cmap)
    plt.scatter(X[:, 0], X[:, 1], s=50, linewidth=2, c=y, cmap=cmap)
    plt.xlim(XX_min, XX_max)
    plt.ylim(YY_min, YY_max)
    plt.grid(False)
    
plot_perceptron(1)    

from ipywidgets import widgets
widgets.interact(plot_perceptron, n=widgets.IntSlider(min=1, max=100, step=1, value=1)) #변화하면서 보여주는 듯 하다

plot_perceptron(500)

from sklearn.metrics import confusion_matrix, classification_report
model = Perceptron(n_iter=500, eta0=0.1, random_state=1).fit(X, y)
confusion_matrix(y, model.predict(X))


#SGDClassifier class이용
from sklearn.linear_model import SGDClassifier
def plot_sgd(n):
    model = SGDClassifier(loss='hinge', n_iter=n, random_state=1).fit(X, y)
    XX_min = X[:,0].min()-1; XX_max = X[:,0].max()+1;
    YY_min = X[:,1].min()-1; YY_max = X[:,1].max()+1;
    XX, YY = np.meshgrid(np.linspace(XX_min, XX_max, 1000), np.linspace(YY_min, YY_max, 1000))
    ZZ = model.predict(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape) #np.c_는 R에서 cbind와 같은 역할. r_은 rbind
    cmap = mpl.colors.ListedColormap(sns.color_palette('Set2')) #palette
    plt.contourf(XX, YY, ZZ, cmap=cmap)
    plt.scatter(X[:, 0], X[:, 1], s=50, linewidth=2, c=y, cmap=cmap)
    plt.xlim(XX_min, XX_max)
    plt.ylim(YY_min, YY_max)
    plt.grid(False)
    
plot_sgd(1)

from ipywidgets import widgets
widgets.interact(plot_sgd, n=widgets.IntSlider(min=1, max=100, step=1, value=1))

plot_sgd(1000)
