# -*- coding: utf-8 -*-
'''------------------------------------------------------------------------
 Deokseong Seo
 M.S. Student in Industrial Management Engineering
 Korea University, Seoul, Republic of Korea
 E-mail    heyhi16@korea.ac.kr
 Data Science and Business Analytics Lab
 Lab Homepage http://dsba.korea.ac.kr
------------------------------------------------------------------------'''

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
news = fetch_20newsgroups(subset='all')
model = Pipeline()
model = Pipeline([
                  ('vect', TfidfVectorizer(stop_words='english')),
                  ('nb', MultinomialNB())])

model.fit(news.data, news.target)
x = news.data[:1]
y = model.predict(x)[0]

print(x[0])
print(x[0])
print("=" * 80)
print("Actual Category:", news.target_names[news.target[0]])
print("Predicted Category:", news.target_names[y])
model.predict(x)
y
y = model.predict(x)[1]
y
y = model.predict(x)[6]
y = model.predict(x)[2]
y = model.predict(x)[0]
y
model.classes_
model.predict_log_proba(x)
y = model.predict(x)
y
y = model.predict(x)[0]
y
model.predict_log_proba(x)
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
X0, y = make_classification(n_features=1, n_redundant=0, n_informative=1, n_clusters_per_class=1, random_state=4)
X0
y
model = LogisticRegression().fit(X0, y)
xx = np.linspace(-3, 3, 100)
import numpy as np
xx = np.linspace(-3, 3, 100)
model.coef_[0]
model.coef_
sigm = 1/(1+np.exp(-model.coef_[0][0]**xx - model.intercept_[0]))
sigm
import matplotlib as plt
plt.subplot(211)
plt.plot(xx, sigm)
plt.scatter(X0, y, marker='o', c=y, s=100)
plt.scatter(X0[0], model.predict(X0[:1]), marker='o', s=300, c='r', lw=5, alpha=0.5)
plt.plot(xx, model.predict(xx[:, np.newaxis]) > 0.5, lw=2)
plt.scatter(X0[0], model.predict_proba(X0[:1])[0][1], marker='x', s=300, c='r', lw=5, alpha=0.5)
plt.axvline(X0[0], c='r', lw=2, alpha=0.5)
plt.xlim(-3, 3)
plt.subplot(212)
plt.bar(model.classes_, model.predict_proba(X0[:1])[0], align="center")
plt.xlim(-1, 2)
plt.gca().xaxis.grid(False)
plt.xticks(model.classes_)
plt.title("conditional probability")
plt.tight_layout()
plt.show()


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
iris = load_iris()
X = iris.data[:, [2,3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train
X_test
y_train
iris.target
iris.target_names
X.combined = np.vstack(X_train, X_test)
X.combined = np.vstack([X_train, X_test])
X.combined = np.vstack((X_train, X_test))
X_combined = np.vstack((X_train, X_test))
X.combined = np.vstack((X_train, X_test))
X_combined = np.vstack((X_train, X_test))
y_combined = np.vstack((y_train, y_test))
y_train
y_test
y_combined = np.hstack((y_train, y_test))
y_combined 
X_combined
y_combined = np.vstack((y_train[:,np.newaxis], y_test[:,np.newaxis]))
y_combined 
y_combined = np.hstack((y_train, y_test))
tree = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=0).fit(X_train, y_train)
test_idx = range(105, 150)
test_idx 
resolution = 0.01
markers=('s', 'x', 'o', '^', 'v')
colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
import matplotlib as mpl
cmap = mpl.colors.ListedCorormap(colors[:len(np.unique(y_combined))])
cmap = mpl.colors.ListedColormap(colors[:len(np.unique(y_combined))])
x1_min, x1_max = X_combined[:,0].min()-1, X_combined[:,0].max()+1
x2_min, x2_max = X_combined[:,1].min()-1, X_combined[:,1].max()+1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution))
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
xx1
xx2 
xx1.ravel()
Z = tree.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
Z 
xx1.shape
xx1
Z
Z.shape
Z.reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
import matplotlib.pyplot as plt
import matplotlib.pyplot
y_combined = np.vstack((y_train[:,np.newaxis], y_test[:,np.newaxis]))
tree = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=0).fit(X_train, y_train)
test_idx = range(105, 150)
resolution = 0.01
markers = ('s', 'x', 'o', '^', 'v')
colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

cmap = mpl.colors.ListedColormap(colors[:len(np.unique(y_combined))])
x1_min, x1_max = X_combined[:,0].min()-1, X_combined[:,0].max()+1
x2_min, x2_max = X_combined[:,1].min()-1, X_combined[:,1].max()+1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
Z = tree.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
Z.reshape(xx1.shape)

rv1 = sp.stats.norm(-2,1.5); 
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_iris
iris = load_iris()
idx = np.in1d(iris.target, [0,2])
iris.target
idx 
X = iris.data[idx, 0:2]
X 
y = iris.target[idx]
model = Perceptron(n_iter=100, eta0=0.1, random_state=1).fit(X, y)
XX_min = X[:,0].min()-1; XX_max=X[:,0].max()+1;
YY_min = X[:,1].min()-1; XX_max=X[:,1].max()+1;
np.c_
help(np.c_)
XX, YY = np.meshgrid(np.linspace(XX_min, XX_max, 1000), np.linspace(YY_min, YY_max, 1000))
XX_min = X[:,0].min()-1; XX_max=X[:,0].max()+1;
YY_min = X[:,1].min()-1; YY_max=X[:,1].max()+1;
XX, YY = np.meshgrid(np.linspace(XX_min, XX_max, 1000), np.linspace(YY_min, YY_max, 1000))
XX
XX.ravel()
# np.r_['-1,2,0', index expression]
ZZ = model.predict(np.c_[XX.ravel(), YY.revel()].reshape(XX.shape))
ZZ = model.predict(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
ZZ
from sklearn import svm
xx, yy = np.meshgrind(np.linspace(-3, 3, 500))
xx, yy = np.meshgrid(np.linspace(-3, 3, 500))
xx, yy = np.meshgrid(np.linspace(-3, 3, 500), np.linspace(-3, 3, 500))
np.random.seed(0)
X = np.random.randn(300, 2)
X
X.mean()
X.var()
Y = np.loginal_xor(X[:,0]>0, X[:,1]>0)
Y = np.logical_xor(X[:,0]>0, X[:,1]>0)
Y
model = svm.NuSVC().fit(X, Y)
Z = model.decision_function(np.c[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
Z = model.decision_function(np.c[xx.ravel(), yy.ravel()])
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)