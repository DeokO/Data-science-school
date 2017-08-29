# -*- coding: utf-8 -*-
'''------------------------------------------------------------------------
 Deokseong Seo
 M.S. Student in Industrial Management Engineering
 Korea University, Seoul, Republic of Korea
 E-mail    heyhi16@korea.ac.kr
 Data Science and Business Analytics Lab
 Lab Homepage http://dsba.korea.ac.kr
------------------------------------------------------------------------'''

#혼동행렬
from sklearn.metrics import confusion_matrix
y_true = [2,0,2,2,0,1]
y_pred = [0,0,2,2,0,2]
confusion_matrix(y_true, y_pred)

y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
confusion_matrix(y_true, y_pred, labels=['bird', 'cat', 'ant'])

from sklearn.metrics import classification_report
y_true = [0, 1, 2, 2, 2]
y_pred = [0, 0, 2, 2, 1]
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_true, y_pred, target_names=target_names))

y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
print(classification_report(y_true, y_pred, target_names=["ant", "bird", "cat"]))


#ROC curve
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
X, y = make_classification(n_features=1, n_redundant=0, n_informative=1, n_clusters_per_class=1, random_state=4)
model = LogisticRegression().fit(X, y)
print(confusion_matrix(y, model.predict(X)))
print(classification_report(y, model.predict(X)))
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y, model.decision_function(X))

#multiclass
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
iris = load_iris()
model = LogisticRegression().fit(iris.data, iris.target)

from sklearn.metrics import roc_curve
fpr0, tpr0, thresholds0 = roc_curve(iris.target, model.decision_function(iris.data)[:, 0], pos_label=0)
fpr1, tpr1, thresholds1 = roc_curve(iris.target, model.decision_function(iris.data)[:, 1], pos_label=1)
fpr2, tpr2, thresholds2 = roc_curve(iris.target, model.decision_function(iris.data)[:, 2], pos_label=2)
model.decision_function(iris.data)[:, 1].min()
thresholds1.min()
from sklearn.metrics import auc
auc(fpr1, tpr1)
