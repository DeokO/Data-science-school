# -*- coding: utf-8 -*-
'''------------------------------------------------------------------------
 Deokseong Seo
 M.S. Student in Industrial Management Engineering
 Korea University, Seoul, Republic of Korea
 E-mail    heyhi16@korea.ac.kr
 Data Science and Business Analytics Lab
 Lab Homepage http://dsba.korea.ac.kr
------------------------------------------------------------------------'''

from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
sy = pd.Series(iris.target, dtype = 'category')
sy.cat.rename_categories(iris.target_names)
df['species'] = sy

from sklearn.preprocessing import StandardScaler
iris = load_iris()

data1 = iris.data
data2 = scale(iris.data)

print("old mean:", np.mean(data1, axis=0))
print("old std: ", np.std(data1, axis=0))
print("new mean:", np.mean(data2, axis=0))
print("new std: ", np.std(data2, axis=0))
scaler = StandardScaler()
scaler.fit(data1)
data2 = scaler.transform(data1)
data1.std()



import seaborn as sns
from sklearn.datasets import load_boston

boston = load_boston()
x = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.DataFrame(boston.target, columns=['y'])
data = pd.concat([x, y], axis=0) #1이 열로 붙임 0이 행
data.tail()
cols = ["LSTAT", "NOX", "RM", "MEDV"]
sns.pairplot(df[cols])
plt.show()

import numpy as np
from sklearn.datasets import fetch_20newsgroups
dd = fetch_20newsgroups(subset='all')
print(dd.description)
print(dd.keys())
np.unique(dd.target, return_inverse=True)
from pprint import pprint
pprint(list(dd.target_names))
dd.data[1]
dd.target_names[dd.target[1]]

from sklearn.preprocessing import scale, robust_scale, minmax_scale, maxabs_scale

x = (np.arange(10, dtype=np.float) - 3).reshape(-1, 1)
df = pd.DataFrame(np.hstack([x, scale(x), robust_scale(x), minmax_scale(x), maxabs_scale(x)]), 
                  columns=["x", "scale(x)", "robust_scale(x)", "minmax_scale(x)", "maxabs_scale(x)"])
df


from sklearn.preprocessing import normalize

x = np.vstack([np.arange(5, dtype=float) - 20, np.arange(5, dtype=float) - 2]).T
y1 = scale(x)
y2 = normalize(x)

print("original x:\n", x)
print("scale:\n", y1)
print("norms (scale)\n", np.linalg.norm(y1, axis=1))
print("normlize:\n", y2)
print("norms (normalize)\n", np.linalg.norm(y2, axis=1))

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
x = np.array([0, 1, 2]).reshape(3,-1)
x
ohe.fit(x)
ohe.n_values_
ohe.feature_indices_
ohe.active_features_
ohe.transform(x).toarray()

x = np.array([[0, 0, 4], [1, 1, 0],  [0, 2, 1], [1, 0, 2]])
x
ohe.fit(x)
ohe.n_values_

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit([1, 2, 2, 6])
le.classes_
le.transform([1, 1, 2, 6])
le.inverse_transform([0, 0, 2, 1])
le.fit(['a', 'a', 'b', 'b', 'c'])
le.classes_
le.fit_transform(['b', 'b', 'a', 'c'])
le.inverse_transform([0, 0, 1, 2, 2])


from sklearn.preprocessing import Binarizer
X = [[ 1., -1.,  2.],
     [ 2.,  0.,  0.],
     [ 0.,  1., -1.]]
binarizer = Binarizer()
binarizer.fit(X)
binarizer.transform(X)
binarizer = Binarizer(threshold=1.1)
binarizer.transform(X)

from sklearn.feature_extraction import DictVectorizer
v = DictVectorizer(sparse=False)
D = [{'foo':1, 'bar':2}, {'foo':3, 'baz':1}]
X = v.fit_transform(D)
X
v.feature_names_
v.inverse_transform(X)
v.transform({'foo':4, 'unseen_feature':3})

from sklearn.preprocessing import FunctionTransformer
def all_b(x):
    return(x[:, 1:])
x = np.arange(12).reshape(4,3)
func = FunctionTransformer(all_b)
func.fit_transform(x)
