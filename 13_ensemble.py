# -*- coding: utf-8 -*-
'''------------------------------------------------------------------------
 Deokseong Seo
 M.S. Student in Industrial Management Engineering
 Korea University, Seoul, Republic of Korea
 E-mail    heyhi16@korea.ac.kr
 Data Science and Business Analytics Lab
 Lab Homepage http://dsba.korea.ac.kr
------------------------------------------------------------------------'''

#==============================================================================
# 앙상블
# 1. 다수결 방법(다수결, 배깅, 랜덤 포레스트)
# 2. 부스팅 방법(에이다부스트, GBM)
#==============================================================================

#다수결 방법
import numpy as np
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt

X = np.array([[-1.0, -1.0], [-1.2, -1.4], [1, -0.5], [-3.4, -2.2], [1.1, 1.2], [-2.1, -0.2]])
y = np.array([1, 1, 1, 2, 2, 2])
x_new = [0,0]
plt.scatter(X[y==1, 0], X[y==1, 1], s=100, c='r')
plt.scatter(X[y==2, 0], X[y==2, 1], s=100, c='b')
plt.scatter(x_new[0], x_new[1], s=100, c='g')


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

#각 classifier를 1, 2, 3으로 객체 생성해두고, ensemble에 합칠 준비를 함
clf1 = LogisticRegression(random_state=1)
clf2 = SVC(random_state=1, probability=True)
clf3 = GaussianNB()
eclf = VotingClassifier(estimators=[('lr', clf1), ('ksvc', clf2), ('gnb', clf3)], voting='soft', weights=[2, 1, 1])

probas = [c.fit(X, y).predict_proba([x_new]) for c in (clf1, clf2, clf3, eclf)]
class1_1 = [pr[0,0] for pr in probas]
class2_1 = [pr[0,1] for pr in probas]

ind = np.arange(4)
width = 0.35 #bar width
p1 = plt.bar(ind, np.hstack([class1_1[:-1], [0]]), width, align='center', color='green')
p2 = plt.bar(ind+width, np.hstack([class2_1[:-1], [0]]), width, align='center', color='lightgreen')
p3 = plt.bar(ind, [0, 0, 0, class1_1[-1]], width, align = 'center', color='blue')
p4 = plt.bar(ind+width, [0, 0, 0, class2_1[-1]], width, align = 'center', color='lightblue')
plt.xticks(ind + 0.5 * width, ['LogisticRegression\nweight 2',
                               'Kernel svc\nweight 1',
                               'GaussianNB\nweight 1',
                               'VotingClassifier'])
plt.ylim([0, 1.1])
plt.title('Class probabilities for sample 1 by different classifiers')
plt.legend([p1[0], p2[0]], ['class 1', 'class 2'], loc='upper left')
plt.show()


#각각의 classifier가 어떻게 적합됐는지 확인해보기
from itertools import product
import matplotlib as mpl
x_min, x_max = -4, 2
y_min, y_max = -3, 2
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.025), np.arange(y_min, y_max, 0.025))
f, axarr = plt.subplots(2, 2)

for idx, clf, tt in zip(product([0, 1], [0, 1]),
                        [clf1, clf2, clf3, eclf],
                        ['LogisticRegression', 'Kernel SVC', 'Gaussian NB', 'Ensemble']):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.2, cmap=mpl.cm.jet)
    axarr[idx[0], idx[1]].scatter(X[:,0], X[:,1], c=y, alpha=0.5, s=50, cmap=mpl.cm.jet)
    axarr[idx[0], idx[1]].set_title(tt)
plt.tight_layout()
plt.show()


#여러 모형들과 합쳐서 살펴보기
from itertools import product
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sp

iris = load_iris()
X, y = iris.data[:,[0,2]], iris.target

model1 = DecisionTreeClassifier(max_depth=4).fit(X, y)
model2 = LogisticRegression().fit(X, y)
model3 = SVC(probability=True).fit(X, y)
model4 = VotingClassifier(estimators=[('dt', model1),
                                      ('lr', model2),
                                      ('svc', model3)],
                          voting='soft', weights=[1,2,3]).fit(X, y)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.025), np.arange(y_min, y_max, 0.025))
f, axarr = plt.subplots(2, 2)
for idx, clf, tt in zip(product([0,1], [0,1]), [model1, model2, model3, model4],
                        ['Decision Tree', 'Logistic Regression', 'Kernel SVM', 'soft Voting']):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.2, cmap=mpl.cm.jet)
    axarr[idx[0], idx[1]].scatter(X[:,0], X[:,1], c=y, alpha=1, s=50, cmap=mpl.cm.jet)
    axarr[idx[0], idx[1]].set_title(tt)
plt.tight_layout()
plt.show()


def total_error(p, N):
    te = 0.0
    for k in range(int(np.ceil(N/2)), N+1):
        te += sp.misc.comb(N, k) * p**k * (1-p)**(N-k)
    return te

x = np.linspace(0, 1, 100)
plt.plot(x, x, 'g:', lw=3, label="individual model")
plt.plot(x, total_error(x, 10), 'b-', label="voting model (N=10)")
plt.plot(x, total_error(x, 100), 'r-', label="voting model (N=100)")
plt.xlabel("performance of individual model")
plt.ylabel("performance of voting model")
plt.legend(loc=0)
plt.show()



### 배깅 scikit-learn의 ensemble 서브패키지의 BaggingClassifier 클래스를 이용하면 된다.
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier

iris = load_iris()
X, y = iris.data[:, [0,2]], iris.target

model1 = DecisionTreeClassifier().fit(X, y)
model2 = BaggingClassifier(DecisionTreeClassifier(), bootstrap_features=True, n_estimators=100, random_state=0).fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.025), np.arange(y_min, y_max, 0.025))

plt.figure(figsize=(8, 12))

plt.subplot(211)
Z1 = model1.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contourf(xx, yy, Z1, alpha=0.6, cmap=mpl.cm.jet)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=1, s=50, cmap=mpl.cm.jet)

plt.subplot(212)
Z2 = model2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contourf(xx, yy, Z2, alpha=0.6, cmap=mpl.cm.jet)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=1, s=50, cmap=mpl.cm.jet)

plt.tight_layout()
plt.show()


### 랜덤 포레스트
from sklearn import clone
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

#데이터 로딩
iris = load_iris()

#hyper parameter 설정
n_classes = 3 #클래스 개수
n_estimators = 30 #배깅 횟수
plot_colors = 'ryb' #색
cmap = plt.cm.jet #cmpa
plot_step = 0.02 #grid 간격
RANDOM_SEED = 13 #시드

#모형 3개. DT, RF, ET
models = [DecisionTreeClassifier(max_depth=4),
          RandomForestClassifier(max_depth=4, n_estimators=n_estimators),
          ExtraTreesClassifier(max_depth=4, n_estimators=n_estimators)]

plot_idx = 1 #서브플롯 인덱스. 1부터 9까지 간다.
plt.figure(figsize=(12, 12)) #플롯 사이즈
for pair in ([0,1], [0,2], [2,3]): #변수를 임의로 2개만 선택
    for model in models:
        X = iris.data[:, pair]
        y = iris.target
        
        #data sampling
        idx = np.arange(X.shape[0])
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]
    
        #정규화
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X-mean)/std
        
        #모형 적합
        clf = clone(model)
        clf = model.fit(X, y)
        
        
        plt.subplot(3, 3, plot_idx)
        model_title = str(type(model)).split('.')[-1][:-2][:-len("Classifier")]
        if plot_idx <= len(models):
            plt.title(model_title)
        
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), 
                             np.arange(y_min, y_max, plot_step))

        if isinstance(model, DecisionTreeClassifier):
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, cmap=cmap)
        else:
            estimator_alpha = 1.0/len(model.estimators_)
            for tree in model.estimators_:
                Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)
                
        for i, c in zip(range(n_classes), plot_colors):
            idx = np.where(y==i)
            plt.scatter(X[idx, 0], X[idx, 1], c=c, label=iris.target_names[i], cmap=cmap)
            
        plot_idx += 1

plt.tight_layout()
plt.show()


#랜덤포레스트에서 OOB error 구하기
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

#인공 데이터 생성
X, y = make_classification(n_samples=1000, n_features=10, n_informative=3, n_redundant=0, n_repeated=0,
                           n_classes=2, random_state=0, shuffle=False)

#ExtraTrees 분류기 적합
forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
forest.fit(X, y)

#포레스트에서 중요도 계산
importances = forest.feature_importances_

#각 tree마다 각 변수의 OOB error에 대한 std를 구함
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1] #원래 오름차순을 구하는데, ::-1로 내림차순을 구해줌

print('Feature ranking:')
for f in range(X.shape[1]):
    print("{} feature {} ({})".format(f+1, indices[f], importances[indices[f]]))
    
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


from sklearn.datasets import fetch_olivetti_faces
from sklearn.ensemble import ExtraTreesClassifier

data = fetch_olivetti_faces()
X = data.images.reshape((len(data.images), -1))
y = data.target

mask = y<5
X = X[mask]
y = y[mask]

forest = ExtraTreesClassifier(n_estimators=1000, max_features=128, random_state=0)
forest.fit(X, y)

importances = forest.feature_importances_
importances = importances.reshape(data.images[0].shape)

plt.figure(figsize=(8, 8))
plt.imshow(importances, cmap=plt.cm.bone_r)
plt.grid(False)
plt.title("Pixel importances with forests of trees")
plt.show()



#예 : 이미지 완성
from sklearn.datasets import fetch_olivetti_faces
from sklearn.utils.validation import check_random_state

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression

data = fetch_olivetti_faces()
targets = data.target

data = data.images.reshape((len(data.images), -1))
train = data[targets < 30]
test = data[targets >= 30]

n_faces = 5
rng = check_random_state(4)
face_idx = rng.randint(test.shape[0], size=(n_faces))
test = test[face_idx, :]

n_pixels = data.shape[1]
X_train = train[:, :int(np.ceil(0.5 * n_pixels))] #얼굴 반쪽 위
y_train = train[:, int(np.floor(0.5 * n_pixels)):] #얼굴 반쪽 아래
X_test = test[:, :int(np.ceil(0.5 * n_pixels))]
y_test = test[:, int(np.ceil(0.5 * n_pixels)):]

ESTIMATORS = {"Linear regression": LinearRegression(),
              "Extra trees" : ExtraTreesRegressor(n_estimators=10, max_features=32, random_state=0)}

y_test_predict = dict()
for name, estimator in ESTIMATORS.items():
    estimator.fit(X_train, y_train)
    y_test_predict[name] = estimator.predict(X_test)

image_shape = (64, 64)
n_cols = 1+len(ESTIMATORS)
plt.figure(figsize=(3*n_cols, 3*n_faces))
plt.suptitle("Face completion with multi-output estimators", size=16)

for i in range(n_faces):
    true_face = np.hstack((X_test[i], y_test[i]))
    if i:
        sub = plt.subplot(n_faces, n_cols, i*n_cols+1)
    else:
        sub = plt.subplot(n_faces, n_cols, i*n_cols+1, title="true faces")
    sub.axis('off')
    sub.imshow(true_face.reshape(image_shape), cmap=plt.cm.gray, interpolation='nearest')
    
    for j, est in enumerate(ESTIMATORS):
        completed_face = np.hstack((X_test[i], y_test_predict[est][i]))
        if i:
            sub = plt.subplot(n_faces, n_cols, i*n_cols+2+j)
        else:
            sub = plt.subplot(n_faces, n_cols, i*n_cols+2+j, title=est)
            
        sub.axis('off')
        sub.imshow(completed_face.reshape(image_shape), cmap=plt.cm.gray, interpolation='nearest');



### 에이다 부스트 AdaBoostClassifier 사용
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

#Construct dataset
X1, y1 = make_gaussian_quantiles(cov=2., n_samples=200, n_features=2, n_classes=2, random_state=1)
X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5, n_samples=300, n_features=2, n_classes=2, random_state=1)

X = np.concatenate((X1, X2)) #rbind역할. 아래 모두 동일한 내용
#X = np.vstack((X1, X2))
#X = np.r_[X1, X2]
y = np.concatenate((y1, -y2+1))

#Create and fit an AdaBoosted decision tree
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm='SAMME',
                         n_estimators=200)
bdt.fit(X, y)

plot_colors = 'br'
plot_step = 0.02
class_names = 'AB'

plt.figure(figsize=(12, 6))

plt.subplot(121)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), 
                     np.arange(y_min, y_max, plot_step))

Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.axis('tight')

for i, n, c in zip(range(2), class_names, plot_colors):
    idx = np.where(y==i)
    plt.scatter(X[idx, 0], X[idx, 1], c=c, cmap=plt.cm.Paired, label="Class{}".format(n))
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision Boundary')


twoclass_output = bdt.decision_function(X)
plt_range = (twoclass_output.min(), twoclass_output.max())
plt.subplot(122)
for i, n, c in zip(range(2), class_names, plot_colors):
    plt.hist(twoclass_output[y==i],
             bins=10,
             range=plt_range,
             facecolor=c,
             label="Class {}".format(n),
             alpha=.5)
x1, x2, y1, y2 = plt.axis()
plt.legend(loc='upper right')
plt.ylabel('Samples')
plt.xlabel('Score')
plt.title('Decision Scores')
plt.tight_layout()
plt.subplots_adjust(wspace=0.35)
plt.show()



#그래디언트 부스팅 - 개념을 좀 더 고민하는게 좋을듯. 코드는 너무 간단히 구현
from sklearn.datasets import make_hastie_10_2
X, y = make_hastie_10_2(random_state=0)
x0 = np.ravel(X[:,0])
idx = np.argsort(x0)
plt.plot(x0[idx], y[idx])
plt.xlim(-0.1, 0.1)
plt.show()

from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X, y)
from sklearn.metrics import classification_report
y_pred = model.predict(X)
print(classification_report(y_pred, y))







