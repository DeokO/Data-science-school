# -*- coding: utf-8 -*-
'''------------------------------------------------------------------------
 Deokseong Seo
 M.S. Student in Industrial Management Engineering
 Korea University, Seoul, Republic of Korea
 E-mail    heyhi16@korea.ac.kr
 Data Science and Business Analytics Lab
 Lab Homepage http://dsba.korea.ac.kr
------------------------------------------------------------------------'''

from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pylab as plt

X, y = make_blobs(n_samples=50, centers=2, random_state=4, cluster_std=0.60)
xmin = X[:, 0].min()
xmax = X[:, 0].max()
ymin = X[:, 1].min()
ymax = X[:, 1].max()
xx = np.linspace(xmin, xmax, 10)
yy = np.linspace(ymin, ymax, 10)
X1, X2 = np.meshgrid(xx, yy)

def plot_svm(model):
    Z = np.empty(X1.shape)
    for (i, j), val in np.ndenumerate(X1):
        x1 = val
        x2 = X2[i, j]
        p = model.decision_function([[x1, x2]])
        Z[i, j] = p[0]
    levels = [-1.0, 0.0, 1.0]
    linestyles = ['dashed', 'solid', 'dashed']
    plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap=plt.cm.Paired)
    plt.contour(X1, X2, Z, levels, colors='k', linestyles=linestyles)
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=150, linewidth=3, facecolors='none')
    plt.show()

from sklearn.svm import SVC
model = SVC(kernel='linear').fit(X,y)
plot_svm(model)


x_new = [10,2]
model.decision_function([x_new])
model.coef_.dot(x_new) + model.intercept_
model.support_vectors_
model.support_ #이것들이 무엇을 의미하는지는 추가 탐색 필요
y[model.support_]


#슬랙변수 사용(패널티를 받으며 선형 분류기 적합) 
#panalty를 민감하게 받아들이는 정도를 의미하는 C를 크게주면 패널티를 민감하게 받는 방향으로 하여 margin이 작아짐. 과적합.
#C를 작게 주면 패널티를 많이 허용. 상대적으로 일반화됨. No overfitting
np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20

fignum = 1
for name, penalty in (('C=1', 1), ('C=0.05', 0.05)):
    clf = SVC(kernel='linear', C=penalty).fit(X, Y)
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy + a * margin
    yy_up = yy - a * margin

    plt.figure(fignum)

    x_min = -5; x_max = 5;
    y_min = -9; y_max = 9;
    XX, YY = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])
    Z = Z.reshape(XX.shape)
    plt.contourf(XX, YY, Z, cmap=plt.cm.Accent, alpha=0.6)

    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')
    plt.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], s=120, linewidth=4, facecolors='red')
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=60, linewidth=1, cmap=plt.cm.Paired)
    
    plt.xlim(x_min, x_max)
    plt.xlim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.title(name)
    plt.axis('tight')
    plt.show()
    
    fignum=fignum+1;
    

#얼굴 이미지 인식
from sklearn.datasets import fetch_olivetti_faces
faces = fetch_olivetti_faces()

N=2; M=5;
np.random.seed(0)
fig = plt.figure(figsize=(9,5))
plt.subplots_adjust(top=1, bottom=0, hspace=0, wspace=0.05)
klist = np.random.choice(range(len(faces.data)), N * M)
for i in range(N):
    for j in range(M):
        k = klist[i*M+j]
        ax = fig.add_subplot(N, M, i*M+j+1)
        ax.imshow(faces.images[k], cmap=plt.cm.bone);
        ax.grid(False)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        plt.title(faces.target[k])
plt.tight_layout()
plt.show()

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target, test_size=0.25, random_state=0)
from sklearn.svm import SVC
svc_1 = SVC(kernel='linear').fit(X_train, y_train)
fig = plt.figure(figsize=(3, 30))
for i, k in enumerate(np.random.choice(len(y_test), 10, replace=False)):
    ax = fig.add_subplot(10, 1, i+1)
    ax.imshow(X_test[k, :].reshape(64, 64), cmap=plt.cm.bone);  #X_test[k:(k+1), :] 으로 해도 똑같은 결과
    ax.grid(False)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    plt.title('actual %d => predict %d' %(y_test[k], svc_1.predict(X_test[k:(k+1),:])[0]))
plt.tight_layout()
plt.show()


#안경 쓴사람 맞추기
glasses = [
    ( 10,  19), ( 30,  32), ( 37,  38), ( 50,  59), ( 63,  64),
    ( 69,  69), (120, 121), (124, 129), (130, 139), (160, 161),
    (164, 169), (180, 182), (185, 185), (189, 189), (190, 192),
    (194, 194), (196, 199), (260, 269), (270, 279), (300, 309),
    (330, 339), (358, 359), (360, 369)
]

def create_target(segments):
    y = np.zeros(faces.target.shape[0])
    for (start, end) in segments:
        y[start:end+1]=1
    return y

target_glasses = create_target(glasses)

X_train, X_test, y_train, y_test = train_test_split(faces.data, target_glasses, test_size=0.25, random_state=0)
svc_2 = SVC(kernel='linear').fit(X_train, y_train)
fig = plt.figure(figsize=(3, 30))
for i, k in enumerate(np.random.choice(len(y_test), 10, replace=False)):
    ax = fig.add_subplot(10, 1, i + 1)
    ax.imshow(X_test[k:(k+1), :].reshape(64,64), cmap=plt.cm.bone);
    ax.grid(False)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    plt.title('prediction: %s' %('glasses' if (svc_2.predict(X_test[k:(k+1), :])[0]) else 'no glasses'))
plt.tight_layout()
plt.show()


