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
# Numpy 배열 사용하기 
#==============================================================================


import numpy as np

#1차원 배열 만들기
a = np.array([0,1,2,3,4,5,6,7,8,9])
a
type(a)

#벡터화 연산
x = a*2
x
np.exp(x)
np.log(x)
np.sin(x)
np.all(x)
#x.으로 되는 것이 있고, x.으로 안되는 것이 있음
#np.FUNCTION(x) 이렇게 하는 것이 안전할 듯
x.all()
x.sin()
x.exp()

#2차원 배열 만들기
b = np.array([[0,1,2],[3,4,5]])
b
len(b)
len(b[0])

#3차원 배열 만들기
c = np.array([[[1,2,3,4],[5,6,7,8],[9,10,11,12]],[[11,12,13,14],[15,16,17,18],[19,20,21,22]]])   # 2 x 3 x 4 array
c
len(c)
len(c[0])
len(c[0][0])

#배열의 차원과 크기 알아내기
print(a.ndim)
print(np.ndim(a))
a.shape
np.shape(a)

#배열의 인덱싱
a=np.array([[0,1,2], [3,4,5]])
a
a[0,0]
a[-1,-1]

#배열의 슬라이싱
a=np.array([[0,1,2,3], [4,5,6,7]])
a
a[0,:]
a[0,]
a[0,:]
a[:,1]
#a[,1] 이렇게 하면 안됨. 빈곳은 :를 채워주자
a[1,1:]
a[:2,:2]

#배열 인덱싱(팬시 인덱싱)
a = np.array([0,1,2,3,4,5,6,7,8,9])
idx = np.array([True, False, True, False, True, False, True, False, True, False])
a[idx]
listidx = [True, False, True, False, True, False, True, False, True, False]
#a[listidx] 이렇게 하면 잘 안된다. array형으로 indexing 해야 함
a[idx]
a[a%2==0] #안쪽의 a%2==0 값이 array이면서 boolean이다.
a = np.array([0, 1, 2, 3]) * 10
idx=np.array([0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2]) #정수값을 이용해 하나하나 집을수 있다.
a[idx]












#==============================================================================
# Numpy 배열 생성과 변형
#==============================================================================
#inf, nan 등의 값을 표현할 수 있다.
np.inf
np.array([1,0]) / np.array([0,0])

#배열의 생성
x = np.array([1,2,3])
x
#기본값 없이 다음의 함수들을 이용해서 객체를 생성할 수 있다.
a = np.zeros(5)
a
#dtype을 이용해서 자료형을 명시할 수 있다.
b = np.zeros((5,2), dtype='f8')
b
#모든 원소의 문자열의 크기가 같아야 한다.
c = np.zeros(5, dtype="S4")
c
c[0] = 'abcd'
c[1] = 'ABCD'
#1로 배열을 만드는 경우
d = np.ones((2,3,4))
d
#크기를 튜플로 명시하지 않고 특정 배열이나 리스트와 같은 크기로 만들고 싶은 경우
e = range(10)
print(e)
f = np.ones_like(e, dtype='f')
f
#빈칸으로 만들고 싶은 경우. 쓰레기값이 들어가게 된다.
g = np.empty((4,3))
g
#arange (시작, 끝, step 크기)
np.arange(10)
np.arange(3, 21, 2)
np.arange(0, 100, 5)
#linear나 log구간을 지정한 구간 수만큼 분할한다
np.linspace(0, 100, 5)
np.logspace(0, 4, 4)
#난수 생성은 random 서브패키지의 rand(균등분포)또는 randn(정규분포)이용. 
np.random.seed(1234)
np.random.rand(4)
np.random.randn(4)
np.random.randn(3,5)
#배열의 크기 변형
a = np.arange(12)
a
b = np.reshape(a, (3,4)) #자료에는 a.reshape(3,4) 로 나와있음
b
a.reshape(3,4)
np.reshape(a, (2,-1,2)) #전체 원소개수가 정해져 있어서 하나의 차원은 -1로 주면 알아서 계산
a.flatten() #한 벡터로 쭉 펴주는 함수
#주의사항
x = np.arange(5)
x
np.reshape(x, (1,5))
np.reshape(x, (5,1))
x = np.arange(5)
x[:, np.newaxis]

#배열의 연결
#cbind = hstack
a1 = np.ones((2,3))
a1
a2 = np.zeros((2,2))
a2
np.hstack([a1, a2])
#rbind = vstack
b1 = np.ones((2,3))
b1
b2 = np.zeros((3,3))
b2
np.vstack([b1, b2]) #b1.vstack(b2) b1.vstack([b2]) 이런식으로 되지 않는다.
#3차원으로 합침 dstack
c1 = np.ones((2,3))
c1
c2 = np.zeros((2,3))
c2
np.dstack([c1, c2])
#R의 list처럼 그냥 모으는 역할. 크기가 모두 같아야 함 stack
c = np.stack([c1, c2], axis=0) #axis는 -3부터 3까지 가능
c
#array_equal 로 배열 원소가 아닌 전체를 비교
np.array_equal(c[0,:,:], c1)
np.array_equal(c[1,:,:], c2)
#hstack과 같은 역할을 하는 r_은 대괄호를 이용해 붙인다.
np.r_[np.array([1,2,3]), np.array([4,5,6])]
#vstack과 같은 역할을 하는 c_은 대괄호를 이용해 붙인다.
np.c_[np.array([1,2,3]), np.array([4,5,6])]
#tile은 동일한 배열을 반복하여 연결
a = np.array([0,1,2])
np.tile(a, 2)
np.tile(a, (3,2))

#그리드 생성 (작은 길이를 가진 변수에 맞춰져서 zip을 생성함
x = np.arange(3)
y = np.arange(5)
X, Y = np.meshgrid(x, y)
[zip(x, y) for x, y in zip(X, Y)]
list(zip(x, y))





#==============================================================================
# Numpy 배열의 연산
#==============================================================================
x = np.arange(1, 1001)
y = np.arange(1001, 2001)

z = np.zeros_like(x)
for i in range(1000) :
    z[i] = x[i] + y[i]
# %%time #계산하고자 하는 식을 돌릴때 같이 끼워서 돌리면 소요시간 확인 가능
z = x + y
#비교
a = np.array([1, 2, 3, 4])
b = np.array([4, 2, 2, 4])
a == b
a >= b
c = np.array([1, 2, 3, 4])
np.all(a == b)
np.all(a == c)
a = np.arange(5)
np.exp(a)
10 ** a
np.log(a)
np.log10(a)

#스칼라와 벡터/행렬의 곱셈
x = np.arange(10)
x
100 * x
x = np.arange(12).reshape((3,4))
100 * x

#브로드캐스팅 (다른 길이의 벡터를 더하거나 빼는 경우, 작은 배열을 반복해서 늘림)
x = np.arange(5)
x
y = np.ones_like(x)
y
x + y
x + 1
a = np.tile(np.arange(0, 40, 10), (3, 1)).T
a
b = np.array([0, 1, 2])
b
a + b
a = np.arange(0, 40, 10)[:, np.newaxis]
a + b

#차원 축소 연산(통계량 뽑는 용으로 사용 가능)
x = np.array([1, 2, 3, 4])
x
np.sum(x)
x.sum()
x = np.array([1, 3, 2])
x.min()
np.min(x)
x.argmin() #최소값의 위치
x = np.array([1, 2, 3, 1])
x.mean()
np.mean(x)
np.median(x)
np.all([True, False, True])
np.any([True, True, False])
a = np.zeros((100, 100), dtype=np.int)
a
np.any(a != 0)
np.all(a == a)
a = np.array([1, 2, 3, 2])
b = np.array([2, 2, 3, 2])
c = np.array([6, 4, 4, 5])
((a<=b) & (b<=c)).all()

#2차원 이상인 경우의 차원축소에서는 axis를 준다. 0:행, 1:열, 디폴트:1
x = np.array([[1,1], [2,2]])
x
x.sum()
x.sum(axis=0)
x.sum(axis=1)

#sort
a = np.array([[4, 3, 5], [1, 2, 1]])
a
np.sort(a)
np.sort(a, axis=0) #행 자체를 순서를 줌. 첫번째 원소만 확인하는 듯
np.sort(a, axis=1) #행 내부적으로 순서를 줌.
#순서(order)만 알고싶다면 argsort
a = np.array([4, 3, 1, 2])
j = np.argsort(a)
a[j]




#==============================================================================
# Numpy를 활용한 선형대수 입문
#==============================================================================
x = np.array([[1], [2], [3], [4]])
x #원래 정의대로라면 이렇게 열벡터로 표현해야함
x = np.array([1, 2, 3, 4])
x #그런데 그냥 행벡터로 표현해도 연산에 문제가 없음
X = np.array([[11, 12, 13], [14, 15, 16]])
X
#대각행렬
np.diag([1, 2, 3])
#단위행렬
np.identity(3)
np.eye(4)
#전치행렬. 함수가 아니라 attribute, 속성값이다.
X = np.array([[11, 12, 13], [14, 15, 16]])
X.T
#np.T(X) 이렇게 하면 안됨
#행렬의 사칙연산
x = np.array([1, 2, 3, 4])
y = np.array([4, 5, 6, 7])
x+y
x-y
x = np.arange(10)
x.mean()
np.mean(x)
N = len(x)
np.dot(np.ones(N), x)/N
#행렬의 곱셈
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[1, 2], [3, 4], [5, 6]])
C = np.dot(A, B)
A
B
C
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.array([[9, 8], [7, 6]])
np.dot(A, B+C)
np.dot(A, B) + np.dot(A, C)
np.dot(A+B, C)
np.dot(A, C)+np.dot(B, C)
I = np.eye(2)
np.dot(I, A)

#가중합
from sklearn.datasets import make_regression
X, y = make_regression(4, 3)
X
y
w = np.linalg.lstsq(X, y)[0]
w
e = y-np.dot(X, w)
np.dot(e.T, e)
#2차형식
x = np.array([1, 2, 3])
A= np.arange(1, 10).reshape(3, 3)
A
np.dot(np.dot(x, A), x)




#==============================================================================
# 행렬의 연산과 성질
#==============================================================================
#norm
A = (np.arange(9)-4).reshape((3, 3))
A
np.linalg.norm(A)
#trace
np.trace(np.eye(3))
#determinant
np.linalg.det(np.array([[1, 2], [3, 4]]))




#==============================================================================
# 연립방정식과 역행렬
#==============================================================================
A = np.array([[1, 3, -2], [3, 5, 6], [2, 4, 3]])
A
Ainv = np.linalg.inv(A)
Ainv

b = np.array([[5], [7], [8]])
x = np.dot(Ainv, b)
x
np.dot(A, x) - b 
x, resid, rank, s = np.linalg.lstsq(A, b)
x
resid
rank
s
#linalg.lstsq는 최소자승법을 푸는 함수이다.
A = np.array([[2,0], [-1,1], [0,2]])
A
b = np.array([[1], [0], [-1]])
b
Apinv = np.dot(np.linalg.inv(np.dot(A.T, A)), A.T) #(At * A)^-1 * At
Apinv
x = np.dot(Apinv, b)
x
np.dot(A, x) - b
x, resid, rank, s = np.linalg.lstsq(A, b)
x

