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

#==============================================================================
# 시각화 패키지 matplotlib 소개 http://matplotlib.org/1.5.1/api/pyplot_api.html#matplotlib.pyplot.plot
#==============================================================================
#예제 http://matplotlib.org/1.5.1/gallery.html
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np

plt.plot([1, 4, 9, 16]) #x는 자동으로 0, 1, 2, 3 가 됨
plt.show()

plt.plot([10, 20, 30, 40], [1, 4, 9, 16]) #x를 할당할 수 있음
plt.show()

#선 스타일 지정
plt.plot([1, 4, 9, 16], ls='-')
plt.plot([1, 4, 9, 16], ls='--')
plt.plot([1, 4, 9, 16], ls='-.')
plt.plot([1, 4, 9, 16], ls=':')
plt.plot([1, 4, 9, 16], 'rs:') #color:r, marker:s, line style::
plt.show()

#색 스타일 지정 http://matplotlib.org/examples/color/named_colors.html
plt.plot([1, 4, 9, 16], c='blue')
plt.plot([1, 4, 9, 16], c='b')
plt.plot([1, 4, 9, 16], c='green')
plt.plot([1, 4, 9, 16], c='g')
plt.plot([1, 4, 9, 16], c='red')
plt.plot([1, 4, 9, 16], c='r')
#이외에도 cyan(c), magenta(m), yellow(y), black(k), white(w)가 있음

#마커 스타일 지정(데이터의 위치 포인트)
plt.plot([1, 4, 9, 16], marker='.')
plt.plot([1, 4, 9, 16], marker=',')
plt.plot([1, 4, 9, 16], marker='o')
plt.plot([1, 4, 9, 16], marker='v')
plt.plot([1, 4, 9, 16], marker='^')
plt.plot([1, 4, 9, 16], marker='>')
plt.plot([1, 4, 9, 16], marker='<')
plt.plot([1, 4, 9, 16], marker='1')
plt.plot([1, 4, 9, 16], marker='+')
#이외에도 여러개...

#종합
plt.plot([1, 4, 9, 16], c='b', lw=5, ls='--', marker='o', ms=15, mec='g', mew=5, mfc='r')
plt.show()

#그림 범위 지정
plt.plot([1, 4, 9, 16], c='b', lw=5, ls='--', marker='o', ms=15, mec='g', mew=5, mfc='r')
plt.xlim(-0.2, 3.2)
plt.ylim(-1, 18)
plt.show()

#틱 설정
X = np.linspace(-np.pi, np.pi, 256)
C = np.cos(X)
plt.plot(X, C)
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
plt.yticks([-1, 0, +1])
plt.show()
#틱 라벨 문자열에 $$사이에 LaTeX 수학 문자식을 넣을 수 있음
X = np.linspace(-np.pi, np.pi, 256)
C = np.cos(X)
plt.plot(X, C)
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
plt.yticks([-1, 0, 1], ["Low", "Zero", "High"])
plt.show()

#그리드 설정
X = np.linspace(-np.pi, np.pi, 256)
C = np.cos(X)
plt.plot(X, C)
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
plt.yticks([-1, 0, 1], ["Low", "Zero", "High"])
plt.grid(True)
plt.show()

#여러개 선 그리기
t = np.arange(0., 5., 0.2)
plt.plot(t, t, 'r--', t, 0.5*t**2, 'bs:', t, 0.2*t**3, 'g^-')

#홀드 명령 : 플롯을 합쳐서 그릴 수 있다.
plt.plot([1, 4, 9, 16], c='b', lw=5, ls='--', marker='o', ms=15, mec='g', mew=5, mfc='r')
plt.hold(True)
plt.plot([9, 16, 4, 1], c='k', lw=3, ls=':', marker='s', ms=15, mec='m', mew=5, mfc='c')
plt.hold(False)
plt.show()


#범례
X = np.linspace(-np.pi, np.pi, 256)
C, S = np.cos(X), np.sin(X)
plt.plot(X, C, label='cosine')
plt.hold(True)
plt.plot(X, S, label='sine')
plt.legend(loc=0) #0부터 10까지 있음. 0이 베스트이므로 그냥 0 쓰는게 좋을 듯
plt.show()

#x축, y축 라벨, 타이틀
X = np.linspace(-np.pi, np.pi, 256)
C, S = np.cos(X), np.sin(X)
plt.plot(X, C, label="cosine")
plt.xlabel('time')
plt.ylabel('amplitude')
plt.title('Cosine Plot')
plt.show()

#annotate이용해서 그림 내에 화살표 있게 플롯 그리기. 크게 필요하지 않을 듯
plt.plot(X, S, label="sine")
plt.scatter([0], [0], color="r", linewidth=10) #0, 0을 빨간색으로 칠한다
plt.annotate(r'$(0,0)$', xy=(0, 0), xycoords='data', xytext=(-50, 50), 
             textcoords='offset points', fontsize=16, 
             arrowprops=dict(arrowstyle="->", linewidth=3, color="g"))
plt.show()

#plt.plot은 figure 객체를 생성해 주므로 명시적으로 figure를 만들 필요는 없다.
f1 = plt.figure(figsize=(10, 2))
plt.plot(np.random.randn(100))
plt.show()

#명시적으로 figure 명령을 사용하지 않은 경우, figure객체를 얻으려면 gcf 명령을 사용
f1 = plt.figure(1)
plt.plot([1, 2, 3, 4], 'ro:')
f2 = plt.gcf()
print(f1, id(f1))
print(f2, id(f2))
plt.show()

#Figure라는 큰 틀 안에 axes라는 작은 플롯이 같이 그려져야 하는 경우도 있다. http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes
#원래는 subplot 명령을 이용해서 axes객체를 만들어야 함. 근데 plot만으로 할 수 있음
#subplot은 3개의 인수를 가짐(n, by m, 그중 몇번째)
x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2.0)
y1 = np.cos(2*np.pi*x1) * np.exp(-x1)
y2 = np.cos(2*np.pi*x2)

ax1 = plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'yo-')
plt.title('A tale of 2 subplots')
plt.ylabel('Damped oscillation')
print(ax1)

ax2 = plt.subplot(2, 1, 2)
plt.plot(x2, y2, 'r.-')
plt.xlabel('time (s)')
plt.ylabel('Undamped')
print(ax2)

plt.show()

#2 x 2 형태인 경우
plt.subplot(221); plt.plot([1, 2]); plt.title(1)
plt.subplot(222); plt.plot([1, 2]); plt.title(2)
plt.subplot(223); plt.plot([1, 2]); plt.title(3)
plt.subplot(224); plt.plot([1, 2]); plt.title(4)
plt.tight_layout() #약간 margin을 주는 느낌
plt.show()

#xkcd 스타일
with plt.xkcd():
    plt.title('XKCD style plot!!!')
    plt.plot(X, C, label='cosine')
    t = 2*np.pi/3
    plt.scatter(t, np.cos(t), 50, color='b')
    plt.annotate(r'0.5 Here!', xy=(t, np.cos(t)), xycoords='data', xytext=(-90, -50), 
                 textcoords='offset points', fontsize=16, arrowprops=dict(arrowstyle='->'))
plt.show()




#==============================================================================
# matplotlib의 여러가지 플롯
#==============================================================================
#matplotlib의 한글 적용
#나눔고딕 폰트가 깔려있다면 이렇게 적용해 줄 수 있다. 유니문자열 이용해서 한글 사용해야 함
mpl.rc('font', family='nanumgothic')

#bar chart
#http://matplotlib.org/1.5.1/api/pyplot_api.html#matplotlib.pyplot.bar
#http://matplotlib.org/1.5.1/api/pyplot_api.html#matplotlib.pyplot.barh
y = [2, 3, 1]
x = np.arange(len(y))
xlabel = [u'가', u'나', u'다']
plt.bar(x, y, align='center') #이렇게 해야 tick이 가운데에 오도록 bar가 생성된다.
plt.xticks(x, xlabel)
plt.show()
#xerr, yerr 로 에러 바를 추가할 수 있다.
people = (u'가', u'나', u'다', u'라', u'마')
y_pos = np.arange(len(people))
performance = 3+10*np.random.rand(len(people))
error = np.random.rand(len(people))
plt.barh(y_pos, performance, xerr=error, align='center', alpha=0.4)
plt.yticks(y_pos, people)
plt.xlabel(u'x 라벨')
#두개 bar chart를 한번에 그리는 경우


