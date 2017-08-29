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
# python의 날짜 및 시간 관련 패키지 소개
#==============================================================================
import datetime
dt = datetime.datetime.now()
dt
type(dt)
dt.year #dt[0] 이렇게는 안됨
dt.month
dt.day
dt.hour
dt.minute
dt.second
dt.microsecond
dt.tzinfo
dt.weekday() #일요일의 경우 6

#문자열을 시간으로 변환 / format 변환
dt1 = datetime.datetime.strptime("2015-12-31 11:32", "%Y-%m-%d %H:%M")
dt1
dt1.strftime("%d/%m/%y")
dt1.strftime("%A %d. %B %Y")
#%(Y, m, d, H, M, S) 연 월 일 시 분 초
#s = dt1.strftime(u"%Y년 %m월 %d일 %H시 %M분 %S초".encode('utf-8')) #안되는군...

#시간 쪼개기
dt = datetime.datetime.now()
dt
dt.date(), dt.time()
d = datetime.date(2015, 12, 31)
d
t = datetime.time(11, 31, 29)
t
datetime.datetime.combine(d, t)

#시간 차이 구하기
dt1 = datetime.datetime(2016, 2, 19, 14)
dt2 = datetime.datetime(2016, 1, 2, 13)
td = dt1 - dt2
td #48일하고 3600초 차이
td.days, td.seconds, td.microseconds
td.total_seconds() #모든걸 모아서 초로 계산

#time 패키지
import time
print("start...")
time.sleep(1) # 시간 1초간 지연시키기
print(1)
time.sleep(1)
print(2)
time.sleep(1)
print(3)
time.sleep(1)
print(4)
time.sleep(1)
print("finish!")

time.time()#1970년 1월 1일 0시 기준 초 차이
ts = time.localtime()
ts
time.mktime(ts)

#세계 시간대 변환
import pytz
seoul = pytz.timezone("Asia/Seoul") #서울 객체 생성

#localize 메소드
t1 = datetime.datetime.now()
t1
lt1 = seoul.localize(t1)
lt1
t2 = datetime.datetime.utcnow()
t2
lt2 = pytz.utc.localize(t2)
lt2
lt2 = t2.replace(tzinfo=pytz.utc)
lt2 #뭘 한건지 자세히는 이해 안되지만, 어떤 지역에 대해서 시간을 구하는듯

#타시간대로 변환
t1 = datetime.datetime.now()
lt1 = seoul.localize(t1)
lt3 = lt1.astimezone(pytz.timezone("US/Eastern"))
lt3

#dateutil
from dateutil.parser import parse
parse('2016-04-16')
parse('Apr 16, 2016 04:05:32 PM')
parse('6/7/2016')







#==============================================================================
# Pandas
#==============================================================================
#Series : 시계열 데이터, index를 가지는 1차원 numpy array
#DataFrame : 복수 필드 시계열 데이터 또는 테이블 데이터, index를 가지는 2차원 numpy array
#Index : Label - 각각의 Row/Column에 대한 이름(rownames, colnames), Name - 인덱스 자체에 대한 이름(rec, 변수명set등?)
import pandas as pd
import numpy as np
s = pd.Series([4, 7, -5, 3])
s
s.values
type(s.values)
s.index
type(s.index)

#Vectorized Operation
s * 2
np.exp(s)

#명시적인 index를 가지는 series
#생성시 index 인수로 index 지정. index 원소는 각 데이터에 대한 key역할을 하는 label. dict
s2 = pd.Series([4, 7, -5, 3], index = ['d', 'b', 'a', 'c'])
s2
s2.index

#series indexing 1 : label indexing
#single label
s2['a']
#label slicing
s2['b':'c']
#label을 원소로 가지는 label(label을 list형태로 해서 dict 탐색하는 느낌)
s2[['a', 'b']]

#series indexing 2 : integer indexing
#single label
s2[2]
#label slicing
s2[1:4]
#label을 원소로 가지는 label(label을 list형태로 해서 dict 탐색하는 느낌)
s2[[2, 1]]
#boolean fancy indexing
s2[s2>0]

#dict 연산
'a' in s2, 'e' in s2
#k와 v는 각각 key와 value가 되며, iterable한 형태가 되어서 print 된다.
for k, v in s2.iteritems():
    print(k, v)
s2['d':'a']

#dict 데이터를 이용ㅎ서 series 생성
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
s3 = pd.Series(sdata)
s3
states = ['California', 'Ohio', 'Oregon', 'Texas']
s4 = pd.Series(sdata, index = states) #index 명시해준 데이터를 저장하는데, California는 데이터가 없음
s4
pd.isnull(s4)
pd.notnull(s4)
s4.isnull()
s4.notnull()

#index 기준 연산
print(s3.values, s4.values)
s3.values + s4.values

#index 이름
s4
s4.name = "population" #데이터의 이름 자체를 population이라고 지정
s4
s4.index.name = 'state' #행이름의 대표로 state를 줌
s4

#index 변경
s
s.index
s.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
s
s.index
#연습문제
pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])

#DataFrame
#Multi-Series : 동일한 Row 인덱스를 사용하는 복수 Series. Series를 vlaue로 가지는 dict
#2차원 행렬 : DataFrame을 행렬로 생각하면 각 Series는 행렬의 column에 해당. (Row)index와 column index를 가짐
#Numpy array와 차이점 : 각 column마다 type이 달라도 된다.
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
    'year': [2000, 2001, 2002, 2001, 2002],
    'pop': [1.5, 1.7, 3.6, 2.4, 2.9]
}
df = pd.DataFrame(data = data)
df
pd.DataFrame(data = data, columns = ['year', 'state', 'pop'])
df.dtypes
#index를 가지는 DataFrame
df2 = pd.DataFrame(data = data,
                   columns = ['year', 'state', 'pop', 'debt'],
                   index = ['one', 'two', 'three', 'four', 'five'])
df2
#single column access
df['state']
type(df['state'])
df.state

#column data update
df2['debt'] = 16.5
df2
df2['debt'] = np.arange(5)
df2
df2['debt'] = pd.Series([-1.2, -1.5, -1.7], index = ['two', 'four', 'five'])
df2

#add column
df2['eastern'] = df2.state == 'Ohio'
df2

#delete column
del df2['eastern']
df2

#inplace 옵션
#함수/메소드는 두 가지 종류. - 객체 자체를 변형하는 경우, 객체는 그대로 두고 변형된 새로운 객체 출력하는 경우
#DataFrame 메소드 대부분은 inplace 옵션을 가짐. True를 줄 경우 출력을 None으로 하고 객체 자체를 변형
#False인 경우 객체 자체는 보존하고 변형된 새로운 객체를 출력
x = [3, 6, 1, 4]
sorted(x) #x는 그대로 데이터를 유지한다.
x
x.sort() #x자체가 변형된다.
x

#drop 메소드 이용, row/column 삭제
#del 함수는 inplace 연산 column을 지우는 것이 defualt?
#drop 함수는 inplace False 개념. row를 지우는 것이 defualt
s = pd.Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
s
s2 = s.drop('c')
s2
s
s.drop(['b', 'c'])
df = pd.DataFrame(np.arange(16).reshape((4, 4)),
                   index=['Ohio', 'Colorado', 'Utah', 'New York'],
                   columns=['one', 'two', 'three', 'four'])
df
df.drop(['Colorado', 'Ohio']) #행에서 두 도시를 drop
df.drop('two', axis=1) #column에서 two를 drop

#nested dict를 사용한 dataframe 생성
pop = {
       'Nevada' :{
                  2001:2.4,
                  2002:2.9},
        'Ohio' :{2000:1.5,
                 2001:1.7,
                 2002:3.6}} #안쪽 dict가 행이되고 밖이 col(series)가 된다.
df3 = pd.DataFrame(data = pop)
df3

#Series dict를 사용한 DataFrame 생성
pdata = {
         'Ohio' : df3['Ohio'][:-1],
        'Nevada': df3['Nevada'][:2]}
pd.DataFrame(pdata)

#numpy array로 변환. values만 써주면 바로 그 내용물만 남고 array가 된다. index를 떼낸느낌
df3.values
df2.values

#DataFrame의 column indexing
df2
#single label key
df2['year']
#single label attribute
df2.year
#label list fancy indexing
df2[['state', 'debt', 'year']]
df2[['year']]



#==============================================================================
# Pandas 데이터 입출력
#==============================================================================
import pandas as pd

#csv파일 입력
df = pd.read_csv('ch06/ex1.csv')
df
pd.read_csv('ch06/ex1.csv', names=['a', 'b', 'c', 'd', 'message'])
#특정 column을 index로 지정하고 싶은 경우
pd.read_csv('ch06/csv_mindex.csv', index_col=['key1', 'key2'])
#seperator가 콤마가 아닌경우 명시해준다.
pd.read_table('ch06/ex3.txt', sep='\s+')
#건너뛰어야 할 행이 있으면 skiprows 사용
pd.read_csv('ch06/ex4.csv', skiprows=[0, 2, 3])
#특정한 값을 NA로 하고싶다면 na_values
sentinels = {'message':['foo', 'NA'], 'something':['two']}
pd.read_csv('ch06/ex5.csv', na_values=sentinels)
#일부 행만 읽고 싶다면 nrows 사용
pd.read_csv('ch06/ex6.csv', nrows=5)
#csv파일 출력. 구분자 : |, na 표현, index, header 표현 가능
df.to_csv('ch06/out.csv', sep='|', na_rep='NULL', index=False, header=False)
#인터넷상의 csv 읽기
titanic = pd.read_csv('http://dato.com/files/titanic.csv', index_col=0)
titanic.tail()
#이터넷 상의 데이터 베이스 자료 입력
#여러 데이터를 pandas_datareader패키지의 DataReader로 바로 입력 가능
#Yahoo! Finance, Google Finance, St.Louis FED(FRED), Kenneth French's data library, World Bank, Google Analytics
import pandas_datareader.data as web
import datetime
start = datetime.datetime(2015, 1, 1)
end = datetime.datetime(2016, 6, 30)
df = web.DataReader('005930.KS', 'yahoo', start, end) #005930.KS(삼성전자)라는 회사를 yahoo finance에서 해당 날짜만큼 데이터를 가져온다
df.tail()
df = web.DataReader('KRX:005930', 'google', start, end) #google 데이터
df.tail()
inflation = web.DataReader(["CPIAUCSL", "CPILFESL"], "fred", start, end) #fred 데이터
inflation





#==============================================================================
# Pandas 데이터 변환
#==============================================================================
import pandas as pd
import numpy as np
#applymap 변환
np.random.seed(0)
df = pd.DataFrame(np.random.randn(4,3), columns = list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
df
format = lambda x : '%.2f' % x
df2 = df.applymap(format) #df에게 format이라는 함수를 적용하는 것
df2
df.values.dtype, df2.values.dtype
#apply 변환
df = pd.DataFrame({
                   'Qu1':[1, 3, 4, 3, 4],
                    'Qu2':[2, 3, 1, 2, 3],
                    'Qu3':[1, 5, 2, 4, 4]})
df
f = lambda x: x.max() - x.min()
df.apply(f) #default는 column이다. axis=0
df.apply(f, axis=1)
df.apply(pd.value_counts) #각 값이 몇번씩 나왔는지 알 수 있음. 어떤숫자가 나왔는지는 행을 보면 됨
df.apply(pd.value_counts).fillna(0) #na를 0으로 채운다
#데이터프레임과 시리즈의 연산
#각 행을 같은 크기의 시리즈(벡터)와 연산하면 반복 연산(브로드캐스팅)한다. 열은 transpose 해야됨.
df/df.ix[0]
(df.T/df.T.ix[0]).T
#cut, qcut : 실수 자료를 카테고리 자료로 변환
#cut : bins를 사용자 지정
#qcut : quantile 기준
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100] #구간을 이렇게 나눠줌
cats = pd.cut(ages, bins) #ages가 각각 어떤 구간에 들어가는지 나타냄
cats
cats.categories
cats.codes
pd.cut(ages, [18, 26, 36, 61, 100], right=False)
group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
pd.cut(ages, bins, labels=group_names)
df = pd.DataFrame(ages, columns=['ages'])
df['age_cat'] = pd.cut(df.ages, bins, labels=group_names)
df
data = np.random.randn(1000)
cats = pd.qcut(data, 4) #구간을 4개로 나눠서(quantile) 각각 어디 속하는지 할당
cats
pd.value_counts(cats)
pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.])




#==============================================================================
# Pandas 데이터 합성
#==============================================================================
import pandas as pd
#merge는 두 데이터프레임의 공통 열 혹은 인덱스를 기준으로 데이터 베이스 테이블 조인과 같이 합친다.
df1 = pd.DataFrame({'key':list('bbacaab'), 'data1':range(7)})
df1
df2 = pd.DataFrame({'key':list('abd'), 'data2':range(3)})
df2
pd.merge(df1, df2) #공통 열인 key열을 기준으로 데이터를 찾아서 합친다. 그중에서도 왼쪽 데이터 기준으로 붙임
pd.merge(df1, df2, how='outer') #양쪽 데이터 모두를 보여줌
pd.merge(df1, df2, how='left') #왼쪽 데이터 모두를 보여줌
pd.merge(df1, df2, how='right') #오른쪽 데이터 모두를 보여줌

df1 = pd.DataFrame({'key1':['foo', 'foo', 'bar'],
                    'key2':['one', 'two', 'one'],
                    'lval':[1, 2, 3]})
df2 = pd.DataFrame({'key1':['foo', 'foo', 'bar', 'bar'],
                    'key2':['one', 'one', 'one', 'two'],
                    'rval':[4, 5, 6, 7]})
pd.merge(df1, df2, how='outer')
pd.merge(df1, df2, how='outer', on=['key1', 'key2'])
pd.merge(df1, df2, on='key1') #기준 열이 아니면서 같은 이름이면 _x, _y로 표현된다.
pd.merge(df1, df2, on='key1', suffixes=('_left', '_right')) #suffixes로 표현 가능하다.

df1 = pd.DataFrame({'key1':['foo', 'foo', 'bar'],
                    'key2':['one', 'two', 'one'],
                    'lval':[1, 2, 3]})
df2 = pd.DataFrame({'k1':['foo', 'foo', 'bar', 'bar'],
                    'k2':['one', 'one', 'one', 'two'],
                    'rval':[4, 5, 6, 7]})
pd.merge(df1, df2, left_on='key1', right_on='k1') #각각을 연결해줄 수도 있다.

#인덱스(행)기준으로 사용하려면 left_index, right_index인수를 True로 쓴다.
df1 = pd.DataFrame({'key':list('abaabc'), 'value':range(6)})
df2 = pd.DataFrame({'group_val':[3.5, 7]}, index=['a','b'])
pd.merge(df1, df2, left_on='key', right_index=True) #left의 key값과 right의 index를 결합하는 방식

df1 = pd.DataFrame({'key1': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
                    'key2': [2000, 2001, 2002, 2001, 2002],
                    'data': np.arange(5.)})
df2 = pd.DataFrame(np.arange(12).reshape((6, 2)),
                   index=[['Nevada', 'Nevada', 'Ohio', 'Ohio', 'Ohio', 'Ohio'],
                          [2001, 2000, 2000, 2000, 2001, 2002]],
                   columns=['event1', 'event2'])
pd.merge(df1, df2, left_on=['key1', 'key2'], right_index=True)

df1 = pd.DataFrame([[1., 2.], [3., 4.], [5., 6.]], 
                   index=['a', 'c', 'e'], 
                   columns=['Ohio', 'Nevada'])
df2 = pd.DataFrame([[7., 8.], [9., 10.], [11., 12.], [13, 14]], 
                   index=['b', 'c', 'd', 'e'], 
                   columns=['Missouri', 'Alabama'])
pd.merge(df1, df2, how='outer', left_index=True, right_index=True)

#join함수 이용
df1.join(df2, how='outer')

#concat
#기준열을 사용하지 않고, 단순히 데이터를 추가함.
#기본적으로는 행을 추가. axis=1로 두면 인덱스 기준으로 옆으로 열을 붙임
s1 = pd.Series([0, 1], index=['a', 'b'])
s2 = pd.Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = pd.Series([5, 6], index=['f', 'g'])
pd.concat([s1, s2, s3])
pd.concat([s1, s2, s3], axis=1) #새로 열을 추가하면서 index를 살림

df1 = pd.DataFrame(np.arange(6).reshape(3, 2), index=['a', 'b', 'c'], columns=['one', 'two'])
df2 = pd.DataFrame(5 + np.arange(4).reshape(2, 2), index=['a', 'c'], columns=['three', 'four'])
pd.concat([df1, df2])
pd.concat([df1, df2], axis=1)




#==============================================================================
# Pandas 고급 인덱싱
#==============================================================================
#pandas는 numpy 행렬과 같이 comma를 사용한 복수 인덱싱을 지원하기 위해 특별한 인덱서 속성을 제공한다.
#ix : 라벨과 숫자를 동시에 지원하는 복수 인덱싱
#loc : 라벨 기반의 복수 인덱싱
#iloc : 숫자 기반의 복수 인덱싱
import pandas as pd
#ix 인덱서
#행/열 양쪽에서 라벨 인덱싱, 숫자 인덱싱, 불리언 인덱싱 동시 가능
#,를 사용한 복수 인덱싱
#열도 라벨이 아닌 숫자 인덱싱 가능
#열도 라벨 슬라이싱 가능
#,를 사용하지 않고 단일 인덱싱을 하는 경우, 행 기준 인덱싱
#인덱싱으로 행이나 열을 업데이트하거나 새로 생성 가능
data = {
    'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
    'year': [2000, 2001, 2002, 2001, 2002],
    'pop': [1.5, 1.7, 3.6, 2.4, 2.9]
}
df = pd.DataFrame(data)
#순차적 인덱싱과 동일
df.ix[1:3, ['state', 'pop']]
df2 = pd.DataFrame(data, 
                   columns=['year', 'state', 'pop'],
                   index=['one', 'two', 'three', 'four', 'five'])
df2
#,이용
df2.ix[['two', 'three'], ['state', 'pop']]
#열 숫자로 인덱싱 가능
df2.ix[['two', 'three'], :2]
#열 문자로 슬라이싱 가능
df2.ix[['two', 'three'], 'state':'pop']
#: 사용 가능
df2.ix[['two', 'three'], :]
#, 사용하지 않는 경우 행 인덱싱
df2.ix['two']
#열만 선택하려고할땐 주의
df2.ix[:, 'year']

#index label이 없는 경우의 주의점 - integer slicing을 label slicing으로 간주해 마지막값을 포함한다
df = pd.DataFrame(np.random.randn(5, 3))
df
df.columns=['c1', 'c2', 'c3']
df.ix[0:2, 1:2] #끝값까지 포함됨

###loc 인덱서
#라벨기준 인덱싱
#숫자가 오더라도 라벨로 인식한다.
#라벨 리스트 가능
#라벨 슬라이싱 가능
#불리언 배열 가능

###iloc 인덱서
#숫자기준 인덱싱
#문자열 라벨은 불가
#숫자 리스트 가능
#숫자 슬라이싱 가능
#불리언 배열 가능
np.random.seed(1)
#1부터 11까지 숫자중에 12개 뽑아서 4 by 3 매트릭스 만든다
df = pd.DataFrame(np.random.randint(1, 11, size=(4, 3)), 
                  columns=['A', 'B', 'C'], index=['a', 'b', 'c', 'd'])
df.ix[['a', 'c'], "B":"C"]
df.ix[[0,2], 1:3]
df.loc[['a', 'c'], 'B':"C"]
#,를 사용하지 않는 경우에는 행 인덱싱
df.ix['a']
#열 인덱싱
df.ix[:, 'B']
#df.loc[2:4, 1:3] #오류남. 숫자 슬라이싱 안됨
df.iloc[2:4, 1:3]
#df.iloc[['a', 'c'], 'B':'C'] #오류남. 문자 슬라이싱 안됨

#데이터 프레임의 행/열 합계
#열 합계
np.random.seed(1)
df = pd.DataFrame(np.random.randint(10, size=(4, 8))) #0부터 9에서 32개 뽑음
df
df.sum(axis=1) #행별 합계
df["sum"] = df.sum(axis=1)
df
#행 합계
df.sum()
df.ix['total'] = df.sum()
df





#==============================================================================
# Pandas 인덱스 조작
#==============================================================================
import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randint(1, 10, (10, 4)),
                  columns=['c1', 'c2', 'c3', 'c4'])
df
#df.columns = ['1', '2', '3', '4'] 이렇게 열 이름 바꿔줄 수 있다.
df1 = df.set_index('c1') #c1열을 가지고 rowname으로 만들어줌
df2 = df1.set_index('c2') #c2로 바꿔주면 원래 있던 c1이 사라진다.
df1.reset_index() #만들었던 인덱스를 다시 원복시킴. 맨 앞의 열로 끼워짐
df1.reset_index(drop=True) #만들었던 인덱스를 복원시키는데, 그냥 없애버림

#계층적 인덱스
#데이터프레임 생성시 columns 인수에 리스트의 리스트 형태로 인덱스를 넣으면 계층적 열으로 만들어진다.
np.random.seed(0)
df = pd.DataFrame(np.random.randint(1, 10, (10, 4)),
                  columns=[['A', 'A', 'B', 'B'], ['C1', 'C2', 'C3', 'C4']])
df
df.columns.names = ['cdx1', 'cdx2'] #name이 왜 필요한건지는 이해 불가.

#계층적 행 인덱스
np.random.seed(0)
df = pd.DataFrame(np.random.randint(1, 10, (8, 4)), 
                  columns=[["A", "A", "B", "B"], ["C", "D", "C", "D"]],
                  index=[["M", "M", "M", "M", "F", "F", "F", "F"], ["ID" + str(i) for i in range(4)] * 2])
df.columns.names = ["Cdx1", "Cdx2"]
df.index.names = ["Rdx1", "Rdx2"]
df

#행 인덱스와 열 인덱스 교환
#stack() : 열인덱스 -> (최하위)행인덱스
#unstack() : 행인덱스 -> (최하위)열인덱스
df.stack('Cdx1')
df.stack(0) #숫자인덱스도 가능한데, 이왕이면 이름인덱스로 하는게 좋을듯
df.unstack('Rdx2')
df.unstack(1)


#==============================================================================
# Pandas 피봇과 그룹연산
#==============================================================================
#피봇테이블 : 데이터 열 두개를 키로 사용하여 데이터를 선택하는 방법
import pandas as pd
data = {
    'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
    'year': [2000, 2001, 2002, 2001, 2002],
    'pop': [1.5, 2.5, 3.0, 2.5, 3.5]
}
df = pd.DataFrame(data, columns=['state', 'year', 'pop'])
df
#행, 열 인덱스가 될 자료는 key역할을 해야함. 즉, 유일(unique)하게 결정되야 함
df.pivot('state', 'year', 'pop')
df.pivot('year', 'state', 'pop') #먼저나오는 것이 행 인덱스
#df.pivot('year', 'pop', 'state') #이것은 pop 값이 유일하지 않은것이 있으므로 피봇이 되지 못한다.
df.set_index(['state', 'year']).unstack() #두개 열을 행 인덱스로 겹쳐서 만들고 제일 안쪽(year)를 열인덱스로 바꿔주면 피봇형태가 됨

#그룹연산
#키에 의한 데이터가 여러개 있어도 되며, 이를 aggregate해서 하나의 값으로 만듦. split-apply-combine 연산이라고도 함
#split : key에 따라 데이터 그룹을 만듦
#apply : 각 그룹에 대해 원하는 연산을 적용
#combine : 연산결과를 dic형태로 합침
np.random.seed(0)
df = pd.DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],
                   'key2' : ['one', 'two', 'one', 'two', 'one'],
                   'data1' : np.random.randn(5),
                   'data2' : np.random.randn(5)})
df
#key1에 따른 data1의 평균은?
df.data1.groupby(df.key1).mean() #df의 data1의 mean을 key1으로 groupby
gs = df.data1.groupby(df.key1)
gs #key와 group으로 지정되있음.
print("="*50)
for n, g in gs:
    print("[key]:", n)
    print("[group]:", type(g))
    print("-"*50)
    print(g)
    print("-"*50)
    print("[mean]:", g.mean())
    print("="*50)
gs.mean()
#key1, key2에 따른 data1의 평균은?
means = df.data1.groupby([df.key1, df.key2]).mean()
means
#groupby 명령의 인수 : 열 또는 열의 리스트, 행 인덱스, 사전/함수(column의 값을 사전에 매핑하거나 처리하여 나온 결과값을 키로 인식)
np.random.seed(0)
people = pd.DataFrame(np.random.randn(5, 5), 
                      columns = list('abcde'), 
                        index=['joe', 'steve', 'wes', 'jim', 'travis'])
people.ix[2:3, ['b', 'c']] = np.nan
people
print('='*80)
for n, g in people.groupby(people.index):
    print('[key] :', n)
    print('[group] :', type(g))
    print('-'*80)
    print(g)
    print('='*80)

#mapping을 이용한 groupby
mapping = {'Joe': 'J', 'Jim': 'J', 'Steve': 'S', 'Wes': 'S', 'Travis': 'S'}
print("="*80)
for n, g in people.groupby(mapping):
    print("[key]:", n)
    print("[group]:", type(g))
    print("-"*80)
    print(g)
    print("="*80)    
cap1 = lambda x: x[0].upper() #첫글자만 대문자로 딴 것
print("="*80)
for n, g in people.groupby(cap1):
    print("[key]:", n)
    print("[group]:", type(g))
    print("-"*80)
    print(g)
    print("="*80)
print("="*80)
for n, g in people.groupby(people.columns, axis=1): #열로도 groupby 할 수 있음. 어떤 이름인지와 열을 의미하게 넣어줌
    print("[key]:", n)
    print("[group]:", type(g))
    print("-"*80)
    print(g)
    print("="*80)
mapping = {'a': 'red', 'b': 'red', 'c': 'blue', 'd': 'blue', 'e': 'red', 'f' : 'orange'}
for n, g in people.groupby(mapping, axis=1):
    print("[key]:", n)
    print("[group]:", type(g))
    print("-"*80)
    print(g)
    print("="*80)

#특별한 group 별 연산
#통계 : describe()
#그룹을 대표하는 하나의 값을 계산 : agg(), aggregate()
#대표값으로 필드를 교체 : transform()
#그룹 전체를 변형하는 계산 : apply()
tips = pd.read_csv('ch08/tips.csv')
tips['tip_pct'] = tips['tip'] / tips['total_bill']
tips.tail()
tips.describe() #열별 통계자료가 나온다.
tips.groupby(['sex', 'smoker'])[['tips', 'tip_pct']].describe() #'sex', 'smoker'로 묶은 다음 'tips', 'tip_pct'에 대해 통계자료 산출

#그룹별 연산
gs = tips.groupby(['sex', 'smoker'])
gs_pct = gs['tip_pct']
gs_pct.mean()
gs_pct.agg('mean') #종합하는데, mean 방식으로 종합
def peak_to_peak(arr):
    return(arr.max() - arr.min())
gs_pct.agg(['mean', 'std', peak_to_peak]) #사용자 정의함수도 산출할 수 있다. 나중에 클러스터링 결과물을 종합하는데 유용할듯
gs.agg({'tip_pct':'mean', 'total_bill':peak_to_peak}) #각 변수마다도 다른 함수를 적용할 수 있다.

#그룹의 값을 대표값으로 대체
gs.agg('mean')
tips2 = tips.copy()
tips2['tip2'] = gs.transform('mean')['tip_pct'] #'sex', 'smoker' 단위로 tip_pct 평균낸 값을 tip2에 저장
tips2.tail(15)

#그룹 자체를 대체
#apply는 수치값이 아닌 group을 출력
#단순히 대표값을 계산하는 것 뿐 아니라, 순서 정렬, 일부 삭제 등 그룹 내의 레코드 자체를 변형하는 것도 가능
def top(df, n=5, column='tip_pct'):
    return(df.sort_values(by=column)[-n:]) #column기준으로 값을 sorting하고, 뒤에서 n번째까지 return
top(tips, n=6)
tips.groupby('smoker').apply(top) #각 5개씩 나옴
tips.groupby(['smoker', 'day']).apply(top, n=1, column='total_bill') #그룹마다 각 top 1개씩만 출력
f = lambda x: x.describe()
tips.groupby(['smoker']).apply(f) #람다식도 가능. smoker로 묶은 tips 각 변수(열)에 대해 describe

#pivot_table
#pivot과 groupby을 중간적 성격
#pivot을 수행하지만 데이터가 유니크하게 선택되지 않으면 aggfunc로 정의된 함수를 수행해서 대표값 계산. 디폴트는 평균
tips.pivot_table(index=['sex', 'smoker']) #각 그룹마다 평균으로 내가지고 생성. 위의 pivot에서는 각 value가 유일하지 않으면 안됐음
tips.pivot_table(['tips_pct', 'size'], index=['sex', 'day'], columns='smoker') #열에는 tip_pct, size가 행에는 sez, day가 들어가고, smoker에 대해 내용을 mean(디폴트)로 정리함
tips.pivot_table(['tip_pct', 'size'], index=['sex', 'day'],
                 columns='smoker', margins=True) #margin은 모든 smoker가 가지는 경우를 marginal 해서 보여준다.
tips.pivot_table('tip_pct', index=['sex', 'smoker'], columns='day',
                 aggfunc=len, margins=True) #len으로 함수를 정의
tips.pivot_table('size', index=['time', 'sex', 'smoker'],
                 columns='day', aggfunc='sum', fill_value=0) #없는값은 0으로 채움



