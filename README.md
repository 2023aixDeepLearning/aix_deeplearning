# aix_deeplearning

## Title
  #### Markowitz 모델과 머신러닝을 활용한 포트폴리오 최적화 ####

### Members
  문서진, 컴퓨터소프트웨어학부 2021019834, sj1226m@hanyang.ac.kr
  
  박진수, 컴퓨터소프트웨어학부 2019040591, cookiesoup3409@gmail.com
  
  이윤영, 산업공학과 2018015632,dbsduddl77@gmail.com
  
  이예성, 도시공학과 2018005596, nicelee111@gmail.com
  

### Proposal (Option A)
  #### Motivation: Why are you doing this?
  
`Markowitz모델은 포트폴리오 투자에서 위험성 대비 수익률을 최대화 하는 포트폴리오 비중을 계산하여 포트폴리오 최적화하는 수학적 모델이다. Markowitz 모델의 최적화 과정에서 목적함수에 다음 시점의 기대수익의 예측값이 포함된다. 이때 주로 사용하는 기대수익의 예측값으로 머신러닝, 딥러닝 기법을 사용하여 투자를 진행하면 좋은 결과가 나올지 확인해보려 하였다.` 
  
  #### What do you want to see at the end?
  
`머신러닝, 딥러닝 기법으로 도출한 예상 수익률과 Markowitz 모델을 활용하여 포트폴리오 투자를 시뮬레이션한다. FinanceDateReader 모듈로 실제 코스피 주식 데이터를 불러오고, 특정 기간을 주기로 포트폴리오 비중을 조정하는 리밸런싱을 반복한다. 이때 포트폴리오 투자를 마치는 시점에서의 수익률과, 이 투자방법의 위험성을 확인하고 적합한 투자 방법인지 확인한다.`

### Datasets  
파이썬의 FinanceDataReader 모듈을 이용해 코스피에 상장된 주식 데이터를 가져온다.
#### 1. 필요한 모듈을 import 한다. ####
```python
!pip install -U finance-datareader
import FinanceDataReader as fdr
```
```
Collecting finance-datareader
  Downloading finance_datareader-0.9.66-py3-none-any.whl (23 kB)  
  
...  

Installing collected packages: requests-file, finance-datareader
Successfully installed finance-datareader-0.9.66 requests-file-1.5.1  
```
#### 2. 파이썬의 FinanceDataReader 모듈을 이용해 코스피에 상장된 주식 데이터를 가져온다. ####
```python
stock_name = fdr.StockListing('KOSPI')['Name'].to_list()
stock_code = fdr.StockListing('KOSPI')['Code'].to_list()

print(stock_name[0:20])
print(stock_code[0:20])
```
```
['삼성전자', 'LG에너지솔루션', 'SK하이닉스', '삼성바이오로직스', '삼성전자우', 'POSCO홀딩스', '현대차', '기아', 'LG화학', 'NAVER', '삼성SDI', '포스코퓨처엠', '셀트리온', '카카오', '삼성물산', '현대모비스', 'KB금융', '신한지주', 'LG전자', '삼성생명']
['005930', '373220', '000660', '207940', '005935', '005490', '005380', '000270', '051910', '035420', '006400', '003670', '068270', '035720', '028260', '012330', '105560', '055550', '066570', '032830']
```
#### 3. 코스피 주식 중 2000년 이전에 상장된 주요 주식들을 선정하였다. ####
```python
my_portfolio = ['삼성전자', 'SK하이닉스','POSCO홀딩스', '현대차', '기아', '삼성SDI', '현대모비스', 'LG', '카카오', 'SK텔레콤', '기업은행', 'S-Oil', 'KT']
len(my_portfolio)
```
```
13
```
#### 4. 주식의 종가를 바탕으로 포트폴리오 투자를 진행하기 위해 각 주식의 2000년 이후의 종가 데이터를 가져온다. ####
```python
stock_dict = dict(zip(stock_name, stock_code))

stock_df = pd.DataFrame()

for stock in my_portfolio:
    stock_df[stock] = fdr.DataReader(stock_dict[stock], '2000-01-01', '2023-01-01')['Close']


stock_df
```
<img width="888" alt="스크린샷 2023-12-06 오후 2 50 31" src="https://github.com/2023aixDeepLearning/aix_deeplearning/assets/80944952/6b83fe72-0885-4d92-8cb6-c4dbc22956b2">

#### 5. 10 영업일 주기로 포트폴리오 리밸런싱할 것을 고려하여, 10 영업일 주기의 데이터를 가져온다. 영업일은 주말과 휴일을 제외한 기간으로 매수, 매도가 가능한 기간이다. 10 영업일은 약 2주이다. ####
```python
df = stock_df.iloc[::10,:]
df
```
<img width="883" alt="스크린샷 2023-12-06 오후 2 50 37" src="https://github.com/2023aixDeepLearning/aix_deeplearning/assets/80944952/ad8272c7-5d28-45c9-98cf-3da8f518c1d7">

#### 6. 각 종목 별 총 기간의 평균 수익과 변동성을 확인한다. ####
```python
def get_returns(result):
  ans = [0]
  for i in range(1, len(result)):
    ans.append((result[i]-result[i-1])/result[i-1])
  return ans
def get_mean_return(returns):
  return sum(returns)/len(returns)
def get_risk(returns):
  n = len(returns)
  mean = sum(returns)/n
  ans = 0
  for i in range(n):
    ans += (returns[i]-mean)**2
  return (ans/n)**(1/2)

mean_return_of_each_asset = []
risk_of_each_asset = []

for i in my_portfolio:

  ret = get_returns(list(df[i]))
  mean_return_of_each_asset.append(get_mean_return(ret))
  risk_of_each_asset.append(get_risk(ret))

pd.DataFrame({'mean_returns':mean_return_of_each_asset, 'Volatility':risk_of
```
<img width="276" alt="스크린샷 2023-12-06 오후 2 52 11" src="https://github.com/2023aixDeepLearning/aix_deeplearning/assets/80944952/5d2000ef-a0ac-43fc-a6b1-9a2cad9666e0">

### Methodology
> ARIMA, XGBoost, GRU세가지 방식으로 포트폴리오를 예측하여 가장 성능이 좋은 모델을 선정하였다.

#### ARIMA (AutoRegressive Integrated Moving Average) 모델
   - 주로 시계열 데이터 예측에 사용되며, 시계열 데이터에서 과거 관측된 값들을 사용하여 현재 또는 미래의 값을 예측할 수 있다.
   - 비정상 시계열에도 적용할 수 있다는 특징을 가진다.
     - 시계열 (time series) 데이터: 일정 시간 간격으로 배치된 데이터 
     - 정상 시계열 데이터: 시간에 상관없이 일정한 성질을 띠는 시계열 데이터
   - ARIMA 모델: AR(p) 모델과 MA(q) 모델에 d차 차분이 추가된 모델
     - AR(p) 모델: p차수의 자기 회귀 모델로 과거 값의 선형 조합을 이용하여 값을 예측한다.
     - MA(q) 모델: q차수의 이동 평균 모델로, 과거 예측 오차를 이용하여 값을 예측한다.
     - I(d): d차수의 차분을 나타낸다. 원래의 시계열 데이터를 예측과 분석을 쉽게 하기 위해 정상 시계열 데이터로 변환하는 과정이다.
       - 차분 (Differencing): 연이은 관측값들의 차이를 계산해 시계열 수준에서 나타나는 변화를 제거하여, 시계열의 평균 변화를 일정하게 만든다.
      
   - ARIMA model development process
![ARIMA-model-development-process-ARIMA-autoregressive-integrated-moving-average](https://github.com/2023aixDeepLearning/aix_deeplearning/assets/80944952/5778ab60-7d8e-4c19-a791-4315d3095f3f)
    
#### XGBoost(eXtra Gradient Boost)
  - XGBoost(eXtra Gradient Boost)는 기존 Gradient Tree Boosting 알고리즘에 과적합 방지를 위한 기법이 추가된 지도 학습 알고리즘이다.
  - XGBoost 알고리즘은 기본 학습기(Base Learner)를 의사결정 나무로 하며 Gradient Boosting과 같이 Gradent(잔차)를 이용하여 이전 모형의 약점을 보완하는 방식으로 학습을 진행한다.

![image](https://github.com/2023aixDeepLearning/aix_deeplearning/assets/149667956/4c3382d8-1112-40ac-9ce1-c097227222dc)

  - 분류에 있어 일반적으로 다른 머신러닝보다 뛰어난 예측 성능을 나타내며, 병렬 CPU 환경에서 병렬 학습이 가능해 기존 GBM보다 빠르게 학습을 완료할 수 있다는 특징을 가진다.
  - Boosting은 약한 학습기들을 순차적으로 학습시켜 가중치를 부여하여 강력한 학습기를 만드는 방법이며, 원리는 m1~3 모델이 있을때, m1에는 x에서 샘플링된 데이터를 넣는다. 그리고, 나온 결과중에서, 예측이 잘못된 x중의 값들에 가중치를 반영해서 다음 모델인 m2에 넣는다.  마찬가지로 y2 결과에서 예측이 잘못된 x’에 값들에 가중치를 반영해서 m3에 넣는다. 그리고, 각 모델의 성능이 다르기 때문에, 각 모델에 가중치 W를 반영한다. 이를 개념적으로 표현하면 다음 그림과 같다[2].

    <img width="380" alt="다운로드" src="https://github.com/2023aixDeepLearning/aix_deeplearning/assets/149667956/5a8c2a1c-b7f5-4859-b0e9-68515e140216">
  - 각 과정에 등장하는 알고리즘 수식은 [2]의 자료를 참고하였다.
  - 기본적으로 결정 트리(Decision tree), 렌덤 포레스트 등의 기반을 가지고 있기 때문에 'n_estimators:결정 트리 개수', 'max_depth:트리 깊이', 'learning_rete:학습률' 등의 하이퍼파리마터를 갖는다.

#### GRU (Gated Recurrent Unit) 모델
<p align="center">
  <img 
    src="https://github.com/2023aixDeepLearning/aix_deeplearning/assets/54359232/244cc10e-1537-4a11-a7f6-a893fa61043e" 
    width="300"
    />
</p>

   - GRU는 LSTM의 장기 의존성 문제(은닉층의 정보가 마지막까지 전달되지 못하던 기존 RNN의 문제점)에 대한 해결책을 유지하면서, 은닉 상태를 업데이트하는 계산을 줄여 연산 속도를 개선했다.
   - ##### LSTM과 GRU의 차이점
     - LSTM에는 출력, 입력, 삭제를 담당하는 3개의 게이트가 존재한다.
     - GRU의 경우, 업데이트와 리셋을 담당하는 2개의 게이트만 존재한다.
     - 따라서 LSTM 대비 학습 속도는 빠르면서, 비슷한 성능을 보인다.
   - Reset Gate(리셋 게이트)는 이전 은닉 상태(h<sub>t-1</sub>)를 얼마나 잊을지를 결정한다.
   - Update Gate(업데이트 게이트)는 이전 은닉 상태(h<sub>t-1</sub>)와 새로운 정보(x<sub>t</sub>)간의 균형을 결정한다.
   - 경험적으로 데이터의 양이 적은 경우 GRU, 데이터의 양이 많은 경우 LSTM을 더 선호한다. (매개변수의 개수 차이)
   - 텐서플로우의 케라스의 경우 아래와 같이 GRU의 구현을 지원한다.
     - ```python
       model.add(GRU(hidden_size, input_shape=(timesteps, input_dim)))
       ```

### Code
#### 1. 필요한 모듈을 가져온다. ####
```python
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from cvxopt.solvers import qp
from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import minimize
import xgboost as xgb
from keras.models import Sequential
from keras.layers import GRU, Dense
from IPython.display import clear_output
```
#### 2. 시계열 데이터 예측을 위한 예시 데이터를 생성한다. 0 시점부터 199 시점까지의 데이터를 가진다. ####
```python
import random
random.seed(123)

time_series_data = []
for i in range(200):
  time_series_data.append(3*np.sin(i + random.random())+20+2*random.random())

print('Example data')
plt.plot(time_series_data)
```
![KakaoTalk_Photo_2023-12-06-14-54-25 001jpeg](https://github.com/2023aixDeepLearning/aix_deeplearning/assets/80944952/1bd4de6c-7af5-46c0-af96-bd100980aed7)

#### 3. ARIMA 모델을 통한 시계열 예측 ####
   - 처음 시점부터 t 시점까지의 데이터를 이용해 t+1 시점의 값을 예측하는 과정을 반복한다. 이렇게 180시점부터 199시점 까지의 값을 예측한다.
```python
order = (2,1,2)

ARIMA_preds = []
for i in range(180,200):
  model = ARIMA(time_series_data[0:i], order=order)
  res = model.fit()
  prediction = res.forecast(steps=1)
  ARIMA_preds.append(prediction)

plt.plot(time_series_data[180:], label = 'Actual')
plt.plot(ARIMA_preds, label = 'ARIMA_prediction')
plt.legend()
plt.show()
```
![KakaoTalk_Photo_2023-12-06-14-54-26 002jpeg](https://github.com/2023aixDeepLearning/aix_deeplearning/assets/80944952/e1d596bb-a995-4f5e-965d-1f5522f180e2)

#### 4. XGBoost를 통한 시계열 예측 ####
   - t, t-1, ..., t-4 시점까지의 값을 X_train의 각각의 feature로 지정하고, t+1 시점의 값을 y로 지정하여 예측한다.
```python
def create_sequence(data, seq_length):
  sequence = [data[i:i+seq_length+1] for i in range(len(data)-seq_length+1+1)]

  df = pd.DataFrame(sequence)

  return df.dropna()


def XGB_AutoRgression(data, seq_length, XGB_params):
    seq_df = pd.DataFrame(create_sequence(data, seq_length))

    X_train = seq_df.iloc[:-1,1:]
    y_train = seq_df.iloc[:-1,0]
    X_test = seq_df.iloc[-1:,1:]

    model = xgb.XGBRegressor(**XGB_params)
    model.fit(X_train, y_train)
    pred_y = model.predict(X_test)
    return pred_y

XGB_parameters = {
    'n_estimators': 30,
    'learning_rate': 0.1,
    'max_depth': 5,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0,
    'reg_lambda': 1}

XGB_preds = []
for i in range(180,200):
  XGB_preds.append(XGB_AutoRgression(time_series_data[:i], 5, XGB_parameters))

plt.plot(time_series_data[180:], label = 'Actual')
plt.plot(XGB_preds, label = 'XGB_prediction')
plt.legend()
plt.show()
```
![KakaoTalk_Photo_2023-12-06-14-54-26 003jpeg](https://github.com/2023aixDeepLearning/aix_deeplearning/assets/80944952/0633b6c9-ca89-48c1-8999-dae84c7d0fe5)

#### 5. GRU를 통한 시계열 예측 ####
  - GRU 모델을 구축한다.
```python
from keras.models import Sequential
from keras.layers import GRU, Dense

GRU_model = Sequential()
GRU_model.add(GRU(8, input_shape=(5, 1)))
GRU_model.add(Dense(10))
GRU_model.add(Dense(1))
GRU_model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
```
  - sequence를 생성하고 GRU_model에 적합시킨다.
```python
def create_sequence_for_GRU(data, seq_length):
  sequences = []
  for i in range(len(data) - seq_length):
      seq = data[i:i + seq_length]
      target = data[i+seq_length : i+seq_length+1]
      sequences.append((seq, target))
  return np.array(sequences)


def GRU(data, seq_length, GRU_model):

  sequences = create_sequence_for_GRU(data, seq_length)

  train_sequences = sequences[:-1]
  test_sequences = sequences[-1:]

  X_train = np.array([item[0] for item in train_sequences])
  y_train = np.array([item[1] for item in train_sequences])
  X_test = np.array([item[0] for item in test_sequences])

  model = GRU_model
  model.fit(X_train, y_train, epochs= 5, batch_size=5, verbose=0)

  pred_y = model.predict(X_test)
  return pred_y

GRU_preds = []
for i in range(180,200):
  pred = GRU(time_series_data[:i], 5, GRU_model)
  GRU_preds.append(pred[0][0])
```

```python
plt.plot(time_series_data[180:], label = 'Actual')
plt.plot(GRU_preds, label = 'GRU_prediction')
plt.legend()
plt.show()
```
![KakaoTalk_Photo_2023-12-06-14-54-26 004jpeg](https://github.com/2023aixDeepLearning/aix_deeplearning/assets/80944952/9c0895b7-efce-4a1b-b880-5067dcf83f71)

#### 6. 포트폴리오 클래스 생성 ####
   - 포트폴리오 최적화는 Markowitz Model에 따라 진행한다.

주가 정보의 데이터프레임을 이용하여 2000년 1월(0행)부터 2016년 2월(399행) 까지의 return 데이터프레임을 생성한다.

return은 '(전 시점의 주가 - 현 시점의 주가)/전 시점의 주가' 로 계산한다. 이때 초기(0행)의 return은 0이다.
```python
def return_mat(df, n):   # 주가 matrix를 return matrix로 변환

  new_df = {}
  for i in df.columns:
    new_df[i] = [0][:]

  new_df = pd.DataFrame(new_df)
  for i in range(1,n):
    X0 = df.iloc[i-1,:]
    X1= df.iloc[i,:]
    new_df.loc[i] = (X1-X0)/X0
  return new_df

return_matrix = return_mat(df.iloc[:400,:], 400)
return_matrix.index = df.index[0:400]
return_matrix
```

현 시점을 2016년 2월 23일(399행)이라고 가정하자.

이 데이터프레임을 이용하여 다음 시점인 2016년 3월 9일(400행)의 Covariance Matrix와 return을 예측한다.

이때 Covariance Matrix는 390행부터 399행까지의 공분산 행렬을 이용하고, return은 길이가 5인 이동평균으로, 395행부터 399행까지의 평균을 사용하였다.
```python
Cov = matrix(np.array(return_matrix.iloc[max(0,len(return_matrix)-10):,:].cov()))
Mean = return_matrix.iloc[-5:].mean()

print(f'Covariance matrix : \n{Cov}')
print(f'Expected return of each asset : \n{Mean}')
```
```
Covariance matrix : 
[ 3.07e-03  1.21e-03  1.81e-03  1.92e-03  2.02e-03  1.37e-03  2.23e-03 ... ]
[ 1.21e-03  4.58e-03  3.36e-03  1.06e-03  3.31e-04  1.13e-03  5.68e-04 ... ]
[ 1.81e-03  3.36e-03  5.65e-03  1.55e-03  1.02e-03  7.49e-04  1.34e-03 ... ]
[ 1.92e-03  1.06e-03  1.55e-03  3.49e-03  3.18e-03  4.92e-04  1.75e-03 ... ]
[ 2.02e-03  3.31e-04  1.02e-03  3.18e-03  3.91e-03  9.64e-04  1.68e-03 ... ]
[ 1.37e-03  1.13e-03  7.49e-04  4.92e-04  9.64e-04  6.65e-03  9.51e-04 ... ]
[ 2.23e-03  5.68e-04  1.34e-03  1.75e-03  1.68e-03  9.51e-04  1.92e-03 ... ]
[ 1.32e-03  1.87e-04  1.99e-03 -2.73e-04  1.12e-04  2.93e-03  9.38e-04 ... ]
[-8.62e-04 -4.43e-05 -1.66e-03 -2.29e-03 -1.91e-03 -9.47e-04 -1.18e-03 ... ]
[ 1.03e-03  2.63e-03  4.13e-03  1.40e-03  7.73e-04  1.73e-03  6.09e-04 ... ]
[ 2.36e-03  2.28e-03  3.06e-03  2.10e-03  2.20e-03  2.34e-03  1.82e-03 ... ]
[ 1.10e-03  9.68e-04  2.30e-04 -4.92e-04 -1.24e-03  3.03e-03  6.06e-04 ... ]
[ 7.23e-04  1.70e-03  2.37e-03  1.92e-04 -7.29e-05  1.58e-03  3.94e-04 ... ]
```
```
Expected return of each asset : 
삼성전자       -0.012186
SK하이닉스      0.000537
POSCO홀딩스    0.037246
현대차         0.000523
기아         -0.020686
삼성SDI      -0.038873
현대모비스       0.016833
LG          0.004097
카카오        -0.039271
SK텔레콤      -0.001693
기업은행       -0.017452
S-Oil       0.013416
KT         -0.002602
dtype: float64
```

expected return, covariance를 이용하여 포트폴리오 최적화를 진행한다. 이를통해 최적의 포트폴리오 비중을 계산한다.

변수:

&nbsp;&nbsp;&nbsp;&nbsp;weights : 각 주식별 포트폴리오 비중의 벡터 [w1, w2, ...]

&nbsp;&nbsp;&nbsp;&nbsp;initial_weight : 최적화 진행 과정의 초기 해 벡터

&nbsp;&nbsp;&nbsp;&nbsp;risk_free_rate : 무위험 자산에 투자했을 때의 기대 수익률

&nbsp;&nbsp;&nbsp;&nbsp;mean_returns : 기대 수익률

&nbsp;&nbsp;&nbsp;&nbsp;cov_matrix : 공분산행렬

목적함수:

&nbsp;&nbsp;&nbsp;&nbsp;minimize -(∑(weights*mean_returns)-risk_free_rate)/(weight.T​·cov_matrix​·weights)^(1/2)

제약조건:

&nbsp;&nbsp;&nbsp;&nbsp;weights_i >= 0, ∀ i (모든 주식에 대해 공매도를 진행하지 않는다.)

&nbsp;&nbsp;&nbsp;&nbsp;∑weights = 1 (포트폴리오 비중의 합은 1이다.)

```python
def calculate_portfolio_return(weights, mean_returns):
    return np.sum(weights * mean_returns)

def calculate_portfolio_risk(weights, cov_matrix):
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_stddev

def objective_function(weights, mean_returns, cov_matrix, risk_free_rate):
    portfolio_return = calculate_portfolio_return(weights, mean_returns)
    portfolio_risk = calculate_portfolio_risk(weights, cov_matrix)
    return (portfolio_return - risk_free_rate) / portfolio_risk

initial_weights = [1/len(df.columns)] * len(df.columns)
risk_free_rate = 0


constraints = (
    {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},  # 가중치의 합 = 1
    {'type': 'ineq', 'fun': lambda weights: weights}  # 비음조건
)
result = minimize(
    lambda weights: -objective_function(weights, Mean, Cov, risk_free_rate),
    initial_weights, method='SLSQP', constraints=constraints
)

ratio = result.x
portpolio_ratio_of_each_asset = pd.DataFrame(ratio, index = my_portfolio, columns = ['ratio'])
print(f'portfolio ratio of each stock : \n {portpolio_ratio_of_each_asset}')
```
```
portfolio ratio of each stock : 
                  ratio
삼성전자     -3.769811e-16
SK하이닉스   -4.603129e-17
POSCO홀딩스  4.492443e-01
현대차       1.763176e-16
기아       -3.478727e-16
삼성SDI    -5.402379e-16
현대모비스     3.310565e-01
LG        1.950632e-17
카카오       5.714205e-16
SK텔레콤    -5.390244e-16
기업은행     -9.160123e-16
S-Oil     2.196992e-01
KT        1.037297e-16
```

위에서 구한 주식별 비중을 통해 현 시점(2016-02-23)에 리벨런싱 할 주식의 개수를 구한다. 현 시점 가용 금액은 1천만원이고, 포트폴리오 비중과 현 시점의 주가를 고려한 주식의 개수를 계산한다.

밑의 예시에서는 포스코홀딩스 15주, 현대모비스 16주, S-Oil 25주를 현재의 포트폴리오로 한다.

```python
current_asset = 10000000
def invest_num(ratio, data, asset):  # 포트폴리오 비중, 자산 현황에 따른 주식 개수 반환
  ratio = [max(0, x) for x in ratio]
  num_of_stock=[]
  for i in range(len(ratio)):
    num_of_stock.append(asset*ratio[i]//data[i])
  return num_of_stock
num_of_stocks = invest_num(ratio, np.array(df.iloc[-1:,:])[0], current_asset)
number_of_each_stock = pd.DataFrame(num_of_stocks, index = my_portfolio, columns = ['Number of stock'])
print(f'number of stock by portfolio ratio : \n{number_of_each_stock}
```

```
number of stock by portfolio ratio : 
          Number of stock
삼성전자                  0.0
SK하이닉스                0.0
POSCO홀딩스             15.0
현대차                   0.0
기아                    0.0
삼성SDI                 0.0
현대모비스                16.0
LG                    0.0
카카오                   0.0
SK텔레콤                 0.0
기업은행                  0.0
S-Oil                25.0
KT                    0.0
```

위 결과에 따라 포트폴리오 리밸런싱을 하고, 다음 시점인 2016년 3월 9일 주가변동에 따른 자산의 변화를 계산한다.

이 예시에서는 1000만원이 1017만6500원으로 변하였고, 이때의 수익률은 1.756% 이다.

```python
current_stock_price = df.iloc[399,:]
next_stock_price = df.iloc[400,:]

next_asset = current_asset
for i in range(len(df.columns)):

  next_asset = next_asset+ (next_stock_price[i]-current_stock_price[i])*num_of_stocks[i]
print(f'current asset : {current_asset}')
print(f'changed asset after 1 period : {next_asset}')
print(f'portfolio return during the period : {(next_asset-current_asset)/current_asset*100}%')
```
current asset : 10000000
changed asset after 1 period : 10176500.0
portfolio return during the period : 1.765%

같은 기간동안 모든 주식에 동일한 비중으로 투자했을 때의 결과(-1.33%)보다 더 좋은 결과가 나왔음을 확인

```python
actual_return = (current_stock_price-next_stock_price)/current_stock_price
print(f'actual return of each stock : \n{actual_return}')
print(f'mean of actual returns : \n{sum(actual_return)/len(actual_ret
```
```
actual return of each stock : 
삼성전자       -0.011008
SK하이닉스     -0.004934
POSCO홀딩스   -0.069825
현대차        -0.003356
기아         -0.019792
삼성SDI      -0.033092
현대모비스       0.023211
LG          0.055178
카카오        -0.118249
SK텔레콤       0.061281
기업은행       -0.025424
S-Oil      -0.030788
KT          0.003466
dtype: float64
mean of actual returns : 
-1.3333237596424146%
```

return matrix를 구하는 과정부터 expected return을 구하고 최적화를 거쳐 자산의 변화를 거치는 과정을 반복하는 class를 생성한다.

이 class는 위에서 진행했던 현 시점의 포트폴리오를 설정하고 다음 시점의 자산의 변화를 확인하는 일련의 과정을 반복한다.

이때 expected return(r_bar)은 이동평균, ARIMA, XGBoost, GRU 중에 선택할 수 있다. ARIMA는 order (p, d, q)를 지정할 수 있고, XGBoost는 hyperparameter를 지정할 수 있다. GRU는 모델을 미리 설정하고 그 모델을 이용하여 예측한다.

```python
class Markowitz_model:
  def __init__(self, params):
    self.params = params
    try:
      if params['r_bar'] == 'XGB':
        self.XGB_params = hyperparameters = {
        'n_estimators': 50,
        'learning_rate': 0.1,
        'max_depth': 5,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0,
        'reg_lambda': 1}
      elif params['r_bar'] == 'ARIMA':
        self.order = (2,1,2)
    except: pass

  def invest_num(self, ratio, data, asset):  # 포트폴리오 비중, 자산 현황에 따른 주식 개수 반환
    ratio = [max(0, x) for x in ratio]
    num_of_stock=[]
    for i in range(len(ratio)):
      num_of_stock.append(asset*ratio[i]//data[i])
    return num_of_stock

  def return_mat(self, df, n):   # 주가 matrix를 return matrix로 변환
    df2 = df.copy(deep=True)
    new_df = {}
    for i in df.columns:
      new_df[i] = [0][:]

    new_df = pd.DataFrame(new_df)

    for i in range(1,n):
      X0 = df2.iloc[i-1,:]
      X1= df2.iloc[i,:]
      new_df.loc[i] = (X1-X0)/X0
    return new_df

##############################

  def arima(self, data):
    model = ARIMA(data, order = self.order)
    res = model.fit()
    return res.forecast(steps=1)

  def create_sequence(self, data, seq_length):
    sequence = [data.iloc[i:i+seq_length+1].tolist() for i in range(len(data)-seq_length+1+1)]

    # 시퀀스 데이터프레임 생성
    df = pd.DataFrame(sequence)

    return df

  def XGB_AR(self, data, seq_length):

    seq_df = pd.DataFrame(self.create_sequence(data, seq_length))

    X_train = seq_df.iloc[:-1,1:]
    y_train = seq_df.iloc[:-1,0]
    X_test = seq_df.iloc[-1:,1:]
    y_test = seq_df.iloc[-1:,0]

    model = xgb.XGBRegressor(**self.XGB_params)
    model.fit(X_train, y_train)
    pred_y = model.predict(X_test)
    return pred_y

  def create_sequence_for_GRU(self, data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i + seq_length:i + seq_length + 1]
        sequences.append((seq, target))
    return np.array(sequences)

  def GRU(self, data, seq_length):

    sequences = self.create_sequence_for_GRU(data, seq_length)

    train_sequences = sequences[:-1]
    test_sequences = sequences[-1:]

    X_train = np.array([item[0] for item in train_sequences])
    y_train = np.array([item[1] for item in train_sequences])
    X_test = np.array([item[0] for item in test_sequences])
    y_test = np.array([item[1] for item in test_sequences])

    model = self.GRU_model
    model.fit(X_train, y_train,  epochs= 5, batch_size=8, verbose=0)
    pred_y = model.predict(X_test)
    return pred_y


  def get_mean(self, r_bar, return_data, n_columns, len_return):
    if r_bar == 'ARIMA':
      Mean = []
      for j in range (n_columns):
        Mean.append(float(self.arima(return_data.iloc[:,j])))
      Mean = matrix(Mean)
    elif r_bar[0:2] == 'MA':
      MA_len = int(r_bar[2:])
      if MA_len >= len(return_data):
        Mean = matrix(list(return_data.mean()*len_return/(len_return-1)))
      else:
        Mean = matrix(list(return_data.iloc[-MA_len:,:].mean()))
    elif r_bar == 'XGB':
      Mean = []
      for j in range (n_columns):
        Mean.append(float(self.XGB_AR(return_data.iloc[:,j], 5)))
      Mean = matrix(Mean)

    elif r_bar == 'GRU':
      Mean = []
      for j in range (n_columns):
        Mean.append(float(self.GRU(return_data.iloc[:,j], 5)))
      Mean = matrix(Mean)


    else:
      Mean = matrix(list(return_data.mean()*len_return/(len_return-1)))
    return Mean



#################################

  def Markowitz_max_sharp(self, df, r_bar, risk_free_rate, initial_invest, cov_len, start_point):

    asset = initial_invest
    asset_list = [asset]*(start_point-1)
    n_rows = len(df)    # number of rows
    columns = df.columns
    n_columns = len(columns)   # number of columns


    current_invest_num = [0]*n_columns
    return_df = self.return_mat(df, len(df))
    for i in range(start_point,n_rows):
      return_data = return_df.iloc[:i+1,:]
      data = df.iloc[:i+1,:]
      prior_stock_price=[]
      current_stock_price=[]


      for j in range(len(columns)):
        prior_stock_price.append(data.iloc[i-1,j])
        current_stock_price.append(data.iloc[i,j])

        asset = asset+ (current_stock_price[j]-prior_stock_price[j])*current_invest_num[j]  # 주가 변동에 따른 자산변화
      asset_list.append(asset)


      n = len(columns)
      len_return = len(return_data)

      Cov = matrix(np.array(return_data.iloc[max(0,len(return_data)-cov_len):-1,:].cov()))

      Mean = self.get_mean(r_bar, return_data, n_columns, len_return)

      def calculate_portfolio_return(weights, mean_returns):
          return np.sum(weights * mean_returns)

      def calculate_portfolio_risk(weights, cov_matrix):
          portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
          return portfolio_stddev

      def objective_function(weights, mean_returns, cov_matrix, risk_free_rate):
          portfolio_return = calculate_portfolio_return(weights, mean_returns)
          portfolio_risk = calculate_portfolio_risk(weights, cov_matrix)
          return (portfolio_return - risk_free_rate) / portfolio_risk

      initial_weights = [1/n_columns] * n_columns
      risk_free_rate = risk_free_rate


      constraints = (
          {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},  # 가중치의 합 = 1
          {'type': 'ineq', 'fun': lambda weights: weights}  # 비음조건
      )
      result = minimize(
          lambda weights: -objective_function(weights, Mean, Cov, risk_free_rate),
          initial_weights, method='SLSQP', constraints=constraints
      )

      ratio = result.x

      invest_num_data=[]
      for j in range(len(columns)):
        invest_num_data.append(data.iloc[i,j])
      current_invest_num = self.invest_num(ratio, invest_num_data, asset) # 주식 개수 업데이트
      print(f'{(i-start_point+1)/(n_rows-start_point)*100}%')
      clear_output(wait=True)


    for j in range(len(columns)):
      prior_stock_price.append(data.iloc[i-1,j])
      current_stock_price.append(data.iloc[i,j])

      asset = asset+ (current_stock_price[j]-prior_stock_price[j])*current_invest_num[j]
    asset_list.append(asset)

    return asset_list



  def simulate(self, df):

    params = {'r_bar' : 'MA5',
              'risk_free_rate' : 0.0025,
              'initial_invest' : 100000000,
              'cov_len' : 10,
              'start_point' : (int(len(df)*0.7)//1)}


    if self.params == None:
      pass
    else:
      params = {**params, **self.params}
    risk_free_rate = params['risk_free_rate']
    r_bar = params['r_bar']
    initial_invest = params['initial_invest']
    cov_len = params['cov_len']
    start_point = params['start_point']

    if params['r_bar'] == 'ARIMA':
      self.order = params['ARIMA_order']

    elif params['r_bar'] == 'XGB':
      self.XGB_params == params['XGB_params']

    elif params['r_bar'] == 'GRU':
      self.GRU_model = params['GRU_model']


    result = self.Markowitz_max_sharp(df, r_bar, risk_free_rate, initial_invest, cov_len, start_point)

    return result[start_point-1:]
```

model1은 기대 수익 예측에 ARIMA 모델을 사용한다. 여기서 사용하는 ARIMA 모델의 (p, d, q)의 값은 (2, 1, 2)이다.

risk_free_rate는 주식에 투자가 아닌 은행에 넣었을 때의 예상 수익률을 사용한다. 연간 시중금리를 3퍼센트로 가정하면 1 기간(10 영업일) 동안의 risk free rate는 약 0.125퍼센트 이다.

cov_len은 최적화 식의 공분산으로 사용할 값의 길이를 지정한다. cov_len을 10으로 설정하면 10 기간 동안의 공분산을 사용한다.

start_point는 투자를 시작하는 시점을 지정한다. 현재 데이터프레임에서의 start_point가 400일 경우, 2016년 2월 23일 부터 2022년 말까지 투자하는 것을 시뮬레이션한다.

```python
params1 = {'r_bar' : 'ARIMA',
           'ARIMA_order' : (2, 1, 2),
           'risk_free_rate' : 0.00125,
           'cov_len' : 10,
           'start_point' : 400}

model1 = Markowitz_model(params1)
result1 = model1.simulate(df[my_portfolio])
```
```
100.0%
```

model2는 model1과 다른 조건을 동일하게 지정하고, 기대수익의 예측값에 XGBoost를 사용한다. 이전 5 기간의 수익률을 각각의 feature로 지정하고, 자기회귀의 결과를 예측값 한다.

```python
XGB_parameters = {
    'n_estimators': 30,
    'learning_rate': 0.1,
    'max_depth': 5,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0,
    'reg_lambda': 1}


params2 = {'r_bar' : 'XGB',
           'XGB_params' : XGB_parameters,
           'risk_free_rate' : 0.00125,
           'cov_len' : 10,
           'start_point' : 400
           }

model2 = Markowitz_model(params2)
result2 = model2.simulate(df[my_portfolio])
```
```
100.0%
```

model3는 기대수익의 예측값으로 순환신경망 모델인 GRU를 사용한다. 이때 input_shape를 (5, 1)로 지정하였다.

```python
from keras.models import Sequential
from keras.layers import GRU, Dense

GRU_model = Sequential()
GRU_model.add(GRU(8, input_shape=(5, 1)))
GRU_model.add(Dense(10))
GRU_model.add(Dense(1))
GRU_model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

params3 = {'r_bar' : 'GRU',
           'GRU_model' : GRU_model,
           'risk_free_rate' : 0.00125,
           'cov_len' : 10,
           'start_point' : 400
           }

model3 = Markowitz_model(params3)
result3 = model3.simulate(df[my_portfolio])
```
```
1/1 [==============================] - 0s 21ms/step
1/1 [==============================] - 0s 21ms/step
1/1 [==============================] - 0s 22ms/step
1/1 [==============================] - 0s 26ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 22ms/step
1/1 [==============================] - 0s 21ms/step
1/1 [==============================] - 0s 21ms/step
1/1 [==============================] - 0s 26ms/step
1/1 [==============================] - 0s 31ms/step
1/1 [==============================] - 0s 28ms/step
1/1 [==============================] - 0s 21ms/step
1/1 [==============================] - 0s 21ms/step
100.0%
```


### Evaluation & Analysis  
각 모델에서 포트폴리오의 수익률을 확인한다. total return은 (마지막 시점의 자산 총량 - 첫 시점의 자산 총량)/첫 시점의 자상 총량의 퍼센트값으로 사용한다. 예를들어 초기 투자 금액이 1000원, 마지막 기간의 자산 총액이 2000원이면 수익률은 100%가 된다.

-  return의 예측값으로 ARIMA 사용
총 수익률은 139.57퍼센트로, 2016년 2월 23일에 1억을 투자했을 경우 2022년 말에 약 2억3956만원이 된다.
```python
returns1 = get_returns(result1)
plt.plot(returns1)
plt.plot([0]*len(result1))
plt.show()
print(f'mean return by model1 = {sum(returns1)/len(returns1)*100}%')
print(f'risk by model1 = {get_risk(returns1)*100}%')
print(f'total return by model1 = {(result1[-1]-result1[0])/result1[0]*
```
<img width="578" alt="스크린샷 2023-12-06 오후 3 11 11" src="https://github.com/2023aixDeepLearning/aix_deeplearning/assets/80944952/ca3b7c7c-72f1-4ced-9e7a-b8cedc637039">

-  return의 예측값으로 XGBoost 사용
총 수익률은 160.92퍼센트로, 2016년 2월 23일에 1억을 투자했을 경우 2022년 말에 약 2억6092만원이 된다.
```python
returns2 = get_returns(result2)
plt.plot(returns2)
plt.plot([0]*len(result2))
plt.show()
print(f'mean return by model2 = {sum(returns2)/len(returns2)*100}%')
print(f'risk by model2 = {get_risk(returns2)*100}%')
print(f'total return by model2 = {(result2[-1]-result2[0])/result2[0]
```
<img width="577" alt="스크린샷 2023-12-06 오후 3 11 13" src="https://github.com/2023aixDeepLearning/aix_deeplearning/assets/80944952/99a70b0b-cdf9-45ee-9be7-fb9b39f49cce">

-  return의 예측값으로 GRU 사용
총 수익률은 191.27퍼센트로, 2016년 2월 23일에 1억을 투자했을 경우 2022년 말에 약 2억9127만원이 된다.
```python
returns3 = get_returns(result3)
plt.plot(returns3)
plt.plot([0]*len(result3))
plt.show()
print(f'mean return by model3 = {sum(returns3)/len(returns3)*100}%')
print(f'risk by model3 = {get_risk(returns3)*100}%')
print(f'total return by model3 = {(result3[-1]-result3[0])/result3[0]*10
```
<img width="573" alt="스크린샷 2023-12-06 오후 3 11 15" src="https://github.com/2023aixDeepLearning/aix_deeplearning/assets/80944952/4cea88fc-5dc3-4066-8a98-883b3192f050">

### Conclusion  
동일한 기간 동안 모든 주식에 동일한 비중으로 투자하고 유지했을 때의 결과이다. 총 수익률(91.978%)이 Markowitz 모델과 ARIMA, XGBoost, GRU 방법을 활용했을 때보다 낮게 나왔음을 확인할 수 있다.
```python
return_df = return_mat(df,len(df))
returns4 = [0]
result4 = [100000000]
for i in range (400, len(df)):
  returns4.append(sum(return_df.iloc[i,:])/len(return_df.columns))
  result4.append(result4[i-400]*(1+returns4[i-399]))
plt.plot(returns4)
plt.plot([0]*len(returns4))
plt.show()
print(f'mean return by model4 = {sum(returns4)/len(returns4)*100}%')
print(f'risk by model4 = {get_risk(returns4)*100}%')
print(f'total return by model4 = {(result4[-1]-result4[0])/result4[0]*100}%
```
<img width="586" alt="스크린샷 2023-12-06 오후 3 11 17" src="https://github.com/2023aixDeepLearning/aix_deeplearning/assets/80944952/15bcc2f2-c488-4966-a9b2-2fbe19821635">

### 참고 문헌
[1] Peng Chuyin, Mo Zhengma, Zhang Xinyu, A Study of Wordle Reported Outcome Data Based on ARIMA-XGBoost Model, IEEE, 774-779 Aug, 2023  
[2] Linlin Zhao, Jasper Mbachu, Huirong Zhang, Forecasting residential building costs in New Zealand using a univariate approach, IJEBM, Volume 11:1-13, 2019  
[3] https://zephyrus1111.tistory.com/232#c1  
[4] https://bcho.tistory.com/1354  
[5] https://xgboost.readthedocs.io/en/latest/parameter.html  
