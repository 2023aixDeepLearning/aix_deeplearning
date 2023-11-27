# aix_deeplearning

## Title
  Markowitz 모델과 머신러닝을 활용한 포트폴리오 최적화

### Members
  문서진, 컴퓨터소프트웨어학부 2021019834, sj1226m@hanyang.ac.kr
  
  박진수, 컴퓨터소프트웨어학부 2019040591, cookiesoup3409@gmail.com
  
  이윤영, 산업공학과 2018015632,dbsduddl77@gmail.com
  
  이예성, 도시공학과 2018005596, nicelee111@gmail.com
  

### Proposal (Option A)
  -Motivation: Why are you doing this?
  
    머신러닝, 딥러닝 모델을 통해 주식 시계열 데이터를 분석하여 포트폴리오를 모델링해보고, 투자 전략 과정에서 머신러닝을 활용하는 것이 과연 적합할 것인가에 대해 알아보고 싶다.
  
  -What do you want to see at the end?
  
    다양한 주식 시계열 데이터를 input 데이터로 활용하여, 리스크 대비 수익률이 가장 높은 최적의 포트폴리오를 도출해 내는것이 목적이다.

### Datasets  
파이썬의 FinanceDataReader 모듈을 이용해 코스피에 상장된 주식 데이터를 가져온다.
- 데이터 구조
  - ...

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
1. 필요한 모듈을 가져온다.
```
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
2. 주식 데이터를 가져온다.
   - FinanceDataReader 사용을 위해 설치 후 import 한다.
```
!pip install -U finance-datareader
import FinanceDataReader as fdr
```
  -  코스피 주식 중 2000년 이전에 상장된 주식들을 선정하였다.
```
stock_name = fdr.StockListing('KOSPI')['Name'].to_list()
stock_code = fdr.StockListing('KOSPI')['Code'].to_list()

print(stock_name)
print(stock_code)
```
  - 주식의 종가를 바탕으로 포트폴리오 투자를 진행하기 위해 각 주식의 2000년 이후의 종가 데이터를 가져온다.
```
my_portfolio = ['삼성전자', 'SK하이닉스','POSCO홀딩스', '현대차', '기아', '삼성SDI', '현대모비스', 'LG', '카카오', 'SK텔레콤', '기업은행', 'S-Oil', 'KT']
len(my_portfolio)
```
```
stock_dict = dict(zip(stock_name, stock_code))

stock_df = pd.DataFrame()

for stock in my_portfolio:
    stock_df[stock] = fdr.DataReader(stock_dict[stock], '2000-01-01', '2023-01-01')['Close']


stock_df
```
  - 10 근무일 주기로 포트폴리오 리밸런싱할 것을 고려하여, 10근무일 주기의 데이터를 가져온다. 근무일은 주말과 휴일을 제외한 기간으로 매수, 매도가 가능한 기간이다. 10 근무일은 약 2주정도이다.
```
df = stock_df.iloc[::10,:]
df
```
```
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
```
  - 각 종목별 총 기간의 평균 수익과 변동성을 확인한다.
```
mean_return_of_each_asset = []
risk_of_each_asset = []

for i in my_portfolio:

  ret = get_returns(list(df[i]))
  mean_return_of_each_asset.append(get_mean_return(ret))
  risk_of_each_asset.append(get_risk(ret))

print(mean_return_of_each_asset)
print(risk_of_each_asset)
```
```
pd.DataFrame({'mean_returns':mean_return_of_each_asset, 'Volatility':risk_of_each_asset},index=my_portfolio)
```
3. 포트폴리오 클래스 생성
   - 포트폴리오 최적화는 마코비츠 모델에 따라 진행한다.
```
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

    num_of_stock=[]
    for i in range(len(ratio)):
      num_of_stock.append(asset*ratio[i]//data[i])
    return num_of_stock

  def return_mat(self, df, n):   # 주가 matrix를 return matrix로 변환
    df2 = df.copy(deep=True)
    new_df = df.iloc[0:1,:]

    new_df.loc[0] = [0]*len(df.columns)
    for i in range(1,n):
      X0 = df2.iloc[i-1,:]
      X1= df2.iloc[i,:]
      new_df.loc[i] = (X1-X0)/X0
    return new_df

  def create_sequence(self, data, seq_length):
    sequence = [data.iloc[i:i+seq_length+1].tolist() for i in range(len(data)-seq_length+1+1)]

    # 시퀀스 데이터프레임 생성
    df = pd.DataFrame(sequence)

    return df

##############################

  def arima(self, data):
    model = ARIMA(data, order = self.order)
    res = model.fit()
    return res.forecast(steps=1)

  def create_sequence_for_GRU(self, data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i + seq_length:i + seq_length + 1]
        sequences.append((seq, target))
    return np.array(sequences)



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

  def GRU(self, data, seq_length):

    sequences = self.create_sequence_for_GRU(data, seq_length)

    train_sequences = sequences[:-1]
    test_sequences = sequences[-1:]

    X_train = np.array([item[0] for item in train_sequences])
    y_train = np.array([item[1] for item in train_sequences])
    X_test = np.array([item[0] for item in test_sequences])
    y_test = np.array([item[1] for item in test_sequences])

    model = self.GRU_model
    model.fit(X_train, y_train, verbose=0)
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


  def Holding(self, df, startpoint):
    asset = self.initial_invest

    first_price = []
    last_price = []
    asset_dividen = [asset/len(df.columns)]*len(df.columns)
    num_of_stock = []

    for i in range (len(df.columns)):
      first_price.append(df.iloc[startpoint,:][i])
      last_price.append(df.iloc[len(df)-1,:][i])
      num_of_stock.append(asset_dividen[i]//first_price[i])
      asset_dividen[i] = asset_dividen[i]+(last_price[i]-first_price[i])*num_of_stock[i]

    return sum(asset_dividen)



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
4. 모델 생성
model1은 다음 기간의 기대수익의 값으로 5 기간 동안의 이동평균(Moving Average) 값을 사용한다.

risk_free_rate는 주식에 투자가 아닌 은행에 넣었을 때의 예상 수익률을 사용한다. 연간 시중금리를 3퍼센트로 가정하면 1 기간(10 영업일) 동안의 risk free rate는 약 0.125퍼센트 이다.

cov_len은 최적화 식의 공분산으로 사용할 값의 길이를 지정한다. cov_len을 10으로 설정하면 10 기간 동안의 공분산을 사용한다.

start_point는 투자를 시작하는 시점을 지정한다. 현재 데이터프레임에서의 start_point가 400일 경우, 2016년 3월부터 2022년 말까지 투자하는 것을 시뮬레이션한다.
```
params1 = {'r_bar' : 'MA5',
           'risk_free_rate' : 0.00125,
           'cov_len' : 10,
           'start_point' : 400}

model1 = Markowitz_model(params1)
result1 = model1.simulate(df[my_portfolio])
```
model2는 model1과 다른 조건을 동일하게 지정하고, 기대 수익 예측에 ARIMA 모델을 사용한다. 여기서 사용하는 ARIMA 모델의 (p, d, q)의 값은 (1, 1, 1)이다.
```
params2 = {'r_bar' : 'ARIMA',
           'ARIMA_order' : (1, 1, 1),
           'risk_free_rate' : 0.00125,
           'cov_len' : 10,
           'start_point' : 400}

model2 = Markowitz_model(params2)
result2 = model2.simulate(df[my_portfolio])
```
model3는 기대수익의 예측값에 XGBoost를 사용한다. 이전 5 기간의 수익률을 각각의 feature로 지정하고, 자기회귀의 결과를 예측값으로 사용한다.
```
hyperparameters = {
    'n_estimators': 30,
    'learning_rate': 0.1,
    'max_depth': 5,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0,
    'reg_lambda': 1}


params3 = {'r_bar' : 'XGB',
           'XGB_params' : hyperparameters,
           'risk_free_rate' : 0.00125,
           'cov_len' : 10,
           'start_point' : 400
           }

model3 = Markowitz_model(params3)
result3 = model3.simulate(df[my_portfolio])
```
model4는 기대수익의 예측값으로 순환신경망 모델인 GRU를 사용한다. 이때 input_shape를 (5, 1)로 지정하였다.
```
GRU_model = Sequential()
GRU_model.add(GRU(32, input_shape=(5,1)))
GRU_model.add(Dense(16))
GRU_model.add(Dense(1))
GRU_model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

params4 = {'r_bar' : 'GRU',
           'GRU_model' : GRU_model,
           'risk_free_rate' : 0.00125,
           'cov_len' : 10,
           'start_point' : 400
           }

model4 = Markowitz_model(params4)
result4 = model4.simulate(df[my_portfolio])
```
5. 각 모델에서 포트폴리오 수익률 확인
total return은 (마지막 시점의 자산 총량 - 첫 시점의 자산 총량)/첫 시점의 자상 총량의 퍼센트값으로 사용한다. 예를들어 초기 투자 금액이 1000원, 마지막 기간의 자산 총액이 2000원이면 수익률은 100%가 된다.
```
returns1 = get_returns(result1)
plt.plot(returns1)
plt.plot([0]*len(result1))
plt.show()
print(f'mean return by model1 = {sum(returns1)/len(returns1)}')
print(f'risk by model1 = {get_risk(returns1)}')
print(f'total return by model1 = {(result1[-1]-result1[0])/result1[0]*100}%')

returns2 = get_returns(result2)
plt.plot(returns2)
plt.plot([0]*len(result2))
plt.show()
print(f'mean return by model2 = {sum(returns2)/len(returns2)}')
print(f'risk by model2 = {get_risk(returns2)}')
print(f'total return by model2 = {(result2[-1]-result2[0])/result2[0]*100}%')

returns3 = get_returns(result3)
plt.plot(returns3)
plt.plot([0]*len(result3))
plt.show()
print(f'mean return by model3 = {sum(returns3)/len(returns3)}')
print(f'risk by model3 = {get_risk(returns3)}')
print(f'total return by model3 = {(result3[-1]-result3[0])/result3[0]*100}%')

returns4 = get_returns(result4)
plt.plot(returns4)
plt.plot([0]*len(result4))
plt.show()
print(f'mean return by model4 = {sum(returns4)/len(returns4)}')
print(f'risk by model4 = {get_risk(returns4)}')
print(f'total return by model4 = {(result4[-1]-result4[0])/result4[0]*100}%')

model1_result = pd.DataFrame({'model1' : result1})
model2_result = pd.DataFrame({'model1' : result2})
model3_result = pd.DataFrame({'model1' : result3})
model4_result = pd.DataFrame({'model1' : result4})
```
각 모델을 통해 변화한 자산의 정보를 csv파일로 각각 저장한다.
```
from google.colab import drive
drive.mount('/content/drive')

model1_result.to_csv('/content/drive/My Drive/model1_result.csv', index=False)
model2_result.to_csv('/content/drive/My Drive/model2_result.csv', index=False)
model3_result.to_csv('/content/drive/My Drive/model3_result.csv', index=False)
model4_result.to_csv('/content/drive/My Drive/model4_result.csv', index=False)
```
### Evaluation & Analysis  

### Conclusion  


### 참고 문헌
[1] Peng Chuyin, Mo Zhengma, Zhang Xinyu, A Study of Wordle Reported Outcome Data Based on ARIMA-XGBoost Model, IEEE, 774-779 Aug, 2023  
[2] Linlin Zhao, Jasper Mbachu, Huirong Zhang, Forecasting residential building costs in New Zealand using a univariate approach, IJEBM, Volume 11:1-13, 2019  
[3] https://zephyrus1111.tistory.com/232#c1  
[4] https://bcho.tistory.com/1354  
[5] https://xgboost.readthedocs.io/en/latest/parameter.html  
