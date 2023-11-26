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

### Methodology
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

Reference
Peng.Chuyin, A Study of Wordle Reported Outcome Data Based on ARIMA-XGBoost Model
