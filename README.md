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
      
   - Step 1
     - p, d, q 추정 

#### XGBoost(eXtra Gradient Boost)
  - XGBoost(eXtra Gradient Boost)는 기존 Gradient Tree Boosting 알고리즘에 과적합 방지를 위한 기법이 추가된 지도 학습 알고리즘이다.
  - XGBoost 알고리즘은 기본 학습기(Base Learner)를 의사결정 나무로 하며 Gradient Boosting과 같이 Gradent(잔차)를 이용하여 이전 모형의 약점을 보완하는 방식으로 학습을 진행한다.
![image](https://github.com/2023aixDeepLearning/aix_deeplearning/assets/149667956/4c3382d8-1112-40ac-9ce1-c097227222dc)

  - 분류에 있어 일반적으로 다른 머신러닝보다 뛰어난 예측 성능을 나타내며, 병렬 CPU 환경에서 병렬 학습이 가능해 기존 GBM보다 빠르게 학습을 완료할 수 있다는 특징을 가진다.
  - Boosting은 약한 학습기들을 순차적으로 학습시켜 가중치를 부여하여 강력한 학습기를 만드는 방법이며, 원리는 m1~3 모델이 있을때, m1에는 x에서 샘플링된 데이터를 넣는다. 그리고, 나온 결과중에서, 예측이 잘못된 x중의 값들에 가중치를 반영해서 다음 모델인 m2에 넣는다.  마찬가지로 y2 결과에서 예측이 잘못된 x’에 값들에 가중치를 반영해서 m3에 넣는다. 그리고, 각 모델의 성능이 다르기 때문에, 각 모델에 가중치 W를 반영한다. 이를 개념적으로 표현하면 다음 그림과 같다[2].

    <img width="380" alt="다운로드" src="https://github.com/2023aixDeepLearning/aix_deeplearning/assets/149667956/5a8c2a1c-b7f5-4859-b0e9-68515e140216">
  - 각 과정에 등장하는 알고리즘 수식은 [2]의 자료를 참고하였다.
  - 기본적으로 결정 트리(Decision tree), 렌덤 포레스트 등의 기반을 가지고 있기 때문에 'n_estimators:결정 트리 개수', 'max_depth:트리 깊이', 'learning_rete:학습률' 등의 하이퍼파리마터를 갖는다.

    
### 참고 문헌
[1] https://zephyrus1111.tistory.com/232#c1
[2] https://bcho.tistory.com/1354
[3] https://xgboost.readthedocs.io/en/latest/parameter.html
