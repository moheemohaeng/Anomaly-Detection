# 1. 서론
전자상거래와 디지털 금융 거래가 활발한 현대사회에서 신용 카드는 일상생활에 없어서는 안 될 존재이다. 하지만 신용 카드를 통한 거래가 많아짐에 따라 금융 사기에 노출되지 않도록 신용 카드 거래 데이터를 분석하여 비정상적인 거래를 탐지할 필요가 있다.
본 프로젝트는 고객들의 카드 사용 데이터를 기반으로 다양한 이상탐지 알고리즘을 학습시키고 성능을 평가한다. 최종적으로는 본 데이터에 적합한 최적의 이상 탐지 알고리즘을 선정한다.


# 2. 문제 설명
신용카드 이상거래는 신용카드를 사용한 금융 거래중 일반적인 사용 패턴에서 벗어나거나 의심으러운 활동으로 감지되는 거래를 말한다. 이러한 이상거래는 사기와 부정거래의 조짐이 될 수 있으며, 이로인하여 개인 고객과 금융사에 경제적 피해를 유발한다. 본 논문에서는 신용카드 이상거래에 대한 데이터셋을 호가보한 후 해당 데이터 셋에 여러 이상탐지 알고리즘을 적용하여 이상 데이터를 식별하고 분석하고자 한다. 각 모델들에 대해 나타난 성능 수치들을 바탕으로 각 모델들에 대해 ablation study를 진행하여 최적의 모델을 선정한다.

## 2.1 데이터 정의 및 소개
본 프로젝트에 사용된 데이터는 "kaggle"에서 제공하고 있으며, 데이터 셋에 대한 정의 및 소개는 다음과 같다.
실험에 사용된 데이터는 2013년 9월 유럽의 카드 소지자가 신용카드로 결제한 거래정보로 구성되어있다. 이 데이터 셋은 이틀 동안 발생한 거래를 보여주며, 이상 거래는 전체 거래의 0.172%를 차지하는 불균형한 데이터 셋이다.
기밀 문제로 인해 원래 데이터의 features와 자세한 배경정보를 제공하지 않는다. 따라서 개인정보에 민감한 독립 변수는 PCA로 변환하여 제공된다.
Feature 'Class'는 종속 변수로, 이상거래인 경우 1, 그렇지 않은 경우 0의 값을 갖는다.

## 2.2 EDA 및 전처리
이상탐지 모델 적용 전, 탐색적 데이터 분석을 진행하였다. 결과는 아래와 같다.
> 데이터 수 : 284,807 (정상 : 284,315/ 이상 : 492)
> 결측치 : 0
> 독립 변수 : 다중공선성 파악을 위한 변수 간 상관관계분석을 진행
모델 학습과 분류 모델의 성능 평가는 Confusion Matrix의 f1_score로 진행하였다. 이상탐지 모델의 학습을 위한 train set은 정상 데이터로만 구축하였고, 이상탐지 성능 평가를 위한 test set은 정상과 이상 데이터의 비율을 동일하게 설정하여 구축하였다.
train set의 데이터를 정상에 가깝게 구성하여 모델의 강건한 학습을 시키기 위해 ESD와 사분위수 방식을 사용하여 set에 포함된 이상치를 제거하였다.


# 3. 제안 방법론
우리는 이상 탐지에 있어서 밀도, 재구축 기반의 다양한 이상탐지 알고리즘들을 적용하였고, 해당 모델들에 대해 ablation study를 진행하였으며, 다수의 모델에서 우수한 성능 수치를 획득하였다. 활용한 모델들에 대한 설명은 아래와 같다.

## 3.1 IF / DIF
-Isolation Forest
여러 개의 의사결정나무를 종합한 앙상블 기반의 이상탐지 기법으로 의사결정나무를 지속적으로 분기시키며 모든 데이터 관측치의 고립 정도 여부에 따라 이상치를 판별하는 방법이다.
디만 이러한 방법론은 고차원적이거나 비선형 분리가 가능한 데이터 공간에서 분리하기 어려운 이상 현상을 탐지하지 못한다는 단점이 있다.
-Deep Isolation Forest
비교적 Shallow한 모델인 Isolation Forest에서 나타나는 이상점 분할의 제약점에 대해서 해결하기 위해 representation ensemble로 mapping하는 과정에 있어 Neural Network를 활용하였으며 실험 결과, 기존의 Isolation Forest에 비해 성능이 상당 수준 개선되었다.

## 3.2 AE / VAE
-AutoEncoder
AutoEncoder는 비지도 학습 모델이며 입력과 출력층의 뉴런 수가 동일한 신경망 모델이다. 입력 데이터의 가장 중요한 특성을 학습한다는 특징이 있다.   입력과 출력의 차이가 동일한 구조로 대칭적인 구조이며, 모델의 출력값과 입력값이 비슷해지도록 학습이 수행된다. 즉, 복원을 수행했을 시, 기존의 값과 차이가 큰 값을 이상치로 간주한다. 
-Variation AutoEncoder
VAE 는 Overfitting을 피하고 latent space가 새로운 데이터를 생성할 수 있도록 좋은 특성들을 가지도록 훈련하는 AutoEncoder이다. 특히 AutoEncoder는 과적합이 되면 데이터를 한 점으로 압축하는데, 따라서 irregular한 latent space가 나온다. 따라서 VAE는 데이터를 하나의 점이 아닌 여러 개의 정규분포로 나타낸다. 이를 정규분포에서 나온 임의의 input과 연산하여 잠재 벡터로 계산하며, 해당 벡터를 다시 데이터로 복원하며 모델을 학습시킨다.

## 3.3 OCSMV / DeepSVDD
-OCSVM
OCSVM은 SVM을 한 클래스에 속하는 데이터만을 이용하여 학습시킨다. 정상치를 고차원 공간에서 초평면으로 분리하고, 새로운 데이터가 이 초평면에서 멀리 떨어져있다면 이를 이상치로 간주한다.
-DeepSVDD
Deep SVDD는 OCSVM과 마찬가지로 정상 데이터로만 모델을 학습시키는 비지도 학습 알고리즘이다. 입력 데이터를 저차원으로 매핑하는 신경망을 사용하기 때문에 고차원에서 초평면을 찾기 힘든 OCSVM의 문제를 해결한다. 매핑된 정상 데이터를 잘 설명하는 중심을 찾고, 이 중심으로부터 거리가 먼 데이터를 이상치로 간주하게 된다.

## 3.4 LOF / CBLOF
-LOF
LOF는 밀도기반 이상 탐지 모델로, 각각의 지점에 대해서 이웃과의 밀도 편차를 계산하여 이상 여부를 판단한다. 일반적으로 고차원 데이터에서 강건한 성능을 보이며, 데이터에 대해 특정 분포를 가정하지 않는다는 특징이 있다. 하지만 밀도기반 알고리즘 특성상 고차원에서 연산량이 많다.
-CBLOF
CBLOF는 LOF에 clustering을 도입한 것으로, 데이터에서 clusters를 먼저 탐색한 후에 LOF를 적용하여 각각의 지점에 대한 점수를 계산한다. 데이터의 밀도 분포가 다양할 때 강건한 성능을 보이며, LOF보다 높은 성능을 기대할 수 있다. 하지만 LOF와 마찬가지로 고차원 데이터에서 연산량이 많아짐에 따라 비용이 클 수 있다.

## 3.5 LUNAR
기존의 이상 탐지 방법들을 GNN기반으로 합친 모델이다. 기존의 LOF, DBSCAN과 같은 Local outlier method는 간단한 알고리즘으로 고차원 데이터와 같은 실용적인 문제에 해결할 수 있는 장점이 있다. 하지만 학습을 위한 매개변수가 없고, Feature based data에 대해 심층적인 학습을 할 수 없다는 단점이 있다. 
LUNAR는 이런 문제점을 해결하고 이상 탐지를 효율적으로 하기 위해 GNN에서 활용한 메시지 패싱 체계를 기반으로 가까운 이웃들에 대한 정보를 학습한다. 특히 학습 가능한 매개변수를 적용하여 타 모델들에 비해 좀 더 강건한 모델을 구축하였고, 데이터가 서로 다른 이웃들의 분포를 가질 때 다른 이상 탐지 모델보다 성능이 뛰어난 장점이 있다.


![캡처](https://github.com/user-attachments/assets/0909dce4-e516-4be4-983a-eece122aef49)

# 4. 실험 결과 및 결론
대부분의 이상탐지 알고리즘이 개선된 알고리즘에서 더 좋은 성능을 보였다. 대부분의 모델이 본 데이터 셋에 대해서 준수한 성능을 보였다. 
추가된 모델로, LUNAR는 모든 성능 지표에 대해 좋은 결과가 나타났다. 대부분 일반적인 신경망 기반을 사용하여 개선된 모델과 다르게 GNN의 메시지 패싱 체계 기반에 학습 가능한 매개변수를 적용하였기에 데이터의 특성을 효과적으로 반영하였다고 볼 수 있다.
최종적으로 이상탐지 모델 접근으로 비정상적 카드 거래를 탐지하여 금융 범죄를 조기에 예방하고, 금융 생활의 안정화를 도모할 수 있을 것이라 기대하는 바이다.

