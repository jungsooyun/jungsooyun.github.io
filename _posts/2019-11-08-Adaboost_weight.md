---
title: Adaboost의 가중치 이해하기
search: true
categories:
 - Data Science
tags:
 - Boosting
last_modified_at: 2019-11-08 01:51
---

이번 포스트에서는 부스팅 알고리즘 중, Adaptive Boosting (AdaBoost)의 가중치 유도 과정을 [Rojas, R. (2009)](http://www.inf.fu-berlin.de/inst/ag-ki/adaboost4.pdf)와 The Elements of Statistical Learning(ESL)을 바탕으로 다룹니다.

부스팅 알고리즘은 정형 데이터에서 높은 분류, 예측 성능을 보이기 때문에 현재도 Kaggle 등에서 활발하게 사용되고 있으며, 그 부스팅 알고리즘의 시조격인 Adaboost 알고리즘의 첫 제안은 [다음의 논문](https://www.sciencedirect.com/science/article/pii/S002200009791504X) 에서 이루어 졌습니다. 이 논문의 저자인 Yoav Freund와 Robert Schapire는 Adaboost를 개발한 공로로 괴델상을 수상하였습니다.

AdaBoost는 weak model (약한 학습기, ex. Decision Tree)의 학습 결과와 그 오차를 바탕으로 어떻게 더 나은 예측 결과를 만들 수 있을지에 대한 아이디어를 제시해 줍니다. 여기에서는 쉬운 유도를 위해 두 개의 클래스만 존재하는 이진 분류 (binary classification) 문제를 가정하겠습니다.

AdaBoost는 각각의 weak model들을 학습시킨 후, 해당 모델이 분류에 성공한 데이터와 실패한 데이터를 구분한 다음, 분류에 실패를 데이터에 더 높은 가중치을 두어 다음 weak model의 학습에 고려되도록 만듭니다. 그렇다면, 그 가중치는 어떤 값을 주어야 할까요?

이진 타겟 변수 (1 or -1)를 맞추는 AdaBoost Classifier의 가중치 공식을 유도해봅시다.

각 input 값 $x_i$에 대응되는 label $y_i \in { -1,1 }$을 가지고 있는 데이터 ${ (x_1,y_1),...,(x_N,y_N) }$ 에 대해 각각의 weak classifier를 $k_j$ 라 합시다. L개의 weak classifier가 존재한다고 할 때, j 번째 weak classifier에 의해 데이터가 분류된 결과를 다음과 같이 표현할 수 있습니다.
 $$ k_j(x_i) \in \{ -1,1 \}$$

 그리고, $m-1$번째까지 업데이트된 Boosting Model $C_{m-1}$을 다음과 같이 정의합니다. (GAM : Generative Additive Model)

 $$C_{m-1}(x_i)=\alpha_1k_1(x_i)+\cdots+\alpha_{m-1}k_{m-1}(x_i) $$

 어렵게 생각하지 말고, i번째 데이터에 대해 첫 번째 weak classifier의 가중치가 0.1이고 예측값이 True, 두번째 모델의 가중치는 0.4이고 예측값이 False, 세번째 트리는 0.5의 가중치를 가지고 예측값이 True 인 경우,

 $$C_3(x_i) =0.1\cdot 1+0.4\cdot (-1)+0.5\cdot 1=0.2$$

 라고 할 수 있습니다. 이 것을 부호(sign) 함수에 넣으면 그 값이 0보다 크므로 $C_3(x_i)$는 $x_i$를 True라고 분류하게 됩니다.

이제 $C_{m-1}(x)$에 weak classifier $k_m(x_i)$를 추가로 학습시켜  $C_{m}(x_i)$으로 분류기를 업데이트 하는 과정은 아래와 같습니다.

$$C_{m}(x_i)=C_{m-1}(x_i)+\alpha_{m}k_{m}(x_i)$$

그렇다면, 약한 분류기 $k_m$은 무엇이 되는 것이 최선인지, 그 분류기의 가중치 $\alpha_m$은 얼마가 되어야 하는지 결정을 해야 합니다. 이 때, Adaboost에서 사용하는 손실함수인 Exponential Loss를 이용하는데, 그 Loss의 공식은 다음과 같습니다.

$y$ :label 값 (1 or -1, True or False),&nbsp;&nbsp; $f(x)$: 예측값

$$L(y,f(x))=exp(-yf(x))=e^{-yf(x)}$$

Boosting Model $C_m(x_i)$의 총 오류 (cost function)를 $E$ 라고 정의할 때, $E$를 loss function을 이용해 다음과 같이 표현할 수 있습니다.

(<http://www.cs.man.ac.uk/~stapenr5/boosting.pdf> 참조)

$$E=\sum_{i=1}^N e^{-y_iC_m(x_i)}$$

 식은 복잡해 보이지만, 각 데이터 포인트마다 exponential loss를 구한 뒤, 이를 모두 더하는 것입니다. 예시를 들어보자면, $N=3$으로 데이터가 3개이고, 부스팅 모델이 예측한 결과 $C_m(x_i)$ 와 label 값 $y_i$이 다음 표와 같다고 합시다.


index | y_true | y_predict
----- | ------ | ---------
1     | 1      | 1
2     | 1      | -1
3     | -1     | -1


모델이 첫 번째와 세 번째 데이터는 올바르게 분류했지만 두 번째 데이터는 분류를 잘못한 것을 볼 수 있습니다. 이를 식으로 표현하면 다음과 같습니다.

$$
E=e^{-1\cdot1}+e^{-1\cdot(-1)}+e^{-(-1)\cdot(-1)}=e^1+2\cdot e^{-1}
$$

맞은 데이터에는 $e^{-1}$의 error (loss)가, 틀린 데이터에는 $e^1$의 error가 계산되었음을 알 수 있습니다. 0-1 Loss를 사용하는 것이 아닌, exponential loss를 사용하기 때문에 분류를 성공한 데이터에도 error값이 존재합니다. exponential loss는 다음과 같은 그래프를 가집니다.

<center><img src="https://slideplayer.com/slide/5849144/19/images/53/Exponential+Loss+Upper+Bounds+0%2F1+Loss%21+Can+prove+that.jpg" alt="Exponential Loss" width="400"></center>


이러한 Loss를 이용하여, 데이터의 가중치 개념을 포함한 총 오류 $E$ 를 정의해봅시다. 위의 예제에서는 각 데이터의 가중치가 모두 같은, 즉 모두 1이라고 가정할 때, $E$ 값은 $e^1+2\cdot e^{-1}$ 입니다. 이번에는, 각각의 데이터 가중치가 [1,2,3]이라고 해봅시다. 이 때의 총 오류 $E$ 를 계산하는 방법은 다음과 같습니다.

$$ E=1\cdot e^{-1\cdot1}+2\cdot e^{-1\cdot(-1)}+3 \cdot e^{-(-1)\cdot(-1)}=2\cdot e^1+4\cdot e^{-1} $$

 이를 일반화하여 표현하면 다음과 같습니다.

 $$E=\sum_{i=1}^N w_i^{(m)}e^{-y_iC_m(x_i)}$$

$w_i^{(m)}$에서 i는 데이터의 index를 나타내고 m는 m번째 가중치를 나타냅니다.

**총 에러는 각 데이터에서 발생한 Loss를 그 데이터의 가중치에 곱하여 계산됩니다.** 아래 문제에 대한 증명이 이러한 전개방식의 논리를 뒷받침해줍니다.

- Show that if we assign cost $a$ to misses and cost $b$ to hits, where $a>b>0,$ we can rewrite such costs as $a=c^d$ and $b=c^{-d}$ for constants $c$ and $d$. That is, exponential loss costs of the type $e^{\alpha _m}$ and $e^{-\alpha _m}$ do not compromise generality.

$-y_iC_{m-1}(x_i)$ 라는 표현은 얼핏 보면 복잡해보이지만, 분류기가 잘 분류했으면 $-1$의 값을 가지고, 잘못 분류했으면 $1$의 값을 가지게 됨을 의미합니다. 계속 맞는 데이터는 가중치에 $e^{-1}$이 계속 곱해지면서 낮은 가중치를 가질 것이고, 계속 틀리는 데이터일 수록 가중치에 $e^1$ 이 계속 곱해지면서 높은 가중치를 가지게 될 것입니다.
이를 일반화해서 표현하면 가중치가 다음과 같이 정의됩니다.

$m>1$ 에 대하여 $w_i^{(1)} = 1$, &nbsp;&nbsp; $w_i^{(m)}=e^{-y_iC_{m-1}(x_i)}$

다시 위의 문제로 돌아가서, 어떤 $ \alpha_m $ 과 $k_m$ 을 사용 해야할지 구해보도록 합시다.

$$C_{m}(x_i)=\alpha_1k_1(x_i)+\cdots+\alpha_{m}k_{m}(x_i)$$

임을 이용해서 총 오류 $E$를 다시 표현해보면,

$$E=\sum_{i=1}^N w_i^{(m)}e^{-y_iC_m(x_i)}=\sum_{i=1}^N w_i^{(m)}e^{-y_i\alpha_mk_m(x_i)}$$

이제, 이 식을 이용하여 총 오류를 최소화하는 모델별 가중치 $\alpha_m$ 을 찾아봅시다.

$y_ik_m(x_i) = 1$ 인 잘 분류된 데이터와 $y_ik_m(x_i) = -1 $ 인 잘못 분류된 데이터로 위 식을 쪼개봅시다.

$$\begin{align*}
E&=\sum_{y_i=k_m(x_i)} w_i^{(m)}e^{-y_i\alpha_mk_m(x_i)}+\sum_{y_i\neq k_m(x_i)} w_i^{(m)}e^{-y_i\alpha_mk_m(x_i)} \\
&=\sum_{y_i=k_m(x_i)} w_i^{(m)}e^{-\alpha_m}+\sum_{y_i\neq k_m(x_i)} w_i^{(m)}e^{\alpha_m}\\
&=e^{-\alpha_m}\times \sum_{y_i=k_m(x_i)} w_i^{(m)}+e^{\alpha_m} \times \sum_{y_i\neq k_m(x_i)} w_i^{(m)}
\end{align*}
$$

이제 위 식을 기준으로 best $\alpha_m$ 을 찾을 수 있습니다.

총 오류 $E $ 가 최소가 되도록 하는 $\alpha_m$을 편미분을 통해 구해봅시다.

$$ E=\sum_{y_i=k_m(x_i)} w_i^{(m)}e^{-\alpha_m}+\sum_{y_i\neq k_m(x_i)} w_i^{(m)}e^{\alpha_m} $$

$$ \frac{\partial E}{\partial \alpha_m}=-e^{-\alpha_m}\sum_{y_i=k_m(x_i)} w_i^{(m)}+e^{\alpha_m} \sum_{y_i\neq k_m(x_i)} w_i^{(m)}=0 $$


위 방정식을 만족시키는 $\alpha_m$을 구하기 앞서, 표현의 편의를 위해 가중치를 고려한 모델의 오분류율을 다음과 같이 정의합시다.

$$ \epsilon_m = \frac{\sum_{y_i\neq k_m(x_i)} w_i^{(m)}}{\sum_{y_i\neq k_m(x_i)} w_i^{(m)}+\sum_{y_i= k_m(x_i)} w_i^{(m)}} $$

 이 때, $\sum_{y_i\neq k_m(x_i)} w_i^{(m)} = W_e $, $\sum_{y_i= k_m(x_i)} w_i^{(m)} = W_c $라 하면,

$$ \frac{\partial E}{\partial \alpha_m}=-W_ce^{-\alpha_m}+W_e e^{\alpha_m} =0 $$

으로 위 방정식을 간략하게 표현할 수 있고, 양변에 $e^{\alpha_m}$을 곱해 식을 다음과 같이 간략화할 수 있습니다.

$$ \frac{\partial E}{\partial \alpha_m}=-W_c+W_e e^{2\alpha_m} =0 $$

따라서, 최적의 $ \alpha_m$은 다음과 같습니다.

$$ \alpha_m=\frac{1}{2}ln \left( \frac{W_c}{W_e} \right ) $$

이 때, $\epsilon_m = \frac{W_e}{W_c+W_e}$ , $W=W_c+W_e$ 이라 하면

$$ \alpha_m=\frac{1}{2}ln \left( \frac{W-W_e}{W_e} \right ) = \frac{1}{2}ln \left( \frac{1-\epsilon_m}{\epsilon_m} \right ) $$

으로 간단히 표현할 수 있습니다.
