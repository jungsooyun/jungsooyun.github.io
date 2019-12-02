---
title: Youtube의 추천 알고리즘 이해하기
search: true
categories:
 - Recommendation
tags:
 - Recommender System
 - Learning to Rank
 - Deep Learning
 - Youtube
last_modified_at: 2019-11-08 01:51
---


이번 글에서는 Recsys 19에서 공개된 [Recommending What Video to Watch Next: A Multitask Ranking System](https://dl.acm.org/citation.cfm?id=3346997) 을 정리하였습니다.

몸담고 있는 분야의 글이기도 하고, 더불어 유튜브의 추천 알고리즘이 어떻게 동작하는지는 만인의 관심사이니까요!

## 요약

이 논문은 비디오 공유 플랫폼(Youtube) '어떤 비디오를 다음에 볼까'의 문제를 푸는 large scale의 multi-objective ranking system에 대해 다룹니다. 이 추천 시스템은 피드백에 내재(implicit)된 선택 편향(biases) 뿐만 아니라, 여러 가지 순위 목표(ranking objective)가 존재한다는 문제에 직면해 있습니다.

구글 연구진들은 Multi-gate Mixture-of-Experts (이하 MMoE) 등의 soft-parameter sharing 기술들을 조사했으며, 이를 이용해 다양한 순위 목표를 효율적으로 최적화 하는 것을 목표로 하여 이를 적용했습니다. 또한, Wide & Deep 프레임워크를 이용하여 선택 편향 문제를 완화했습니다.

## 서두

이 논문은 유저가 현재 보고 있는 영상에 추천 영상을 띄우는 케이스에 대해 다룹니다. 영상을 보고 있을 때, 추천 결과는 다음과 같이 노출됩니다.

![추천 결과](/images/Screen%20Shot%202019-11-22%20at%205.34.47%20PM.png)

전통적 추천 시스템은 2단계의 디자인을 따르는데, 이는 **후보 선택 과정**, **순위를 매기는 과정** (순위 과정 참조 : *Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep neural networks for YouTube Recommendations. In Proceedings of the 10th ACM conference on recommender systems. ACM, 191–198.* : 기존 유튜브 논문

*XinranHe,JunfengPan,OuJin,TianbingXu,BoLiu,TaoXu,YanxinShi,Antoine Atallah, Ralf Herbrich, Stuart Bowers, et al. 2014. Practical lessons from predicting clicks on ads at facebook. In Proceedings of the Eighth International Workshop on Data Mining for Online Advertising. ACM, 1–9.* : Facebook CTR 예측 논문)

입니다. 이 논문은 2단계 중 순위를 매기는 과정에 집중합니다. 추천 시스템은 후보 선택기에([matrix factorization](45), [neural models](25)) 의해 선택된 후보군을 가지고 있는 상태입니다. 그 후, 정교하고 매우 큰 사이즈의 모델을 순위를 매기는데 사용하며 매력적인 아이템들을 최종 선정합니다.

필자들은 이런 랭킹 시스템을 디자인하고 실험을 진행하면서 크게 2가지의 문제점에 봉착했는데, 그는 다음과 같습니다.

- 최적화 시키고자 하는 objective가 여러 개라 다를 수도 있고 충돌하는 경우도 있습니다. 예를 들어, 목적을 *시청 수 뿐만 아니라, 평점이 높고 조회수가 높은 비디오를 추천하고 싶어!* 라고 설정한 경우에 이런 충돌이 발생합니다.

- implicit feedback으로부터 얻은 정보가 편향되었을 수 있다는 것입니다. 여기에 해당하는 예시로는 [Feedback loop effect](33)이 있는데, 이 예시로는 유저가 추천 결과를 좋아해서 클릭한 것이 아니라, 단지 상단에 추천되었기 때문에 클릭한 경우가 있습니다. 이러한 경우에 implicit feedback을 모델을 학습시키기 위한 데이터로 사용하는 경우, 현재의 랭킹 시스템은 편향될 것이고 feedback loop effect를 야기합니다.

이러한 문제점에 대응하기 위해, 필자들은 랭킹 시스템을 위한 효과적인 multi-task 신경망 구조를 제안합니다. multi-task 학습을 위해 MMoE를 이용하면서 기존의 [Wide & Deep Model architecture](9)를 확장하였고, 
