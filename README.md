# SRGA [TPAMI2023]

## Evaluating the Generalization Ability of Super-Resolution Networks

This paper was accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI). [[paper]](https://ieeexplore.ieee.org/abstract/document/10239439)


This is the first attempt to propose a Generalization Assessment Index for super-resolution and restoration networks, namely SRGA. SRGA exploits the statistical characteristics of the internal features of deep networks to measure the generalization ability.

## News
- [2024/5/31] :zap: We have released the codes! Please refer to the following instructions. (CPU version)

## Brief Introduction
SRGA is built upon the statistical modeling of deep features of SR networks. It is a relative measurement that computes the distance of the feature distributions between the reference dataset and the candidate test dataset. The reference dataset used in SRGA is typically the one on which the model performs well, and is usually within the training distribution.

![framework](framework.jpg)

- First obtain the corresponding deep features (the last layer) of the input datasets.
- These features are then compressed using principal component analysis (PCA).
- The resulting projected feature sets are modeled using a generalized Gaussian distribution (GGD).
- The generalization error is measured by the Kullback-Leibler divergence (KLD) between the two probability distributions, leading to the proposed generalization index SRGA.

![illustration](illustration.jpg)

