## Overview

This is the public code repository for our work [Joint Model and Data Sparsification via the Marginal Likelihood](https://openreview.net/forum?id=vMcu1h3fOV) presented as a conference paper at the [43rd International Conference on Machine Learning (ICML)](https://icml.cc/Conferences/2026).

<!-- <p align="center">
  <img src="method-viz.jpeg" width="100%">
</p> -->

#### Abstract :memo:
---

Sparse recovery in linear systems underpins applications from signal processing to high-dimensional regression. Sparse Bayesian Learning, grounded in the principle of automatic relevance determination (ARD), offers a practical Bayesian mechanism for feature sparsity via marginal likelihood optimization. Yet, its reliance on a homoscedastic noise model renders it sensitive to data contaminations such as outliers or misspecified noise, harming model fit and predictions. Instead, we propose jointly learning individual feature and sample relevancies, enabling simultaneous model and data sparsification via a single Bayesian objective. This symmetric pruning of model and data offers a natural extension that preserves conjugacy, admits closed-form updates for standard optimization procedures, and aligns with perspectives from robust regression and influence functions. Empirical results across diverse regression tasks affirm that a joint ARD approach consistently yields both sparse and robust prediction models.

---

#### Citation
If you find this repository useful, please consider citing our work:

```
@inproceedings{timans2026robustsbl,
    title = {Joint Model and Data Sparsification via the Marginal Likelihoo}, 
    author = {Alexander Timans and Thomas Möllenhoff and Christian Naesseth and Mohammad Emtiyaz Khan and Eric Nalisnick},
    booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
    year = {2026}
}
```

#### Acknowledgements
The [Robert Bosch GmbH](https://www.bosch.com) is acknowledged for financial support.

## Code
Experiment code and instructions will be updated here shortly, please bear with us!

#### Still open questions?

If there are any problems you encounter which have not been addressed, please feel free to create an issue or reach out! 
