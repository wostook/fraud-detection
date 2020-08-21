# Credit Card fraud detection
![image](https://github.com/wostook/fraud-detection/blob/master/data/CC.JPG)
In a project I used Kaggle Credit Card dataset containing transactions made by credit card in September 2013 by European cardholders. It shows transactions that took place in two days, where we have 492 frauds out of 284,807 transactions. The following project is a try to build best model to catch fraudulent transaction, that shows good trade-of between performance and computational efficiency.

## Motivation 

Fraud losses worldwide are estimated to reach $32 billion in 2020 and are projected to gradually rise to in the next years. Since ability to recognize fraudulent card transactions is undeniably extremely important, machine learning technical might come in hands. Is seems to be perfect fit-for purpose application. 

Another thing is class imbalance, it'sa good opportunity to test different techniques and perhaps challenge common interpretation techniques and metrics commonly used for classification tasks. 

## Tools and pipeline 

I downloaded data from Kaggle website, did EDA and simple feature engineering. I tried to oversample minority class, run CV on 3 different, raw models and tuned hyperparameters. In the end I analyzed and visualize results, making conlusions and recommendations.

Because of the size of dataset and algorithms used, I run project in Google Colab Notebook, taking advantage of free GPU and h2o4gpu package (GPU accelerated machine learning package, it allows to run GPU-optimized computations).

I used the following pipeline:

1.	Get data
2.	Quick look analysis and feature engineering
3.	Analysis of variance and detailed EDA
4.  Chose oversampling technique to deal with class imbalance (->> SMOTE algorithm)
4.	Chose clasiffcation metrics
5.	Chose raw models and run CV, test raw models
6.	Chose dataset for futher model building (balanced vs. imbalanced)
7.	Hyperparameters tuning- grid search and randomized grid search
8.	Get best model and metrics
9.	Make precision-recall and ROC plots
10. Interpret models
11. Final choice and recommendation
12. Lessons learned

## Prerequisites

The project was done with [Python 3.6.9](https://www.python.org/downloads/release/python-369/) and the following libraries:

- [h2o4gpu](https://pypi.org/project/h2o4gpu/)
- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [Matplotlib](http://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Imbalanced-learn](https://pypi.org/project/imbalanced-learn/)

## Data and Code

[Credit Card Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)

Code provided in `LDA_sentiment_twitter.ipynb`
