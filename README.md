# Telecom-Customer-Churn-Prediction

![telecomchurn](https://user-images.githubusercontent.com/49653689/94882916-49097100-0437-11eb-8819-5ff8e62107b6.png)

#### -- Project Status: [Completed]

## Project Objective
The purpose of this project is to predict behaviors to retain customers.

### Methods Used
* Performed Data Preprocessing / Cleaning / Feature Engineering / Data Visualization
* Trained Classifiers (KNN, GaussianNB, SVM, Logistics Regression, and Artificial Neural Network) 
* Implemented Tree-based Ensemble Methods (Random Forest, Adaboost and XGBoost)
* Evaluated Algorithms performance in metrics

### Technologies
* Jupyter, Python 3
* Pandas, Numpy, Matplotlib, Seaborn, sklearn

## Project Description

### Data Source

[Kaggle : Telcom Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)

### Data Pre-processing

* Replaced missing 'TotalCharges' with its mean value
* Under-sampled 'Churn' to 'No Churn' ratio to 1:1
* One-Hot Encoding and only dummies which correlation < 50% retained
* PCA dimension reduction (100% variance with 23/45 dummies)

### Model Training & Hyperparameters Tuning

#### Machine Learning Classifiers

* k-nearest neighbors (n_neighbors, leaf_size, p, algorithm)
* GaussianNB (var_smoothing)
* Logistics Regression (solver, penalty, C)
* SVM (kernel, gamma(kernel='rbf'), C)
* Decision Tree (criterion, max_depth, min_samples_split, max_features)

#### Tree-based Ensemble Methods

* Random Forest (criterion, max_depth, min_samples_split, max_features, n_estimators)
* Adaboost (n_estimators, learning_rate)
* XGBoost (learning_rate, max_depth, min_child_weight, gamma, colsample_bytree, reg_alpha, subsample, colsample_bytree)

#### Neural network

* Artificial Neural Network (batch_size, epochs, optimizer)

### Test Accuracy

* Logistics Regression: 74.73%

* KNN Classifier: 75.53%

* SVM Classifier: 76.07%

* Decision Tree: 72.59%

* Random Forest Classifier: 75.67%

* **AdaBoost: 77.14% (Best Model)**

* XGBoost: 76.74%

* NaiveBayes Classifier: 74.73%

* Artificial Neural network: 74.73%


## Further Work

![learningcurve](https://user-images.githubusercontent.com/49653689/94983763-e5a14100-0513-11eb-8f4d-b027883dad27.png)

Training score (red line) decreases and plateau
Indicates underfitting
High bias

Cross-validation score (green line) stagnating throughout
Unable to learn from data

Low scores (high errors)
Should tweak model (perhaps increase model complexity)

## Reference

KNN
https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/

Support Vector Machine — Simply Explained
https://towardsdatascience.com/support-vector-machine-simply-explained-fee28eba5496

Gaussian NB
https://scikit-learn.org/stable/modules/naive_bayes.html

An Introduction to Naïve Bayes Classifier (calculation) 
https://towardsdatascience.com/introduction-to-naïve-bayes-classifier-fa59e3e24aaf

Random Forest in Python
https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

Gradient Boost Part 1: Regression Main Ideas
https://www.youtube.com/watch?v=3CC4N4z3GJc

Gradient Boost Part 3: Classification
https://www.youtube.com/results?search_query=gradient+boost+part+2

Machine Learning Basics - Gradient Boosting & XGBoost
https://www.shirin-glander.de/2018/11/ml_basics_gbm/

AdaBoost, Clearly Explained
https://www.youtube.com/watch?v=LsK-xG1cLYA

AdaBoost Classifier Example In Python
https://towardsdatascience.com/machine-learning-part-17-boosting-algorithms-adaboost-in-python-d00faac6c464

AdaBoost Classification in Python(hyperparameter tuning)
https://educationalresearchtechniques.com/2019/01/02/adaboost-classification-in-python/

Hyperparameter tuning in XGBoost (Awesome!) 
https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f

Complete Guide to Parameter Tuning in XGBoost with codes in Python(the best!)
https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

XGBoost Documentation
https://xgboost.readthedocs.io/en/latest/index.html










