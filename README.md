# BANK CHURN CLASIFFICATION 🗞️

<p align="center">
    <img src="images/cover.png" width="500" height="400"/>
</p>

This repository hosts a notebook featuring an in-depth analysis of a **binary classification** with a bank churn dataset. The notebook contains the following structure

- Executive Summary
- Data Cleansing
- Univariate and Bivariate Analysis
- Feature Extraction 
- Preprocessing
- Baseline Model: LGBMClassifier
- VotingClassifier: LGBMClassifier, XGBoostClassifier, and CatBoostClassifier

The dataset used has been downloaded from the this [Kaggle](https://www.kaggle.com/competitions/playground-series-s4e1/data) competition.

## 👨‍💻 **Tech Stack**


![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23d9ead3.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 👨‍🔬 Exploratory Data Analysis

The first step of the project involved a comprehensive analysis of the dataset, including its columns and distribution. The idea was to identify correlations, outliers and the need to perform feature engineering. 

The dataset contains ten real-valued features  that are computed for each cell nucleus:

- `CustomerId`: A unique identifier for each customer

- `Surname`: The customer's surname or last name

- `CreditScore`: A numerical value representing the customer's credit score

- `Geography`: The country where the customer resides (France, Spain or Germany)

- `Gender`: The customer's gender (Male or Female)

- `Age`: The customer's age.

- `Tenure`: The number of years the customer has been with the bank

- `Balance`: The customer's account balance

- `NumOfProducts`: The number of bank products the customer uses (e.g., savings account, credit card)

- `HasCrCard`: Whether the customer has a credit card (1 = yes, 0 = no)

- `IsActiveMember`: Whether the customer is an active member (1 = yes, 0 = no)

- `EstimatedSalary`: The estimated salary of the customer

- `Exited`: Whether the customer has churned (1 = yes, 0 = no)
- 

The **mean**, **standard error** and **worst** or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features and additionally the target (Diagnosis: Malignant or Benign).

### Labels Distribution

It became apparent that the labels are not well-balanced, representing malignant only 37% of the samples. This means that oversampling or undersampling might be required. The dataset initially contained:

- Number of Benign:  357
- Number of Malignant :  212

<p align="center">
    <img src="images/counts.png" width="700" height="500"/>
</p>

### Features Distribution
The feature distribution revealed a significanT amount of outliers in all features except on the concave points worst feature. Also, all features are right skewed. This means that feature scaling can improve the models.

</p>
<p align="center">
    <img src="images/violin.png"/>
</p>

<p align="center">
    <img src="images/swarm.png"/>
</p>

<p align="center">
    <img src="images/box.png"/>
</p>

<p align="center">
    <img src="images/skewness.png" width="700" height="500"/>
</p>

It can be seen that some features like concavity_mean are well separated, which is good for classification purposes. Others like symmetry_worst are not separated, as the distribution of Benign and Malignant is similar.

### Correlation

In general, all measurement metrics show high correlation, like perimeter, area or radius. Also, concavity, fractal dimension and concave correlate to each other. In contrary, symmetry or smoothness, do not show any correlation with any feature.

<p align="center">
    <img src="images/corr.png"/>
</p>

<p align="center">
    <img src="images/wordcloud.png"/>
</p>

## Feature Engineering

Two approaches were selected for feature engineering:

- Scaling; Robust, Standard, MinMax
- PCA and KMeans

The scaling was performed depending on the distribution and skewness of the features. PCA and KMeans served as an intuition to see if reducing the number of features is significant.

<p align="center">
    <img src="images/pc.png"/>
</p>

Oversampling and undersampling was not selected as feature engineering as later the models showed very good results.

## 👨‍🔬 Modeling

The project involved training 4 models with varying configurations using Spark, Sklearn and feature engineering. All models showed very good performances with and without feature engineering (scaling and feature selection). The results are summarized below (confusion matrix belong to the baseline model Sklearn Random Forest):

<p align="center">
    <img src="images/cm.png"/>
</p>

- Sklearn Random Forest: 96 % and only 4 false negatives
- Sklearn Random Forest + Feature Selection: 94 % and only 6 false negatives
- Spark Random Forest: 97% and only 4 false positives
- Spark Random Forest + Feature Selection: 93% and only 9 false positives and 2 false negatives


### Model Performance Evaluation

All models demonstrated impressive performance, consistently achieving high accuracies, frequently surpassing the 90% mark and low amount of FN/FP. 
