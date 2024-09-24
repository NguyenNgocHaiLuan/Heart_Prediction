# HeartCheck: A Beginner-Friendly Predictive Model for Heart Failure with 91% Accuracy

## Introduction

This project aims to build a machine learning model that can predict whether a patient is likely to experience heart failure based on a set of clinical features. Heart failure is a serious medical condition that can lead to death if not treated promptly, so accurate prediction is crucial for proper diagnosis and treatment.

In this project, we will be using a dataset containing information on patients' age, sex, blood pressure, Cholesterol, and other medical conditions to train a machine learning model. We will be using Python and several popular data science libraries such as Pandas, Numpy, Seaborn, Matplotlib, and Scikit-learn.

## Dataset

The dataset used in this project is available on [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction). It contains a total of 918 instances and 12 features. The features are as follows:

- `Age`: age of the patient [years]
- `Sex`: sex of the patient [M: Male, F: Female]
- `ChestPainType`: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
- `RestingBP`: resting blood pressure [mm Hg]
- `Cholesterol`: serum cholesterol [mm/dl]
- `FastingBS`: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
- `RestingECG`: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or - ST elevation or depression of > 0.05 mV), - LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
- `MaxHR`: maximum heart rate achieved [Numeric value between 60 and 202]
- `ExerciseAngina`: exercise-induced angina [Y: Yes, N: No]
- `Oldpeak`: oldpeak = ST [Numeric value measured in depression]
- `ST_Slope`: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
- `HeartDisease`: output class [1: heart disease, 0: Normal]

The target variable is `HeartDisease`, which indicates 1 for heart disease and 0 for Normal.

## Methodology

The first step of this project involves loading the dataset and performing exploratory data analysis (EDA) to gain insights into the data. We will be using Pandas and Matplotlib, Seaborn for data manipulation and visualization.

Next, we will be preprocessing the data by scaling the numerical features and encoding the categorical features. We will be using Scikit-learn for this task.

After preprocessing, we will be splitting the data into training and testing sets. We will be using Scikit-learn's `train_test_split` function for this task.

Once the data is ready, we will be training several machine learning models on the training set and evaluating their performance on the testing set. The models we will be using are:

- XGBoost (eXtreme Gradient Boosting)
- Catboost

We will be using Scikit-learn for training and evaluating these models.

## The evaluation is done using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score

## Results

After training and evaluating the models, we found that the Catboost classifier performed the best, achieving an accuracy of 91% on the testing set.

## Conclusion

In conclusion, we have successfully built a machine learning model that can predict whether a patient is likely to experience heart failure with an accuracy of 91%. 

## How to Use 

To use this project, you can clone the repository and run the `heart-failure-beginner-friendly-91-accuracy.ipynb` notebook using Jupyter or any other compatible notebook application.

```sh
git clone https://github.com/anik199/Heart-failure-prediction.git
cd Heart-failure-prediction
jupyter notebook heart-failure-beginner-friendly-91-accuracy.ipynb
```
