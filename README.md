# Credit Card Fraud Detection

This project aims to build a machine learning model for detecting fraudulent credit card transactions using the XGBoost algorithm. It applies data preprocessing techniques like SMOTE (Synthetic Minority Oversampling Technique), scaling, and PCA (Principal Component Analysis) to handle class imbalance, feature scaling, and dimensionality reduction.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Features](#features)
- [Visualizations](#visualizations)
- [Technologies](#technologies)
- [Future Enhancements](#future-enhancements)
- [License](#license)

## Project Overview
This project is based on a dataset containing credit card transaction records, where the goal is to predict whether a transaction is fraudulent or not. Class imbalance is handled using the SMOTE technique, and the XGBoost algorithm is used for classification.

## Project Structure
CreditCardFraudDetection/
│
├── data/
│   ├── raw_data.csv              # Original dataset (not uploaded due to size)
│   ├── processed_data.csv        # Preprocessed data (after scaling, PCA, etc.)
│
├── models/
│   ├── xgboost_model.pkl         # Saved XGBoost model
│
├── visualizations/
│   ├── correlation_matrix.png    # Correlation matrix visualization
│   ├── pairplot.png              # Pairplot visualization
│   ├── feature_importance.png    # Feature importance visualization
│
├── notebooks/
│   ├── eda.ipynb                 # Exploratory Data Analysis (EDA) notebook
│
├── scripts/
│   ├── preprocess.py             # Handles data preprocessing (SMOTE, scaling, PCA)
│   ├── train_model.py            # Trains the XGBoost model and saves it as a .pkl file
│   ├── evaluate_model.py         # Evaluates the model on the test set
│   ├── feature_engineering.py    # Visualizes feature importance and other feature relationships
│
└── README.md                     # Project documentation and instructions




## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Fraud-detection.git
   cd Credit-Card-Fraud-Detection

Install the necessary dependencies:
pip install -r requirements.txt


## Usage

Preprocess the data:
python scripts/preprocess.py

Train the XGBoost model:
python scripts/train_model.py

Evaluate the model:
python scripts/evaluate_model.py

Visualize feature importance:
python scripts/feature_engineering.py


## Model Evaluation
The XGBoost model achieves high accuracy, precision, and recall, particularly for detecting fraudulent transactions. Key metrics include:

Accuracy: X%
Precision: X%
Recall: X%
F1 Score: X%
AUC-ROC: X%

## Features
Handling Class Imbalance: Using SMOTE to oversample the minority class.
Feature Scaling: Normalizing features using MinMaxScaler.
Dimensionality Reduction: Applying PCA to reduce the dimensionality of the dataset.
Model Training: Using XGBoost for credit card fraud detection.
Hyperparameter Tuning: Applying techniques like GridSearchCV or RandomizedSearchCV.

## Visualizations
The project includes visualizations to help understand the data, such as:

Correlation Matrix
Feature Importance (XGBoost)
Pairplot of features


## Technologies
Python
XGBoost
SMOTE (imbalanced-learn)
PCA (scikit-learn)
Pandas, NumPy
Seaborn, Matplotlib


## Future Enhancements
Cost-Sensitive Learning: Apply cost-sensitive learning to prioritize the identification of fraudulent transactions.
Model Deployment: Deploy the trained model as a web service using Flask or Streamlit.

