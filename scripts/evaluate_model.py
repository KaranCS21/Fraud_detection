# Evaluating the Model and Hyperparameter Tuning

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib

# Load preprocessed data and trained model
data_path = 'data/processed_data.csv'
df = pd.read_csv(data_path)
model_path = 'models/xgboost_model.pkl'
xgb_model = joblib.load(model_path)

# Separate features and labels
X = df.drop('Class', axis=1)  # Features
y = df['Class']  # Labels

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **Evaluate Original Model**

# Predict on test set
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]  # For AUC-ROC

# Classification report
print("Original Model Classification Report:")
print(classification_report(y_test, y_pred))

# AUC-ROC score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"Original Model AUC-ROC Score: {roc_auc}")

# **Hyperparameter Tuning using RandomizedSearchCV**

param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}

# Initialize RandomizedSearchCV
xgb_random = RandomizedSearchCV(
    estimator=XGBClassifier(scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]), random_state=42),
    param_distributions=param_grid,
    scoring='roc_auc',
    n_iter=20,
    cv=3,  # 3-fold cross-validation 
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Fit RandomizedSearchCV on the training data (not on the entire dataset)
xgb_random.fit(X_train, y_train)

# Best parameters and model
best_xgb_model = xgb_random.best_estimator_
print(f"Best Parameters: {xgb_random.best_params_}")

# Save the tuned model
joblib.dump(best_xgb_model, 'models/tuned_xgboost_model.pkl')


# **Evaluate Tuned Model**

# Predict with the tuned model on the test set
y_pred_tuned = best_xgb_model.predict(X_test)
y_pred_proba_tuned = best_xgb_model.predict_proba(X_test)[:, 1]

# Classification report after tuning
print("Tuned Model Classification Report:")
print(classification_report(y_test, y_pred_tuned))

# AUC-ROC score after tuning
roc_auc_tuned = roc_auc_score(y_test, y_pred_proba_tuned)
print(f"Tuned AUC-ROC Score: {roc_auc_tuned}")
