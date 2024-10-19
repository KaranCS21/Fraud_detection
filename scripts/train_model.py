import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

# Load preprocessed data
data_path = 'data/processed_data.csv'
df = pd.read_csv(data_path)

# Separate features and labels
X = df.drop('Class', axis=1)  # Features
y = df['Class']  # Labels

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost Model
xgb_model = XGBClassifier(
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),  # Adjust for class imbalance
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(xgb_model, 'models/xgboost_model.pkl')
print("XGBoost model saved successfully.")
