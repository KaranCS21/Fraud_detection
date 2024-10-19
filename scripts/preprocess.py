import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def preprocess_data(data_path,save_path):
    # Load the dataset
    df = pd.read_csv(data_path)
    
    # Split features and labels
    X = df.drop('Class', axis=1)  # Features
    y = df['Class']  # Labels

    # Handle Class Imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Scale features using MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_resampled)

    # Apply PCA to reduce dimensionality while retaining 95% of variance
    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(X_scaled)

    # Combine the reduced features and resampled labels into a DataFrame
    processed_data = pd.DataFrame(X_reduced)
    processed_data['Class'] = y_resampled

    # Save the preprocessed data to a CSV file
    processed_data.to_csv(save_path, index=False)
    print(f"Preprocessed data saved to {save_path}")

    return X_reduced, y_resampled

# Example Usage
if __name__ == "__main__":
    data_path = 'data/creditcard.csv'    # Input file
    save_path = 'data/processed_data.csv'  # Output file for preprocessed data
    X_preprocessed, y_preprocessed = preprocess_data(data_path, save_path)
    print("Preprocessing completed.")
   
