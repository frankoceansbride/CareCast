import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filename):
    data = pd.read_csv("C:/Users/joann/Downloads/CareCast/healthcare_data.csv.csv")
    # Assuming 'target' is the column we want to predict
    X = data.drop(columns=["target"])
    y = data["target"]
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

# Example use case
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data("healthcare_data.csv.csv")
