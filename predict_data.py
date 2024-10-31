import joblib
import numpy as np
import pandas as pd

# Load the model and scaler
model = joblib.load('healthcare_model.joblib')
scaler = joblib.load('scaler.joblib')

feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
new_data = np.array([[52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3]])  # Replace with actual feature values

# Convert new_data to a DataFrame with feature names
new_data_df = pd.DataFrame(new_data, columns=feature_names)

# Preprocess the new data and make a prediction
new_data_scaled = scaler.transform(new_data_df)
prediction = model.predict(new_data_scaled)
print("Prediction:", prediction)

# # New data for prediction (example)
# new_data = np.array([[52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3]])  # Replace with actual feature values

# # Preprocess the new data and make a prediction
# new_data_scaled = scaler.transform(new_data)
# prediction = model.predict(new_data_scaled)
# print("Prediction:", prediction)
