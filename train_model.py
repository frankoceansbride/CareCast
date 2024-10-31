import joblib
from sklearn.ensemble import RandomForestClassifier
from data_pre import load_and_preprocess_data

# Load and preprocess the data
X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data("healthcare_data.csv.csv")

# Initialized and trained my model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Saved my trained model and scaler
joblib.dump(model, 'healthcare_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("Model and scaler saved.")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate and print evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

from sklearn.model_selection import GridSearchCV

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30]
}

# Use GridSearchCV to tune parameters
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
grid_search.fit(X_train, y_train)

# Update model with the best parameters
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'healthcare_model_best.joblib')
print("Best Model saved.")
