import numpy as np
import joblib
import pandas as pd

# Load the saved model, scaler, and label encoder
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Define feature names based on selected features
selected_features = ['Protocol Type', 'syn_flag_number', 'syn_count', 'Srate', 'Weight', 'Min', 'Max', 'DHCP',
                     'psh_flag_number', 'HTTP']

# Load the test dataset
X_test_path = "Testing_Data__X_test_.csv"
X_test = pd.read_csv(X_test_path)

# Ensure only selected features are used
X_test = X_test[selected_features]

# Scale input features
X_test_scaled = scaler.transform(X_test)

# Predict using the loaded model
predictions = rf_model.predict(X_test_scaled)

# Decode the predictions
predicted_labels = label_encoder.inverse_transform(predictions)

# Add predictions to the DataFrame
X_test["Predicted Attack Category"] = predicted_labels

# Sort by Predicted Attack Category and display top 3 detections of each class
top_3_per_class = X_test.groupby("Predicted Attack Category").head(10)

# Save the sorted results to a new CSV file
output_path = "predicted_top_3_results.csv"
top_3_per_class.to_csv(output_path, index=False)

print(f"Top 3 detections per class saved to {output_path}")