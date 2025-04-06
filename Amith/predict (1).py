# import numpy as np
# import joblib
# import pandas as pd
#
# # Load the saved model, scaler, and label encoder
# rf_model = joblib.load("rf_model.pkl")
# scaler = joblib.load("scaler.pkl")
# label_encoder = joblib.load("label_encoder.pkl")
#
# # Define feature names based on selected features
# selected_features = ['Protocol Type', 'syn_flag_number', 'syn_count', 'Srate', 'Weight', 'Min', 'Max', 'DHCP',
#                      'psh_flag_number', 'HTTP']
#
#
# def manual_predict(input_data):
#     """
#     Perform manual prediction using the trained Random Forest model.
#     :param input_data: Dictionary with feature names as keys and values as user input.
#     :return: Predicted attack category.
#     """
#
#     # Convert input data to DataFrame
#     input_df = pd.DataFrame([input_data], columns=selected_features)
#
#     # Scale input features
#     input_scaled = scaler.transform(input_df)
#
#     # Predict using the loaded model
#     prediction = rf_model.predict(input_scaled)
#
#     # Decode the prediction
#     predicted_label = label_encoder.inverse_transform(prediction)[0]
#
#     return predicted_label
#
#
# # Example manual input
# test_input = {
#     'Protocol Type': 1.0,
#     'syn_flag_number': 0.0,
#     'syn_count': 10,
#     'Srate': 5000.0,
#     'Weight': 50.0,
#     'Min': 30.0,
#     'Max': 60.0,
#     'DHCP': 1.0,
#     'psh_flag_number': 0.0,
#     'HTTP': 1.0
# }
#
# # Make a prediction
# predicted_attack = manual_predict(test_input)
# print(f"Predicted Attack Category: {predicted_attack}")


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

# Save the results to a new CSV file
output_path = "predicted_results.csv"
X_test.to_csv(output_path, index=False)

print(f"Predictions saved to {output_path}")

