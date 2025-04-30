import torch
from pytorch_tabnet.tab_model import TabNetClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# Load dataset
df = pd.read_csv("CIC_IoMT_2024_WiFi_MQTT_dataset.csv")
# Extract features and target variable
X = df.drop(columns=['label', 'Attack_Category'])  # Drop categorical columns
y = df['Attack_Category']


# Encode target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Define TabNet model
tabnet_model = TabNetClassifier()

# Train the model
tabnet_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_name=['valid'],
    eval_metric=['accuracy'],
    max_epochs=50,
    patience=10,
    batch_size=512,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False
)

# Extract feature importance scores from the trained model
feature_importance = tabnet_model.feature_importances_
feature_names = df.drop(columns=['label', 'Attack_Category']).columns

# Create a DataFrame for feature importance
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values(by="Importance", ascending=False)

feature_importance_df.head(10)



# Get validation accuracy and loss history from TabNet model
history = tabnet_model.history

# Plot accuracy
plt.figure(figsize=(10, 5))
plt.plot(history['valid_accuracy'], label='Validation Accuracy', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('Accuracy.png')
# plt.show()

# Plot loss
plt.figure(figsize=(10, 5))
plt.plot(history['loss'], label='', marker='o', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('Loss.png')
# plt.show()

# Make predictions on test set
y_pred = tabnet_model.predict(X_test)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('cm.png')
# plt.show()

from sklearn.metrics import classification_report

# Generate classification report
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

# Display the classification report
print(report)


# Load dataset
df = pd.read_csv("CIC_IoMT_2024_WiFi_MQTT_dataset.csv")

# Take only 20% of the dataset
df = df.sample(frac=0.2, random_state=42).reset_index(drop=True)

# Select important features based on prior analysis
selected_features = ['Protocol Type', 'syn_flag_number', 'syn_count', 'Srate', 'Weight', 'Min', 'Max', 'DHCP', 'psh_flag_number', 'HTTP']
X = df[selected_features]
y = df['Attack_Category']

# Encode target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Define Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Define hyperparameter grid for tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform Grid Search with Cross-Validation
grid_search = GridSearchCV(rf_model, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model from Grid Search
best_rf_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Train best model on full training data
best_rf_model.fit(X_train, y_train)

# Evaluate model
train_accuracy = best_rf_model.score(X_train, y_train)
test_accuracy = best_rf_model.score(X_test, y_test)
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")

# Cross-validation score
cv_scores = cross_val_score(best_rf_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# Feature Importance
feature_importance = best_rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot Feature Importance
plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importance_df.Importance, y=feature_importance_df.Feature)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importances")
plt.savefig('plt1.png')
# plt.show()

# Make predictions
y_pred = best_rf_model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('plt2.png')
# plt.show()

# Classification Report
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print(report)

import joblib
joblib.dump(best_rf_model, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("Model and encoders saved successfully.")