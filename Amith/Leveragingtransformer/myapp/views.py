from django.shortcuts import render
from .models import userdetails
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
import numpy as np
import joblib
import pandas as pd
import pandas as pd
import os
from django.shortcuts import render
from django.http import FileResponse
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

global df, X, y, label_encoder, y_encoded, scaler, X_scaled
global X_train, X_test, y_train, y_test, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


# Create your views here.

def index(request):
    return render(request,'myapp/index.html')

def login(request):
    if request.method == "POST":
        username = request.POST["uname"]
        pwd = request.POST["pwd"]

        if username == 'admin' and pwd == 'admin':
            # messages.success(request, "lOGIN Successfully")
            return render(request,'myapp/homepage.html')

        else:
            try:

                user = userdetails.objects.get(uname=username, pwd=pwd)
                request.session['uid'] = user.uid
                request.session['uname'] = user.uname
                print(user.uname)
                # messages.success(request, "lOGIN Successfully")
                return render(request,'myapp/homepage.html')

            except:
                pass
    return render(request,'myapp/login.html')

def register(request):
    if request.method == 'POST':
        uname = request.POST['uname']
        address = request.POST['address']
        mobno = request.POST['mobno']
        email = request.POST['email']
        pwd = request.POST['pwd']
        newuser = userdetails(uname=uname, address=address, mobno=mobno, email=email, pwd=pwd)
        newuser.save()

        return render(request, "myapp/login.html", {})
    return render(request,'myapp/register.html')

def homepage(request):
    return render(request,'myapp/homepage.html')

def eda(request):
    # Load dataset
    global df, X, y, label_encoder, y_encoded,scaler,X_scaled
    global X_train, X_test, y_train, y_test, X_train_tensor,y_train_tensor,X_test_tensor,y_test_tensor
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
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42,
                                                        stratify=y_encoded)
    content = {
        'data1': "Dataset size:" + str(df.shape),
        'data2': "No. of training Dataset:" + str(X_train.shape[0]),
        'data3': "No. of testing Dataset:" + str(X_test.shape[0]),
        'data4': "Data Preprocessing Completed"
    }
    return render(request,'myapp/eda.html',content)

def modelcreation(request):
    global df, X, y, label_encoder, y_encoded, scaler, X_scaled
    global X_train, X_test, y_train, y_test, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor= torch.tensor(y_train, dtype=torch.long)
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

    # Make predictions on test set
    y_pred = tabnet_model.predict(X_test)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)


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
    selected_features = ['Protocol Type', 'syn_flag_number', 'syn_count', 'Srate', 'Weight', 'Min', 'Max', 'DHCP',
                         'psh_flag_number', 'HTTP']

    X = df[selected_features]
    y = df['Attack_Category']

    # Encode target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42,
                                                        stratify=y_encoded)

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


    # Make predictions
    y_pred = best_rf_model.predict(X_test)


    # Classification Report
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    print(report)

    import joblib
    joblib.dump(best_rf_model, "rf_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")
    print("Model and encoders saved successfully.")
    content = {
        # 'data1': history,
        'data1': "Training Accuracy:" + str(train_accuracy),
        'data2': "Testing Accuracy:" + str(test_accuracy),
        'data2': "Cross-validation Accuracy:" + str(cv_scores),
        'data4': "Class_Report:" + str(report),
        'data5': "Model and encoders saved successfully."
    }
    return render(request,'myapp/modelcreation.html',content)


def predict(request):
    if request.method=='POST':
        download_flag = request.POST.get('download_flag')
        imgname = request.POST['myFile']
        # Load the saved model, scaler, and label encoder
        rf_model = joblib.load("rf_model.pkl")
        scaler = joblib.load("scaler.pkl")
        label_encoder = joblib.load("label_encoder.pkl")

        # Define feature names based on selected features
        selected_features = ['Protocol Type', 'syn_flag_number', 'syn_count', 'Srate', 'Weight', 'Min', 'Max', 'DHCP',
                             'psh_flag_number', 'HTTP']

        # Load the test dataset
        X_test_path = 'C:/Users/User/Desktop/Mtech final project/4th sem/Amith/Leveragingtransformer/Datasets/'+imgname
            # "Testing_Data__X_test_.csv"
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
        top_3_per_class = X_test.groupby("Predicted Attack Category").head(3)

        # Save the sorted results to a new CSV file
        output_path = "C:/Users/User/Desktop/Mtech final project/4th sem/Amith/Leveragingtransformer/predicted_top_3_results.csv"
        top_3_per_class.to_csv(output_path, index=False)

        print(f"Top 3 detections per class saved to {output_path}")
        # ########
        # Define the CSV file path
        csv_path = os.path.join(os.path.dirname(__file__), "C:/Users/User/Desktop/Mtech final project/4th sem/Amith/Leveragingtransformer/predicted_top_3_results.csv")

        # Read CSV file into a DataFrame
        try:
            df = pd.read_csv(csv_path)
            table_html = df.to_html(classes="table table-striped", index=False)  # Convert DataFrame to HTML table
        except Exception as e:
            table_html = f"<p>Error loading file: {str(e)}</p>"

        #return render(request, 'myapp/predict.html', {"data": table_html})
        if download_flag == '1':
            # Generate PDF from top3predictions.csv
            df = pd.read_csv(csv_path)
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            elements = []

            styles = getSampleStyleSheet()
            elements.append(Paragraph("Top 3 Predictions Report", styles['Title']))
            data = [df.columns.tolist()] + df.values.tolist()
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ]))
            elements.append(table)
            doc.build(elements)
            buffer.seek(0)

            return FileResponse(buffer, as_attachment=True, filename='Prediction_Report.pdf')
    return render(request, 'myapp/predict.html')

def viewgraph(request):
    return render(request,'myapp/viewgraph.html')

