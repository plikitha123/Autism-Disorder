import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import pickle

# Load dataset
data = pd.read_csv(r"C:\Users\likit\Downloads\ASD final\ASD final\ASD\train.csv")

# Drop unnecessary columns
data = data.drop(columns=['ID', 'result', 'contry_of_res'])

# Encode categorical variables to numbers
label_encoders = {}
categorical_features = ['gender', 'ethnicity', 'jaundice', 'austim', 'used_app_before', 'age_desc', 'relation']

for col in categorical_features:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Store encoder for later use

# Define features and target
X = data.drop(columns=['Class/ASD'])
y = data['Class/ASD']

# Print class distribution before SMOTE
print("Class distribution before SMOTE:")
print(y.value_counts())

# Handle class imbalance using SMOTE
smote = SMOTE(sampling_strategy=1.0, random_state=42)  # Fully balance the dataset
X_resampled, y_resampled = smote.fit_resample(X, y)

# Print class distribution after SMOTE
print("Class distribution after SMOTE:")
print(y_resampled.value_counts())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=42)

# Train RandomForest model with class weight adjustment
rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
rf_accuracy = rf.score(X_test, y_test)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# Train Logistic Regression
lr = LogisticRegression(max_iter=500, class_weight='balanced', random_state=42)
lr.fit(X_train, y_train)
lr_accuracy = lr.score(X_test, y_test)
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")

# Train MLP Neural Network
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
mlp_accuracy = mlp.score(X_test, y_test)
print(f"MLP Accuracy: {mlp_accuracy:.4f}")

# Train Support Vector Machine (SVM)
svm = SVC(probability=True, class_weight='balanced', random_state=42)
svm.fit(X_train, y_train)
svm_accuracy = svm.score(X_test, y_test)
print(f"SVM Accuracy: {svm_accuracy:.4f}")

# Train XGBoost
xgb = XGBClassifier(scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]), 
                    use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
xgb_accuracy = xgb.score(X_test, y_test)
print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")

# Evaluate models with classification reports
y_pred_rf = rf.predict(X_test)
y_pred_lr = lr.predict(X_test)
y_pred_mlp = mlp.predict(X_test)
y_pred_svm = svm.predict(X_test)
y_pred_xgb = xgb.predict(X_test)

print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
print("\nLogistic Regression Classification Report:\n", classification_report(y_test, y_pred_lr))
print("\nMLP Classification Report:\n", classification_report(y_test, y_pred_mlp))
print("\nSVM Classification Report:\n", classification_report(y_test, y_pred_svm))
print("\nXGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))

# Check if any positive cases are predicted
def check_positive_predictions(model, name):
    y_pred = model.predict(X_test)
    positive_count = sum(y_pred)
    print(f"{name} - Positive Predictions: {positive_count} out of {len(y_pred)}")
    if positive_count == 0:
        print(f"{name} - No positives found, adjusting threshold...")
        probabilities = model.predict_proba(X_test)[:, 1]
        new_preds = (probabilities > 0.3).astype(int)  # Lowering threshold
        print(f"{name} - Adjusted Positive Predictions: {sum(new_preds)}")

check_positive_predictions(rf, "Random Forest")
check_positive_predictions(lr, "Logistic Regression")
check_positive_predictions(mlp, "MLP")
check_positive_predictions(svm, "SVM")
check_positive_predictions(xgb, "XGBoost")

# Manually test a high ASD-risk input
high_risk_input = np.array([[1,1,1,1,1,1,1,1,1,1,5,1,2,1,1,1,2,1]])  # Modify based on features
print("\nTesting High-Risk ASD Case:")
for model, name in [(rf, "RF"), (lr, "LR"), (mlp, "MLP"), (svm, "SVM"), (xgb, "XGB")]:
    prob = model.predict_proba(high_risk_input)[0, 1]
    print(f"{name} Probability of ASD: {prob:.2f}")

# Save the best-performing model as ASD1.sav
best_model = max([(rf, rf_accuracy), (lr, lr_accuracy), (mlp, mlp_accuracy), (svm, svm_accuracy), (xgb, xgb_accuracy)], key=lambda x: x[1])[0]
pickle.dump(best_model, open("ASD1.sav", 'wb'))

# Save all models separately
pickle.dump(rf, open("ASD_RF.sav", 'wb'))
pickle.dump(lr, open("ASD_LR.sav", 'wb'))
pickle.dump(mlp, open("ASD_MLP.sav", 'wb'))
pickle.dump(svm, open("ASD_SVM.sav", 'wb'))
pickle.dump(xgb, open("ASD_XGB.sav", 'wb'))
