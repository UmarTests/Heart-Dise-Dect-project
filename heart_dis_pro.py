import pandas as pd
import numpy as np
# ------Data Loading & Cleaning------
# Load dataset
df = pd.read_csv(r"C:\Users\mohdq\OneDrive\Desktop\internship projects\Heart_dise_dict\Heart Disease\dataset.csv")  
print("✅ Dataset Loaded")

# Check for missing values
print("\n🛠️ Checking for missing values...")
print(df.isnull().sum())

# Drop duplicates if any
df = df.drop_duplicates()
print("✅ Data Cleaning Done")

# ------Exploratory Data Analysis (EDA)------
import matplotlib.pyplot as plt
import seaborn as sns

print("\n📊 Performing EDA...")

# 1️⃣ Target Variable Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=df["target"], palette="coolwarm")
plt.title("Distribution of Heart Disease Cases")
plt.xticks([0, 1], ["No Disease", "Heart Disease"])
plt.show()

# 2️⃣ Correlation Heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# 3️⃣ Boxplot for Numerical Features
num_cols = ["age", "resting bp s", "cholesterol", "max heart rate", "oldpeak"]
df[num_cols].plot(kind="box", subplots=True, layout=(1, 5), figsize=(15, 5), notch=True)
plt.suptitle("Boxplots of Numeric Features")
plt.show()

print("✅ EDA Completed")

# ----Splitting the data into train and test sets----
from sklearn.model_selection import train_test_split

# Define features and target
X = df.drop(columns=["target"])  
y = df["target"]  

# Split into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"✅ Data Split: Train Size = {X_train.shape[0]}, Test Size = {X_test.shape[0]}")

# ------Model Building------
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# ⚡ Standardize Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define parameter grid for tuning
param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11],  
    "weights": ["uniform", "distance"],  
    "metric": ["euclidean", "manhattan", "minkowski"]
}

# Initialize KNN model
knn = KNeighborsClassifier()

# Grid Search with Cross-Validation (5-fold)
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Best parameters
print("🔍 Best Hyperparameters:", grid_search.best_params_)
print(f"✅ Best Accuracy (Cross-validation): {grid_search.best_score_:.4f}")

# Train final KNN model with best parameters
best_knn = KNeighborsClassifier(**grid_search.best_params_)
best_knn.fit(X_train_scaled, y_train)

# Evaluate Optimized Model
y_pred_best = best_knn.predict(X_test_scaled)
print("🔍 Optimized KNN Model Performance:")
print(f"✅ Test Accuracy: {accuracy_score(y_test, y_pred_best):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_best))

# 💾 Save Optimized Model & Scaler
joblib.dump(best_knn, "best_knn_heart_disease_model.joblib")
joblib.dump(scaler, "scaler.joblib")
print("✅ Optimized Model Saved!")

# ------Test Model with Sample Cases------
import numpy as np

# Select random test samples
num_samples = 5
indices = np.random.choice(X_test.shape[0], num_samples, replace=False)
X_sample = X_test.iloc[indices]
y_sample_actual = y_test.iloc[indices]

# Scale the sample data
X_sample_scaled = scaler.transform(X_sample)

# Predict using best KNN model
y_sample_pred = best_knn.predict(X_sample_scaled)

# Display results
print("🔍 **Testing Optimized KNN Model on Sample Data**")
for i in range(num_samples):
    print(f"🩺 Patient {i+1}: Actual = {'Disease' if y_sample_actual.iloc[i] == 1 else 'No Disease'}, "
          f"Predicted = {'Disease' if y_sample_pred[i] == 1 else 'No Disease'}")

# Predict on full test set
y_pred_test = best_knn.predict(X_test_scaled)

# Accuracy
accuracy = accuracy_score(y_test, y_pred_test)
print(f"✅ Optimized Model Accuracy on Test Set: {accuracy:.2f}")
