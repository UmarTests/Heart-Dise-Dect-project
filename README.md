---
🔹 Author: Mohammad Umar  
🔹 Contact: umar.test.49@gmail.com  
---

# 🩺 Heart Disease Prediction using Machine Learning

---

### 📌 Section 1: Introduction and Objective  

Cardiovascular disease is one of the leading causes of death globally. Early and accurate detection of heart disease can significantly reduce the risk of severe outcomes and fatalities. This project aims to develop a machine learning-based predictive model that can assist healthcare providers in identifying patients at risk of heart disease based on several medical indicators.

**Client:** A healthcare research organization (assumed client) interested in improving diagnostic support using AI tools.

**Problem Statement:**  
Manually diagnosing heart disease is time-consuming and prone to subjectivity. The goal is to automate the prediction process using structured patient data to identify patterns that indicate heart disease.

**Importance of the Problem:**  
Early detection can lead to early intervention. With machine learning, the predictive capability can be enhanced, helping in resource prioritization, timely medication, and saving lives.

**Final Objective:**  
To build and deploy a machine learning model using Streamlit that predicts whether a patient has heart disease based on vital health parameters. The system should be accurate, interpretable, and easy to use by non-technical users.

---

### 📊 Section 2: Dataset  

**Source:** Provided by the internship organization. (Derived from the UCI Heart Disease dataset with slight variations)

**Dimensions:**  
- **Rows:** 918  
- **Columns:** 12  

**Key Features:**  
- `age`: Age in years  
- `sex`: 1 = male, 0 = female  
- `chest pain type`: 1–4 (typical angina to asymptomatic)  
- `resting bp s`: Resting blood pressure (mmHg)  
- `cholesterol`: Serum cholesterol (mg/dl)  
- `fasting blood sugar`: >120 mg/dl (1 = true; 0 = false)  
- `resting ecg`: 0 = normal, 1 = ST-T abnormality, 2 = LV hypertrophy  
- `max heart rate`: Maximum heart rate achieved  
- `exercise angina`: 1 = yes, 0 = no  
- `oldpeak`: ST depression induced by exercise  
- `ST slope`: 1 = upsloping, 2 = flat, 3 = downsloping  

**Target Variable:**  
- `target`: 0 = No Disease, 1 = Heart Disease

**Preprocessing Steps:**  
- Dropped duplicates  
- No missing values found  
- Scaled features using `StandardScaler`  
- Label encoded categorical features for model compatibility

**Key Observations:**  
- Dataset is balanced (~400 no-disease, ~500 disease cases)  
- ST Slope, Chest Pain Type, and Exercise-Induced Angina had strong correlation with heart disease  
- Cholesterol had weak correlation, but was retained for domain relevance

---

### ⚙️ Section 3: Design / Workflow  

Below is the end-to-end ML pipeline followed in this project:

#### 🔹 Data Loading & Cleaning
- Loaded `.csv` dataset using Pandas  
- Removed duplicate records  
- Validated absence of null values  

#### 🔹 Exploratory Data Analysis (EDA)
- Target class distribution → Balanced  
- Boxplots to detect outliers in `resting bp s`, `cholesterol`, and `oldpeak`  
- Correlation heatmap → ST Slope and Chest Pain had strong positive correlation with heart disease  
- Observed that high max heart rate correlated with **absence** of heart disease

#### 🔹 Feature Engineering
- Converted binary/categorical features (sex, angina, slope, etc.) to numerical  
- Scaled numerical features for models like KNN and Logistic Regression  
- Handled outliers by capping extreme values (IQR method)

#### 🔹 Model Training & Testing
- Train-test split: 80/20 with stratification  
- Trained and evaluated the following models:
  - Logistic Regression  
  - K-Nearest Neighbors (KNN)  
  - Random Forest  
  - SVM  
  - Gradient Boosting

#### 🔹 Hyperparameter Tuning
- Used GridSearchCV for KNN:
  - `n_neighbors`: [3, 5, 7, 9, 11]  
  - `weights`: ['uniform', 'distance']  
  - `metric`: ['euclidean', 'manhattan', 'minkowski']  
- Best parameters found: `n_neighbors=7`, `weights='distance'`, `metric='manhattan'`  
- Improved test accuracy from 0.89 to **0.91**

#### 🔹 Model Evaluation
- Metrics: Accuracy, Precision, Recall, F1-score  
- Sample predictions tested using real patient data from test set  
- Consistent performance and generalizability observed

#### 🔹 Streamlit Deployment
- Developed a clean UI with dropdowns and sliders for non-technical users  
- Handled scaling warnings using `DataFrame` input format  
- Users can input real patient data and get instant prediction

---

### 📈 Section 4: Results  

**Final Model Used:**  
Optimized **K-Nearest Neighbors (KNN)** after hyperparameter tuning

**Final Test Accuracy:** `0.91`  
**F1-Score:**  
- No Disease: 0.87  
- Heart Disease: 0.91

**Comparison Summary:**

| Model                | Accuracy |
|----------------------|----------|
| Logistic Regression  | 0.89     |
| Random Forest        | 0.89     |
| SVM                  | 0.89     |
| Gradient Boosting    | 0.89     |
| **KNN (Tuned)**      | **0.91** |

**Visuals (Generated during EDA & Tuning):**
- 📊 Correlation heatmap  
- 📉 Boxplots for numeric outliers  
- 📌 Classification reports with precision, recall, F1  
- ✅ Sample patient prediction results from test set  

**Key Insights:**  
- ST slope, chest pain type, exercise-induced angina, and oldpeak are key indicators  
- Age and fasting blood sugar had moderate correlation  
- Cholesterol and resting ECG had weaker predictive power  

---

### ✅ Section 5: Conclusion  

**Summary of Achievements:**
- Successfully built and tuned a heart disease prediction model using KNN  
- Achieved 91% accuracy after optimization  
- Developed and deployed an interactive Streamlit app for real-time predictions  

**Challenges Faced:**
- Dealing with inconsistent feature naming between training and app inputs  
- Hyperparameter tuning was computationally expensive  
- Balancing interpretability and accuracy during model selection  

**Scope for Future Work:**
- Try deep learning models on larger datasets  
- Integrate real-time patient data collection via web APIs  
- Use SHAP values for model explainability  
- Deploy using Docker + Cloud for scalability

**Personal Learnings:**
- End-to-end ML project structure  
- Real-world deployment using Streamlit  
- Importance of clean UI/UX in health-tech apps  
- How tuning small parameters can significantly improve results

---

✅ **Thank you for reading!** If you liked this project, feel free to ⭐ the repo or reach out via [umar.test.49@gmail.com](mailto:umar.test.49@gmail.com).
