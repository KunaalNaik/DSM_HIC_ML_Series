## **Project Description: Regression Feature Selection Model **

## **Project Overview**  
This project focuses on **feature selection** techniques to improve model performance by identifying and removing redundant, irrelevant, or highly correlated features. By applying multiple selection methods, we analyze their impact on model accuracy and interpretability. The goal is to **train a logistic regression model** using different feature selection approaches and compare results to determine the optimal feature subset.

---

## **Learning Objectives**  

1. **Preprocess Data:**  
   - Handle missing values in numerical and categorical features.  
   - Detect and remove outliers from numerical features.  
   - Scale numerical features for consistency.  
   - Handle rare categories in categorical data.  
   - Encode categorical variables for machine learning models.  

2. **Train and Evaluate a Baseline Model:**  
   - Train a logistic regression model using all available features.  
   - Compute classification metrics:  
     - Accuracy  
     - Precision  
     - Recall  
     - F1 Score  
     - ROC-AUC  
   - Identify feature importance using model coefficients.  

3. **Apply Feature Selection Techniques:**  
   - **Filter Methods:** Remove features based on correlation threshold.  
   - **VIF-Based Selection:** Detect and remove multicollinear features.  
   - **Wrapper Methods:** Use Recursive Feature Elimination (RFE) to select the most relevant features.  
   - **Embedded Methods:** Use Lasso Regression to shrink irrelevant feature coefficients to zero.  

4. **Compare and Select the Best Feature Selection Approach:**  
   - Train logistic regression models using different feature selection techniques.  
   - Evaluate and compare model performance across methods.  
   - Identify the feature selection method that improves accuracy and interpretability.  
   - Train and test the final model using the best-selected feature subset.  

---

## **Skills Developed**  

- **Data Preprocessing:**  
  - Handling missing values in numerical and categorical data.  
  - Detecting and removing outliers using the IQR method.  
  - Scaling numerical features for standardization.  
  - Encoding categorical variables.  

- **Feature Selection Methods:**  
  - **Filter-Based:** Removing correlated features using a correlation threshold.  
  - **Variance Inflation Factor (VIF):** Detecting and removing multicollinear features.  
  - **Wrapper-Based:** Recursive Feature Elimination (RFE) for feature selection.  
  - **Embedded-Based:** Lasso Regression for automated feature selection.  

- **Model Training & Evaluation:**  
  - Implementing logistic regression with different feature selection approaches.  
  - Computing classification metrics for model evaluation.  
  - Analyzing feature importance for selected feature sets.  

- **Model Comparison & Selection:**  
  - Comparing model performance across different feature selection methods.  
  - Selecting the best feature selection technique based on accuracy and ROC-AUC.  
  - Training a final model using the optimal feature set.  

---

## **Application**  

This project is valuable for real-world applications where feature selection plays a crucial role in improving model performance and efficiency. The techniques learned here are applicable to:  

- **Finance:** Credit scoring models with reduced feature sets.  
- **Healthcare:** Disease prediction models with only the most relevant patient attributes.  
- **E-Commerce:** Customer segmentation models using optimized feature subsets.  
- **Marketing Analytics:** Churn prediction with selected behavioral features.  

By mastering feature selection techniques, practitioners can build **efficient, interpretable, and high-performing machine learning models** across various industries.


## Tasks: Feature Selection

## **Main Task 1: Data Cleaning and Preprocessing**

---

### **Task 1-1: Import Libraries and Load the Dataset**

**Description:**  
Import necessary libraries for data handling, preprocessing, feature selection, model training, and evaluation. Load the dataset into a pandas DataFrame.

```python
# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Step 2: Load the dataset
data = pd.read_csv('/mnt/data/train_loan_prediction.csv')  # Replace with correct file path

# Step 3: Preview the dataset
data.head()
```

---

### **Task 1-2: Preprocess Numerical Features (Impute Missing Values, Remove Outliers, Scale Features)**

**Description:**  
Preprocess numerical features by imputing missing values, removing outliers, and scaling.

```python
# Identify numerical features
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features = [col for col in numerical_features if col != 'Loan_ID']

# Step 1: Create an instance of SimpleImputer with median strategy
imputer = SimpleImputer(strategy='median')

# Step 2: Fit the imputer on the numerical features
imputer.fit(data[numerical_features])

# Step 3: Transform the numerical features
data[numerical_features] = imputer.transform(data[numerical_features])

# Step 4: Remove outliers using IQR
for feature in numerical_features:
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data[feature] >= lower_bound) & (data[feature] <= upper_bound)]

# Step 5: Scale numerical features
scaler = StandardScaler()
scaler.fit(data[numerical_features])
data[numerical_features] = scaler.transform(data[numerical_features])

# Display preprocessed numerical features
data[numerical_features].head()
```

---

### **Task 1-3: Preprocess Categorical Features (Impute Missing Values, Handle Rare Categories, Encode Features)**

**Description:**  
Impute missing values, handle rare categories, and encode categorical features.

```python
# Identify categorical features
categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
categorical_features = [col for col in categorical_features if col != 'Loan_Status']

# Step 1: Create an instance of SimpleImputer with most frequent strategy
imputer = SimpleImputer(strategy='most_frequent')

# Step 2: Fit the imputer on the categorical features
imputer.fit(data[categorical_features])

# Step 3: Transform the categorical features
data[categorical_features] = imputer.transform(data[categorical_features])

# Step 4: Handle rare categories
threshold = 10
for feature in categorical_features:
    rare_categories = data[feature].value_counts()[data[feature].value_counts() < threshold].index
    data[feature] = data[feature].replace(rare_categories, 'Other')

# Step 5: Encode categorical features using Ordinal Encoding
encoder = OrdinalEncoder()
encoder.fit(data[categorical_features])
data[categorical_features] = encoder.transform(data[categorical_features])

# Display preprocessed categorical features
data[categorical_features].head()
```

---

## **Main Task 2: Baseline Model Building**

---

### **Task 2-1: Split Data into Training and Testing Sets**

**Description:**  
Split the dataset into training and testing sets.

```python
# Define features and target
X = data[numerical_features + categorical_features]
y = data['Loan_Status']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### **Task 2-2: Train Logistic Regression Model with All Features**

**Description:**  
Train a logistic regression model using all features.

```python
# Train logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
```

---

### **Task 2-3: Evaluate Model and Show Feature Importance**

**Description:**  
Evaluate model performance and display feature importance.

```python
# Predict on the test set
y_test_pred = logistic_model.predict(X_test)

# Calculate metrics
baseline_metrics = {
    "Accuracy": accuracy_score(y_test, y_test_pred),
    "Precision": precision_score(y_test, y_test_pred),
    "Recall": recall_score(y_test, y_test_pred),
    "F1 Score": f1_score(y_test, y_test_pred),
    "ROC-AUC": roc_auc_score(y_test, logistic_model.predict_proba(X_test)[:, 1]),
}
baseline_metrics

# Display feature importance
feature_importance = abs(logistic_model.coef_[0])
feature_names = X.columns
plt.barh(feature_names, feature_importance)
plt.xlabel("Feature Importance")
plt.title("Baseline Model Feature Importance")
plt.show()
```

---

## **Main Task 3: Filter-Based Feature Selection**

---

### **Task 3-1: Apply Filter Methods (Correlation Threshold)**

**Description:**  
Remove highly correlated features to avoid redundancy.

```python
# Calculate correlation matrix
correlation_matrix = X.corr()

# Filter features with high correlation
correlation_threshold = 0.8
filtered_features = [col for col in correlation_matrix.columns if abs(correlation_matrix[col].max()) < correlation_threshold]

# Display filtered features
filtered_features
```

---

### **Task 3-2: Build and Train Model with Filtered Features**

**Description:**  
Train a logistic regression model using filtered features.

```python
# Train model with filtered features
X_train_filtered = X_train[filtered_features]
X_test_filtered = X_test[filtered_features]
logistic_model_filtered = LogisticRegression()
logistic_model_filtered.fit(X_train_filtered, y_train)
```

---

### **Task 3-3: Evaluate Model and Compare Metrics with Baseline**

**Description:**  
Evaluate the model trained with filtered features and compare performance metrics.

```python
# Predict on the test set with filtered features
y_test_pred_filtered = logistic_model_filtered.predict(X_test_filtered)

# Calculate metrics for filtered features model
metrics_filtered = {
    "Accuracy": accuracy_score(y_test, y_test_pred_filtered),
    "Precision": precision_score(y_test, y_test_pred_filtered),
    "Recall": recall_score(y_test, y_test_pred_filtered),
    "F1 Score": f1_score(y_test, y_test_pred_filtered),
    "ROC-AUC": roc_auc_score(y_test, logistic_model_filtered.predict_proba(X_test_filtered)[:, 1]),
}
metrics_filtered
```

---

## **Main Task 4: VIF-Based Feature Selection**

---

### **Task 4-1: Calculate Variance Inflation Factor (VIF)**

**Description:**  
Calculate VIF for numerical features to detect multicollinearity.

```python
# Calculate VIF for numerical features
vif_data = pd.DataFrame()
vif_data["Feature"] = X_train.columns
vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

# Display VIF
vif_data
```

---

### **Task 4-2: Remove Features with High VIF, Train Model, and Evaluate Performance**

**Description:**  
Remove features with high VIF values, train a logistic regression model, and evaluate its performance.

```python
# Define VIF threshold
vif_threshold = 5
reduced_features = vif_data[vif_data["VIF"] < vif_threshold]["Feature"].tolist()

# Train and evaluate model with reduced features
X_train_reduced = X_train[reduced_features]
X_test_reduced = X_test[reduced_features]
logistic_model_reduced = LogisticRegression()
logistic_model_reduced.fit(X_train_reduced, y_train)

# Predict on the test set with reduced features
y_test_pred_reduced = logistic_model_reduced.predict(X_test_reduced)

# Calculate metrics for reduced features model
metrics_reduced = {
    "Accuracy": accuracy_score(y_test, y_test_pred_reduced),
    "Precision": precision_score(y_test, y_test_pred_reduced),
    "Recall": recall_score(y_test, y_test_pred_reduced),
    "F1 Score": f1_score(y_test, y_test_pred_reduced),
    "ROC-AUC": roc_auc_score(y_test, logistic_model_reduced.predict_proba(X_test_reduced)[:, 1]),
}
metrics_reduced
```

---

## **Main Task 5: Wrapper-Based Feature Selection**

---

### **Task 5-1: Apply Recursive Feature Elimination (RFE)**

**Description:**  
Use Recursive Feature Elimination (RFE) to select the most important features.

```python
# Apply RFE
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=10)
rfe.fit(X_train, y_train)

# Get selected features
selected_features_rfe = X_train.columns[rfe.support_]
selected_features_rfe
```

---

### **Task 5-2: Train Model with RFE-Selected Features and Evaluate Performance**

**Description:**  
Train a logistic regression model using only the features selected by RFE and evaluate its performance.

```python
# Train and evaluate model with RFE-selected features
X_train_rfe = X_train[selected_features_rfe]
X_test_rfe = X_test[selected_features_rfe]
logistic_model_rfe = LogisticRegression()
logistic_model_rfe.fit(X_train_rfe, y_train)

# Predict on the test set with RFE-selected features
y_test_pred_rfe = logistic_model_rfe.predict(X_test_rfe)

# Calculate metrics for RFE model
metrics_rfe = {
    "Accuracy": accuracy_score(y_test, y_test_pred_rfe),
    "Precision": precision_score(y_test, y_test_pred_rfe),
    "Recall": recall_score(y_test, y_test_pred_rfe),
    "F1 Score": f1_score(y_test, y_test_pred_rfe),
    "ROC-AUC": roc_auc_score(y_test, logistic_model_rfe.predict_proba(X_test_rfe)[:, 1]),
}
metrics_rfe
```

---

## **Main Task 6: Embedded Feature Selection**

---

### **Task 6-1: Apply Lasso Regression for Feature Selection**

**Description:**  
Use Lasso Regression (L1 regularization) to perform feature selection.

```python
# Apply Lasso Regression
lasso = LassoCV(cv=5)
lasso.fit(X_train, y_train)

# Select non-zero coefficient features
selected_features_lasso = X_train.columns[lasso.coef_ != 0]
selected_features_lasso
```

---

### **Task 6-2: Train Model with Lasso-Selected Features and Evaluate Performance**

**Description:**  
Train a logistic regression model using only the features selected by Lasso Regression and evaluate its performance.

```python
# Train and evaluate model with Lasso-selected features
X_train_lasso = X_train[selected_features_lasso]
X_test_lasso = X_test[selected_features_lasso]
logistic_model_lasso = LogisticRegression()
logistic_model_lasso.fit(X_train_lasso, y_train)

# Predict on the test set with Lasso-selected features
y_test_pred_lasso = logistic_model_lasso.predict(X_test_lasso)

# Calculate metrics for Lasso model
metrics_lasso = {
    "Accuracy": accuracy_score(y_test, y_test_pred_lasso),
    "Precision": precision_score(y_test, y_test_pred_lasso),
    "Recall": recall_score(y_test, y_test_pred_lasso),
    "F1 Score": f1_score(y_test, y_test_pred_lasso),
    "ROC-AUC": roc_auc_score(y_test, logistic_model_lasso.predict_proba(X_test_lasso)[:, 1]),
}
metrics_lasso
```
---

## **Main Task 7: Model Comparison and Final Selection**

---

### **Task 7-1: Compare Model Performance Across Feature Selection Methods**

**Description:**  
Compare classification metrics of different feature selection techniques.

```python
# Compare model performance across all feature selection methods
comparison_metrics = pd.DataFrame({
    "Baseline Model": baseline_metrics,
    "Filter-Based Model": metrics_filtered,
    "VIF-Based Model": metrics_reduced,
    "RFE Model": metrics_rfe,
    "Lasso Model": metrics_lasso
}).T

# Display the comparison metrics
comparison_metrics
```

---

### **Task 7-2: Select the Best Feature Selection Method Based on Performance**

**Description:**  
Identify the feature selection method that provides the highest accuracy and ROC-AUC.

```python
# Identify the best feature selection method
best_method = comparison_metrics.sort_values(by=["Accuracy", "ROC-AUC"], ascending=False).index[0]
best_method
```

---

### **Task 7-3: Train the Final Model Using the Best Feature Subset**

**Description:**  
Train a logistic regression model using the feature selection method with the best performance.

```python
# Train the final model using the best feature selection method
if best_method == "Filter-Based Model":
    X_train_final, X_test_final = X_train_filtered, X_test_filtered
elif best_method == "VIF-Based Model":
    X_train_final, X_test_final = X_train_reduced, X_test_reduced
elif best_method == "RFE Model":
    X_train_final, X_test_final = X_train_rfe, X_test_rfe
else:
    X_train_final, X_test_final = X_train_lasso, X_test_lasso

# Train final logistic regression model
final_model = LogisticRegression()
final_model.fit(X_train_final, y_train)

# Predict on the final test set
y_test_pred_final = final_model.predict(X_test_final)

# Calculate final model metrics
final_model_metrics = {
    "Accuracy": accuracy_score(y_test, y_test_pred_final),
    "Precision": precision_score(y_test, y_test_pred_final),
    "Recall": recall_score(y_test, y_test_pred_final),
    "F1 Score": f1_score(y_test, y_test_pred_final),
    "ROC-AUC": roc_auc_score(y_test, final_model.predict_proba(X_test_final)[:, 1]),
}
final_model_metrics
```

---

### **Task 7-4: Evaluate and Present the Final Model Results**

**Description:**  
Display the final model results and its impact compared to the baseline model.

```python
# Print final model performance
print("Final Model Performance Based on Best Feature Selection Method:")
final_model_metrics

# Display feature importance for the final model
feature_importance_final = abs(final_model.coef_[0])
feature_names_final = X_train_final.columns
plt.barh(feature_names_final, feature_importance_final)
plt.xlabel("Feature Importance")
plt.title(f"Feature Importance - {best_method}")
plt.show()
```