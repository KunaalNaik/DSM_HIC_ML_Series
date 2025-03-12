# **Project Description: Regression Model – Categorical & Numerical Combined**

## **Project Overview**  
This project focuses on building a logistic regression model that incorporates both **numerical and categorical features**. The primary objective is to explore multiple **categorical encoding techniques**, preprocess both feature types, and evaluate their impact on model performance. By comparing different encoding methods, we aim to identify the most effective technique for improving model accuracy and interpretability.  

---

## **Learning Objectives**  

1. **Preprocess Data:**  
   - Handle missing values in numerical and categorical features.  
   - Detect and remove outliers from numerical features.  
   - Scale numerical features for consistency.  
   - Handle rare categories in categorical data.  

2. **Apply Categorical Encoding Techniques:**  
   - One-Hot Encoding (OHE).  
   - Ordinal Encoding.  
   - Mean Encoding.  
   - Leave-One-Out Encoding.  

3. **Train and Evaluate Models:**  
   - Train logistic regression models using different encoding techniques.  
   - Compute and compare classification metrics:  
     - Accuracy  
     - Precision  
     - Recall  
     - F1 Score  
     - ROC-AUC  

4. **Compare and Select the Best Encoding Method:**  
   - Evaluate the impact of different encodings on model performance.  
   - Select the encoding method that yields the highest accuracy and stability.  
   - Train and evaluate a final logistic regression model using the best encoding method.  

---

## **Skills Developed**  

- **Data Preprocessing:**  
  - Handling missing values in numerical and categorical data.  
  - Detecting and removing outliers using the IQR method.  
  - Scaling numerical features for standardization.  

- **Feature Engineering & Encoding:**  
  - One-Hot Encoding (OHE) for categorical variables.  
  - Ordinal Encoding for ranked categorical features.  
  - Mean Encoding to incorporate target relationships.  
  - Leave-One-Out Encoding to reduce overfitting.  

- **Model Training & Evaluation:**  
  - Implementing logistic regression with different encoding methods.  
  - Computing classification metrics for model evaluation.  
  - Analyzing feature importance for encoded features.  

- **Model Comparison & Selection:**  
  - Comparing model performance across different categorical encoding techniques.  
  - Selecting the best encoding method based on accuracy and ROC-AUC.  
  - Training a final model using the optimal encoding strategy.  

---

## **Application**  

This project is valuable for real-world applications where categorical data plays a crucial role in predictive modeling. The techniques learned here are applicable to:  

- **Finance:** Credit risk modeling with categorical customer attributes.  
- **Healthcare:** Disease prediction using categorical patient history.  
- **E-Commerce:** Customer purchase behavior prediction using categorical features.  
- **Marketing Analytics:** Churn prediction and campaign effectiveness analysis.  

By mastering categorical encoding methods, practitioners can build more **accurate, interpretable, and robust machine learning models** across diverse industries. 🚀

---

## Tasks: Baseline Numerical & Categorical

##

## **Main Task 1: Data Cleaning and Preprocessing**

---

### **Task 1-1: Import Libraries and Load the Dataset**

**Description:**  
Import necessary libraries for data handling, preprocessing, encoding, modeling, and evaluation. Load the dataset into a pandas DataFrame.

```python
# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from category_encoders import MeanEncoder, LeaveOneOutEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the dataset
data = pd.read_csv('/mnt/data/train_loan_prediction.csv')  # Replace with correct file path

# Step 3: Preview the dataset
data.head()
```

---

### **Task 1-2: Preprocess Numerical Features (Impute Missing Values, Remove Outliers, Scale Features)**

**Description:**  
Preprocess numerical features by imputing missing values with the median, removing outliers using the IQR method, and scaling features using `StandardScaler`.

```python
# Identify numerical features
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features = [col for col in numerical_features if col != 'Loan_ID']

# Impute missing values
imputer = SimpleImputer(strategy='median')
imputer.fit(data[numerical_features])
data[numerical_features] = imputer.transform(data[numerical_features])

# Remove outliers using IQR
for feature in numerical_features:
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data[feature] >= lower_bound) & (data[feature] <= upper_bound)]

# Scale numerical features
scaler = StandardScaler()
scaler.fit(data[numerical_features])
data[numerical_features] = scaler.transform(data[numerical_features])

# Display preprocessed numerical features
data[numerical_features].head()
```

---

### **Task 1-3: Preprocess Categorical Features (Impute Missing Values, Handle Rare Categories)**

**Description:**  
Impute missing values with the most frequent value and handle rare categories by grouping low-frequency categories as "Other."

```python
# Identify categorical features
categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
categorical_features = [col for col in categorical_features if col != 'Loan_Status']

# Impute missing values
imputer = SimpleImputer(strategy='most_frequent')
imputer.fit(data[categorical_features])
data[categorical_features] = imputer.transform(data[categorical_features])

# Handle rare categories
threshold = 10
for feature in categorical_features:
    rare_categories = data[feature].value_counts()[data[feature].value_counts() < threshold].index
    data[feature] = data[feature].replace(rare_categories, 'Other')

# Display preprocessed categorical features
data[categorical_features].head()
```

---

## **Main Task 2: Categorical Feature Encoding Methods**

---

### **Task 2-1: Apply One-Hot Encoding (OHE)**

**Description:**  
Convert categorical features into multiple binary columns using One-Hot Encoding.

```python
# Step 1: Create an instance of OneHotEncoder
ohe_encoder = OneHotEncoder(sparse=False, drop='first')

# Step 2: Fit the encoder on the categorical features
ohe_encoder.fit(data[categorical_features])

# Step 3: Transform the categorical features
ohe_encoded = ohe_encoder.transform(data[categorical_features])

# Step 4: Convert transformed features to a DataFrame
ohe_feature_names = ohe_encoder.get_feature_names_out(categorical_features)
ohe_encoded_df = pd.DataFrame(ohe_encoded, columns=ohe_feature_names, index=data.index)

# Step 5: Concatenate with numerical features
data_ohe = pd.concat([data[numerical_features], ohe_encoded_df, data['Loan_Status']], axis=1)

# Display transformed data
data_ohe.head()
```

---

### **Task 2-2: Apply Ordinal Encoding**

**Description:**  
Convert categorical features into integer values based on category order.

```python
# Step 1: Create an instance of OrdinalEncoder
ordinal_encoder = OrdinalEncoder()

# Step 2: Fit the encoder on the categorical features
ordinal_encoder.fit(data[categorical_features])

# Step 3: Transform the categorical features
data_ordinal = data.copy()
data_ordinal[categorical_features] = ordinal_encoder.transform(data_ordinal[categorical_features])

# Display transformed data
data_ordinal.head()
```

---

### **Task 2-3: Apply Mean Encoding (Using `category_encoders`)**

**Description:**  
Replace each category with the mean of the target variable.

```python
# Step 1: Create an instance of MeanEncoder
mean_encoder = MeanEncoder(cols=categorical_features)

# Step 2: Fit the encoder on the categorical features
mean_encoder.fit(data[categorical_features], data['Loan_Status'])

# Step 3: Transform the categorical features
data_mean = data.copy()
data_mean[categorical_features] = mean_encoder.transform(data_mean[categorical_features])

# Display transformed data
data_mean.head()
```

---

### **Task 2-4: Apply Leave-One-Out Encoding (Using `category_encoders`)**

**Description:**  
Replace each category with the mean of the target variable while leaving out the current observation.

```python
# Step 1: Create an instance of Leave-One-Out Encoder
loo_encoder = LeaveOneOutEncoder(cols=categorical_features)

# Step 2: Fit the encoder on the categorical features
loo_encoder.fit(data[categorical_features], data['Loan_Status'])

# Step 3: Transform the categorical features
data_loo = data.copy()
data_loo[categorical_features] = loo_encoder.transform(data_loo[categorical_features])

# Display transformed data
data_loo.head()
```

---

## **Main Task 3: Baseline Model Building**

---

### **Task 3-1: Split Data into Training and Testing Sets**

**Description:**  
Split the dataset into training and testing sets.

```python
# Define features and target
X = data_ohe.drop(columns=['Loan_Status'])
y = data_ohe['Loan_Status']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### **Task 3-2: Train Logistic Regression Model with All Features**

**Description:**  
Train a logistic regression model using all encoded features.

```python
# Train logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
```

---

### **Task 3-3: Evaluate Model and Show Feature Importance**

**Description:**  
Evaluate model performance and display feature importance.

```python
# Predict on the test set
y_test_pred = logistic_model.predict(X_test)

# Calculate metrics
model_metrics = {
    "Accuracy": accuracy_score(y_test, y_test_pred),
    "Precision": precision_score(y_test, y_test_pred),
    "Recall": recall_score(y_test, y_test_pred),
    "F1 Score": f1_score(y_test, y_test_pred),
    "ROC-AUC": roc_auc_score(y_test, logistic_model.predict_proba(X_test)[:, 1]),
}
model_metrics

# Display feature importance
feature_importance = abs(logistic_model.coef_[0])
feature_names = X.columns
plt.barh(feature_names, feature_importance)
plt.xlabel("Feature Importance")
plt.title("Feature Importance")
plt.show()
```

---

## **Main Task 4: Categorical Encoding Impact on Model Performance**

---

### **Task 4-1: Train and Evaluate Model with One-Hot Encoding**

**Description:**  
Train and evaluate a logistic regression model using One-Hot Encoding.

```python
# Define features and target for One-Hot Encoded data
X_ohe = data_ohe.drop(columns=['Loan_Status'])
y_ohe = data_ohe['Loan_Status']

# Split the dataset
X_train_ohe, X_test_ohe, y_train_ohe, y_test_ohe = train_test_split(X_ohe, y_ohe, test_size=0.2, random_state=42)

# Train logistic regression model
logistic_model_ohe = LogisticRegression()
logistic_model_ohe.fit(X_train_ohe, y_train_ohe)

# Predict on test set
y_test_pred_ohe = logistic_model_ohe.predict(X_test_ohe)

# Calculate and display metrics
metrics_ohe = {
    "Accuracy": accuracy_score(y_test_ohe, y_test_pred_ohe),
    "Precision": precision_score(y_test_ohe, y_test_pred_ohe),
    "Recall": recall_score(y_test_ohe, y_test_pred_ohe),
    "F1 Score": f1_score(y_test_ohe, y_test_pred_ohe),
    "ROC-AUC": roc_auc_score(y_test_ohe, logistic_model_ohe.predict_proba(X_test_ohe)[:, 1]),
}
metrics_ohe
```

---

### **Task 4-2: Train and Evaluate Model with Ordinal Encoding**

**Description:**  
Train and evaluate a logistic regression model using Ordinal Encoding.

```python
# Define features and target for Ordinal Encoded data
X_ordinal = data_ordinal.drop(columns=['Loan_Status'])
y_ordinal = data_ordinal['Loan_Status']

# Split the dataset
X_train_ordinal, X_test_ordinal, y_train_ordinal, y_test_ordinal = train_test_split(X_ordinal, y_ordinal, test_size=0.2, random_state=42)

# Train logistic regression model
logistic_model_ordinal = LogisticRegression()
logistic_model_ordinal.fit(X_train_ordinal, y_train_ordinal)

# Predict on test set
y_test_pred_ordinal = logistic_model_ordinal.predict(X_test_ordinal)

# Calculate and display metrics
metrics_ordinal = {
    "Accuracy": accuracy_score(y_test_ordinal, y_test_pred_ordinal),
    "Precision": precision_score(y_test_ordinal, y_test_pred_ordinal),
    "Recall": recall_score(y_test_ordinal, y_test_pred_ordinal),
    "F1 Score": f1_score(y_test_ordinal, y_test_pred_ordinal),
    "ROC-AUC": roc_auc_score(y_test_ordinal, logistic_model_ordinal.predict_proba(X_test_ordinal)[:, 1]),
}
metrics_ordinal
```

---

### **Task 4-3: Train and Evaluate Model with Mean Encoding**

**Description:**  
Train and evaluate a logistic regression model using Mean Encoding.

```python
# Define features and target for Mean Encoded data
X_mean = data_mean.drop(columns=['Loan_Status'])
y_mean = data_mean['Loan_Status']

# Split the dataset
X_train_mean, X_test_mean, y_train_mean, y_test_mean = train_test_split(X_mean, y_mean, test_size=0.2, random_state=42)

# Train logistic regression model
logistic_model_mean = LogisticRegression()
logistic_model_mean.fit(X_train_mean, y_train_mean)

# Predict on test set
y_test_pred_mean = logistic_model_mean.predict(X_test_mean)

# Calculate and display metrics
metrics_mean = {
    "Accuracy": accuracy_score(y_test_mean, y_test_pred_mean),
    "Precision": precision_score(y_test_mean, y_test_pred_mean),
    "Recall": recall_score(y_test_mean, y_test_pred_mean),
    "F1 Score": f1_score(y_test_mean, y_test_pred_mean),
    "ROC-AUC": roc_auc_score(y_test_mean, logistic_model_mean.predict_proba(X_test_mean)[:, 1]),
}
metrics_mean
```

---

### **Task 4-4: Train and Evaluate Model with Leave-One-Out Encoding**

**Description:**  
Train and evaluate a logistic regression model using Leave-One-Out Encoding.

```python
# Define features and target for Leave-One-Out Encoded data
X_loo = data_loo.drop(columns=['Loan_Status'])
y_loo = data_loo['Loan_Status']

# Split the dataset
X_train_loo, X_test_loo, y_train_loo, y_test_loo = train_test_split(X_loo, y_loo, test_size=0.2, random_state=42)

# Train logistic regression model
logistic_model_loo = LogisticRegression()
logistic_model_loo.fit(X_train_loo, y_train_loo)

# Predict on test set
y_test_pred_loo = logistic_model_loo.predict(X_test_loo)

# Calculate and display metrics
metrics_loo = {
    "Accuracy": accuracy_score(y_test_loo, y_test_pred_loo),
    "Precision": precision_score(y_test_loo, y_test_pred_loo),
    "Recall": recall_score(y_test_loo, y_test_pred_loo),
    "F1 Score": f1_score(y_test_loo, y_test_pred_loo),
    "ROC-AUC": roc_auc_score(y_test_loo, logistic_model_loo.predict_proba(X_test_loo)[:, 1]),
}
metrics_loo
```

---

## **Main Task 5: Final Model Selection and Summary**

---

### **Task 5-1: Compare Model Performance Across Encoding Methods**

**Description:**  
Compare the classification metrics of all encoding methods to identify the best-performing approach.

```python
# Compare model performance across all encoding methods
comparison_metrics = pd.DataFrame({
    "One-Hot Encoding": metrics_ohe,
    "Ordinal Encoding": metrics_ordinal,
    "Mean Encoding": metrics_mean,
    "Leave-One-Out Encoding": metrics_loo
}).T

# Display the comparison metrics
comparison_metrics
```

---

### **Task 5-2: Select the Best Encoding Method Based on Performance**

**Description:**  
Select the encoding method that provides the highest **accuracy and ROC-AUC**.

```python
# Identify the best encoding method based on accuracy and ROC-AUC
best_encoding_method = comparison_metrics.sort_values(by=["Accuracy", "ROC-AUC"], ascending=False).index[0]
best_encoding_method
```

---

### **Task 5-3: Train and Evaluate Final Model with Best Encoding Method**

**Description:**  
Train and evaluate the final model using the best encoding method.

```python
# Train the final model using the best encoding method
if best_encoding_method == "One-Hot Encoding":
    X_train_final, X_test_final, y_train_final, y_test_final = X_train_ohe, X_test_ohe, y_train_ohe, y_test_ohe
elif best_encoding_method == "Ordinal Encoding":
    X_train_final, X_test_final, y_train_final, y_test_final = X_train_ordinal, X_test_ordinal, y_train_ordinal, y_test_ordinal
elif best_encoding_method == "Mean Encoding":
    X_train_final, X_test_final, y_train_final, y_test_final = X_train_mean, X_test_mean, y_train_mean, y_test_mean
else:
    X_train_final, X_test_final, y_train_final, y_test_final = X_train_loo, X_test_loo, y_train_loo, y_test_loo

# Train final logistic regression model
final_model = LogisticRegression()
final_model.fit(X_train_final, y_train_final)

# Predict on the final test set
y_test_pred_final = final_model.predict(X_test_final)

# Calculate final model metrics
final_model_metrics = {
    "Accuracy": accuracy_score(y_test_final, y_test_pred_final),
    "Precision": precision_score(y_test_final, y_test_pred_final),
    "Recall": recall_score(y_test_final, y_test_pred_final),
    "F1 Score": f1_score(y_test_final, y_test_pred_final),
    "ROC-AUC": roc_auc_score(y_test_final, final_model.predict_proba(X_test_final)[:, 1]),
}
final_model_metrics
```

---