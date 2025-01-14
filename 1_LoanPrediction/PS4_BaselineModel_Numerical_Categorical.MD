## **Project Description: Comprehensive Dataset Understanding**

#### **Overview**
This project is designed to help beginners gain a step-by-step understanding of how to explore and comprehend a dataset using Python and the `pandas` library. By completing this project, participants will learn the fundamental techniques required to prepare for data analysis and build a solid foundation in working with datasets.

The tasks are divided into logical buckets—loading the dataset, extracting basic information, analyzing columns, and inspecting data distribution—allowing learners to progress systematically. The focus is not on Exploratory Data Analysis (EDA) or visualization but on building the skills necessary to understand the structure and characteristics of the dataset.

---

### **Learning Objectives**

By the end of this project, participants will be able to:

1. **Load and Preview a Dataset**:
   - Understand how to import libraries, load data, and preview its contents.
   - Extract index labels to understand how rows are indexed.

2. **Extract Basic Dataset Information**:
   - Identify the number of rows and columns in the dataset.
   - Retrieve data types, column names, and basic metadata about the dataset.

3. **Analyze Columns**:
   - Differentiate between numerical and categorical columns.
   - Count unique values and non-null values in each column.
   - Identify missing data and understand where data imputation may be needed.

4. **Generate Data Statistics**:
   - Compute summary statistics for numerical and categorical columns.
   - Understand the memory usage of the dataset and optimize data handling.

5. **Sample and Inspect Data**:
   - Randomly sample rows to inspect data quality.
   - Analyze value distributions for specific columns to understand categorical and numerical data diversity.

---

### **Skills Developed**

- **Data Handling**:
  - Loading and previewing datasets in Python.
  - Inspecting rows, columns, and data types.

- **Basic Data Exploration**:
  - Identifying data structures like numerical and categorical features.
  - Extracting and interpreting missing values, unique counts, and column-level statistics.

- **Descriptive Analytics**:
  - Summarizing datasets using statistical methods.
  - Memory usage analysis for efficient data management.

- **Programming Fundamentals**:
  - Writing modular, reusable code for dataset exploration.
  - Using Python methods and attributes effectively in data analysis.

---

### **Application**
The skills and techniques learned in this project can be applied to any dataset in industries such as finance, healthcare, e-commerce, and technology. They form the foundation for performing more advanced analyses, including Exploratory Data Analysis (EDA), machine learning, and data visualization.
This project ensures participants are ready to approach datasets with confidence and systematically uncover insights from raw data.


## Tasks: Baseline Numerical Only Model

## Main Task 1: Data Preparation

### Task 1-1: Import Libraries and Load the Dataset

**Description:**  
Import the necessary libraries and load the dataset into a pandas DataFrame. This step sets up the environment and ensures the dataset is ready for preprocessing.

```python
# Step 1: Import necessary libraries
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the dataset
data = pd.read_csv('/mnt/data/train_loan_prediction.csv')  # Replace with the correct path

# Step 3: Preview the dataset
data.head()
```

---

### Task 1-2: Identify Numerical Features

**Description:**  
Identify all numerical features in the dataset, excluding the target variable (`Loan_Status`) and ID column.

```python
# Step 1: Identify numerical features
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Step 2: Exclude the ID column
numerical_features = [col for col in numerical_features if col != 'Loan_ID']

# Display numerical features
numerical_features
```

---

### Task 1-3: Check Missing Values

**Description:**  
Check for missing values in the selected numerical features to assess data completeness.

```python
# Step 1: Check missing values for numerical features
missing_values = data[numerical_features].isnull().sum()

# Display missing values
missing_values
```

---

### Task 1-4: Treat Missing Values Using SimpleImputer

**Description:**  
Use SimpleImputer to replace missing values in numerical features with the median of each column.

```python
# Step 1: Create an instance of SimpleImputer with median strategy
imputer = SimpleImputer(strategy='median')

# Step 2: Fit the imputer on the numerical features
imputer.fit(data[numerical_features])

# Step 3: Transform the numerical features
data[numerical_features] = imputer.transform(data[numerical_features])

# Display transformed data
data[numerical_features].head()
```

---

## Main Task 2: Preprocessing

### Task 2-1: Check for Outliers in Numerical Features

**Description:**  
Detect outliers in numerical features using the Interquartile Range (IQR) method. Outliers can skew the model and need to be handled.

```python
# Step 1: Initialize a dictionary to store outlier counts
outliers = {}

# Step 2: Calculate outliers for each numerical feature
for feature in numerical_features:
    Q1 = data[feature].quantile(0.25)  # First quartile
    Q3 = data[feature].quantile(0.75)  # Third quartile
    IQR = Q3 - Q1  # Interquartile range
    lower_bound = Q1 - 1.5 * IQR  # Lower bound
    upper_bound = Q3 + 1.5 * IQR  # Upper bound
    outliers[feature] = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)].shape[0]

# Display outliers count for each feature
outliers
```

---

### Task 2-2: Remove Outliers Using the IQR Method

**Description:**  
Remove rows containing outliers in numerical features based on the IQR method.

```python
# Step 1: Remove outliers for each numerical feature
for feature in numerical_features:
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data[feature] >= lower_bound) & (data[feature] <= upper_bound)]

# Display data shape after outlier removal
data.shape
```

---

### Task 2-3: Check for Normal Distribution of Numerical Features

**Description:**  
Visualize the distributions of numerical features to determine if they follow a normal distribution.

```python
# Step 1: Plot histograms for numerical features
for feature in numerical_features:
    plt.figure()
    sns.histplot(data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()
```

---

### Task 2-4: Scale Numerical Features Using StandardScaler

**Description:**  
Standardize numerical features using StandardScaler to bring all features to the same scale.

```python
# Step 1: Create an instance of StandardScaler
scaler = StandardScaler()

# Step 2: Fit the scaler on the numerical features
scaler.fit(data[numerical_features])

# Step 3: Transform the numerical features
data[numerical_features] = scaler.transform(data[numerical_features])

# Display scaled data
data[numerical_features].head()
```

---

## Main Task 3: Model Building and Evaluation

### Task 3-1: Split Data into Training and Testing Sets

**Description:**  
Split the data into training and testing sets with an 80-20 split, ensuring the target variable (`Loan_Status`) is used correctly.

```python
# Step 1: Define features (X) and target (y)
X = data[numerical_features]
y = data['Loan_Status']

# Step 2: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the splits
X_train.shape, X_test.shape, y_train.shape, y_test.shape
```

---

### Task 3-2: Train Logistic Regression Model

**Description:**  
Train a logistic regression model using the preprocessed numerical features.

```python
# Step 1: Create logistic regression model
logistic_model = LogisticRegression()

# Step 2: Train the model
logistic_model.fit(X_train, y_train)
```

---

### Task 3-3: Predict on Both Training and Testing Sets

**Description:**  
Use the trained logistic regression model to make predictions on both the training and testing sets. This will help evaluate the model's performance on seen (training) and unseen (testing) data.

```python
# Step 1: Predict on the training set
y_train_pred = logistic_model.predict(X_train)

# Step 2: Predict on the testing set
y_test_pred = logistic_model.predict(X_test)

# Display predictions for both training and testing sets
y_train_pred[:10], y_test_pred[:10]
```

---

### Task 3-4: Calculate Metrics for Both Training and Testing Sets

**Description:**  
Compute accuracy, precision, recall, F1 score, and ROC-AUC for both the training and testing sets to evaluate model performance.

```python
# Compute metrics for the training set (excluding ROC-AUC)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred, pos_label='Y')  # Adjust pos_label as needed
train_recall = recall_score(y_train, y_train_pred, pos_label='Y')
train_f1 = f1_score(y_train, y_train_pred, pos_label='Y')

# Compute metrics for the testing set (excluding ROC-AUC)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, pos_label='Y')  # Adjust pos_label as needed
test_recall = recall_score(y_test, y_test_pred, pos_label='Y')
test_f1 = f1_score(y_test, y_test_pred, pos_label='Y')

# Display metrics for both sets
{
    "Train Metrics": {
        "Accuracy": train_accuracy,
        "Precision": train_precision,
        "Recall": train_recall,
        "F1 Score": train_f1,
    },
    "Test Metrics": {
        "Accuracy": test_accuracy,
        "Precision": test_precision,
        "Recall": test_recall,
        "F1 Score": test_f1,
    }
}
```
---

### Task 3-5: Calculate and Plot ROC-AUC for Both Training and Testing Sets

**Description:**  
Calculate ROC-AUC for both the training and testing sets and plot the ROC curve to visualize the model's discriminatory ability.

```python
from sklearn.metrics import roc_curve, auc

# Compute ROC curve and AUC for the training set
train_fpr, train_tpr, _ = roc_curve(y_train, logistic_model.predict_proba(X_train)[:, 1], pos_label='Y')
train_roc_auc = auc(train_fpr, train_tpr)

# Compute ROC curve and AUC for the testing set
test_fpr, test_tpr, _ = roc_curve(y_test, logistic_model.predict_proba(X_test)[:, 1], pos_label='Y')
test_roc_auc = auc(test_fpr, test_tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(train_fpr, train_tpr, label=f"Train ROC curve (area = {train_roc_auc:.2f})")
plt.plot(test_fpr, test_tpr, label=f"Test ROC curve (area = {test_roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()
```
---

## Main Task 4: Metrics Compilation

### Task 4-1: Compile Metrics into a Summary Table

**Description:**  
Create a pandas DataFrame to compile all metrics into a structured table for easier interpretation.

```python
# Create a summary table
metrics_summary = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
    'Value': [accuracy, precision, recall, f1, roc_auc]
})

# Display the summary table
metrics_summary
```
