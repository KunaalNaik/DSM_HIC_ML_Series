### **Main Task 1: Data Cleaning and Preprocessing**

---

### **Task 1-1: Import Libraries and Load the Dataset**

**Instructions:**
1. Import `pandas` and `numpy` for data handling.
2. Import `SimpleImputer`, `StandardScaler`, `OrdinalEncoder` from `sklearn.preprocessing` for preprocessing.
3. Import `train_test_split` from `sklearn.model_selection` to split data.
4. Import `LogisticRegression`, `LassoCV`, `RFE` for modeling and feature selection.
5. Import `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score` from `sklearn.metrics` for evaluation.
6. Import `matplotlib.pyplot` and `seaborn` for visualization.
7. Load the dataset using `pd.read_csv()` and store it in `data`.
8. Display the first five rows using `.head()`.

---

### **Task 1-2: Preprocess Numerical Features (Impute Missing Values, Remove Outliers, Scale Features)**

**Instructions:**
1. Identify numerical features by selecting columns with numerical data types.
2. Remove non-feature columns (e.g., `Loan_ID`).
3. Create an instance of `SimpleImputer` with `strategy='median'` and fit it on `numerical_features`.
4. Transform `numerical_features` using the fitted imputer.
5. Remove outliers using the IQR method:
   - Compute Q1 (25th percentile) and Q3 (75th percentile).
   - Calculate the IQR as `Q3 - Q1`.
   - Remove values outside the range `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`.
6. Scale numerical features using `StandardScaler`:
   - Fit `StandardScaler` on `numerical_features`.
   - Transform `numerical_features` using the fitted scaler.

---

### **Task 1-3: Preprocess Categorical Features (Impute Missing Values, Handle Rare Categories, Encode Features)**

**Instructions:**
1. Identify categorical features by selecting object-type columns.
2. Remove the target variable from the categorical feature list.
3. Create an instance of `SimpleImputer` with `strategy='most_frequent'` and fit it on `categorical_features`.
4. Transform `categorical_features` using the fitted imputer.
5. Handle rare categories:
   - Define a threshold for rare categories (e.g., `threshold=10`).
   - Replace categories that occur less than the threshold with `"Other"`.
6. Create an instance of `OrdinalEncoder` and fit it on `categorical_features`.
7. Transform categorical columns using the fitted encoder.

---

## **Main Task 2: Baseline Model Building**

---

### **Task 2-1: Split Data into Training and Testing Sets**

**Instructions:**
1. Define the feature set `X` as all preprocessed categorical and numerical features.
2. Define the target variable `y` as `Loan_Status`.
3. Use `train_test_split()` to split the dataset into:
   - `X_train`, `X_test` for features.
   - `y_train`, `y_test` for target labels.
   - `test_size=0.2` for an 80-20 split.
   - `random_state=42` for reproducibility.

---

### **Task 2-2: Train Logistic Regression Model with All Features**

**Instructions:**
1. Create an instance of `LogisticRegression`.
2. Train the model using `.fit()` with `X_train` and `y_train`.

---

### **Task 2-3: Evaluate Model and Show Feature Importance**

**Instructions:**
1. Use `.predict()` to generate predictions on `X_test`.
2. Compute classification metrics:
   - `accuracy_score()`, `precision_score()`, `recall_score()`, `f1_score()`, `roc_auc_score()`.
3. Extract model coefficients for feature importance.
4. Plot a horizontal bar chart to visualize feature importance.

---

## **Main Task 3: Filter-Based Feature Selection**

---

### **Task 3-1: Apply Filter Methods (Correlation Threshold)**

**Instructions:**
1. Compute the correlation matrix for `X`.
2. Identify features with high correlation (above `0.8` threshold).
3. Remove highly correlated features.

---

### **Task 3-2: Train Model with Filtered Features and Evaluate Performance**

**Instructions:**
1. Train a logistic regression model using only the filtered features.
2. Use `.predict()` on the filtered `X_test`.
3. Compute classification metrics and display the results.

---

## **Main Task 4: VIF-Based Feature Selection**

---

### **Task 4-1: Calculate Variance Inflation Factor (VIF)**

**Instructions:**
1. Compute VIF for numerical features using `variance_inflation_factor`.
2. Create a DataFrame to store the feature names and their VIF values.

---

### **Task 4-2: Remove Features with High VIF, Train Model, and Evaluate Performance**

**Instructions:**
1. Remove numerical features with high VIF values (above `5`).
2. Train a logistic regression model using only the reduced feature set.
3. Use `.predict()` on the test set.
4. Compute classification metrics and display the results.

---

## **Main Task 5: Wrapper-Based Feature Selection**

---

### **Task 5-1: Apply Recursive Feature Elimination (RFE)**

**Instructions:**
1. Apply Recursive Feature Elimination (RFE) with `LogisticRegression()`.
2. Select the top 10 most important features.

---

### **Task 5-2: Train Model with RFE-Selected Features and Evaluate Performance**

**Instructions:**
1. Train a logistic regression model using only the features selected by RFE.
2. Use `.predict()` on the test set.
3. Compute classification metrics and display the results.

---

## **Main Task 6: Embedded Feature Selection**

---

### **Task 6-1: Apply Lasso Regression for Feature Selection**

**Instructions:**
1. Apply Lasso Regression (`LassoCV(cv=5)`) to perform feature selection.
2. Identify features with non-zero coefficients.

---

### **Task 6-2: Train Model with Lasso-Selected Features and Evaluate Performance**

**Instructions:**
1. Train a logistic regression model using only the features selected by Lasso Regression.
2. Use `.predict()` on the test set.
3. Compute classification metrics and display the results.

---

## **Main Task 7: Model Comparison and Final Selection**

---

### **Task 7-1: Compare Model Performance Across Feature Selection Methods**

**Instructions:**
1. Compare classification metrics (Accuracy, Precision, Recall, F1 Score, ROC-AUC) across:
   - Baseline Model
   - Filter-Based Model
   - VIF-Based Model
   - RFE Model
   - Lasso Model

---

### **Task 7-2: Select the Best Feature Selection Method Based on Performance**

**Instructions:**
1. Identify the feature selection method that provides the highest **accuracy and ROC-AUC**.
2. Store the best method for final model training.

---

### **Task 7-3: Train and Evaluate Final Model with Best Feature Subset**

**Instructions:**
1. Select the dataset using the best feature selection method.
2. Train a final logistic regression model using the selected feature set.
3. Use `.predict()` on the final test set.
4. Compute and display final classification metrics.

---

### **Task 7-4: Evaluate and Present the Final Model Results**

**Instructions:**
1. Print the final model performance.
2. Plot feature importance for the final model.
3. Summarize findings on feature selection impact.

---