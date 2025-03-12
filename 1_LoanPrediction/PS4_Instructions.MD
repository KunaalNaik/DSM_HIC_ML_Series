## **Main Task 1: Data Cleaning and Preprocessing**

---

### **Task 1-1: Import Libraries and Load the Dataset**

**Instructions:**
1. Import the necessary libraries for:
   - Data handling (`pandas`, `numpy`).
   - Data preprocessing (`SimpleImputer`, `StandardScaler`, `OrdinalEncoder`, `OneHotEncoder`).
   - Feature selection (`MeanEncoder`, `LeaveOneOutEncoder` from `category_encoders`).
   - Model training and evaluation (`train_test_split`, `LogisticRegression`, `accuracy_score`, etc.).
   - Visualization (`matplotlib.pyplot`, `seaborn`).
2. Load the dataset using `pd.read_csv()`.
3. Display the first five rows using `.head()`.

---

### **Task 1-2: Preprocess Numerical Features (Impute Missing Values, Remove Outliers, Scale Features)**

**Instructions:**
1. Identify numerical features by selecting columns with numerical data types.
2. Remove any non-feature columns (e.g., `Loan_ID`).
3. Use `SimpleImputer` with `strategy='median'` to replace missing values:
   - Fit the imputer on `numerical_features`.
   - Transform the numerical columns using the fitted imputer.
4. Detect and remove outliers using the IQR method:
   - Compute Q1 (25th percentile) and Q3 (75th percentile).
   - Calculate the IQR as `Q3 - Q1`.
   - Remove values outside the range `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`.
5. Scale numerical features using `StandardScaler`:
   - Fit the scaler on `numerical_features`.
   - Transform the numerical columns using the fitted scaler.

---

### **Task 1-3: Preprocess Categorical Features (Impute Missing Values, Handle Rare Categories)**

**Instructions:**
1. Identify categorical features by selecting object-type columns.
2. Remove the target variable from the categorical feature list.
3. Use `SimpleImputer` with `strategy='most_frequent'` to replace missing values:
   - Fit the imputer on `categorical_features`.
   - Transform the categorical columns using the fitted imputer.
4. Handle rare categories:
   - Define a threshold for rare categories (e.g., `threshold=10`).
   - Replace categories that occur less than the threshold with `"Other"`.

---

## **Main Task 2: Categorical Feature Encoding Methods**

---

### **Task 2-1: Apply One-Hot Encoding (OHE)**

**Instructions:**
1. Create an instance of `OneHotEncoder` with `drop='first'` to avoid multicollinearity.
2. Fit the encoder on `categorical_features`.
3. Transform categorical columns into one-hot encoded columns.
4. Convert the encoded features into a pandas DataFrame.
5. Concatenate the transformed categorical features with numerical features.
6. Display the updated dataset.

---

### **Task 2-2: Apply Ordinal Encoding**

**Instructions:**
1. Create an instance of `OrdinalEncoder`.
2. Fit the encoder on `categorical_features`.
3. Transform categorical columns into ordinal-encoded values.
4. Display the updated dataset.

---

### **Task 2-3: Apply Mean Encoding (Using `category_encoders`)**

**Instructions:**
1. Create an instance of `MeanEncoder` from `category_encoders`.
2. Fit the encoder on `categorical_features` using the target variable.
3. Transform categorical columns into mean-encoded values.
4. Display the updated dataset.

---

### **Task 2-4: Apply Leave-One-Out Encoding (Using `category_encoders`)**

**Instructions:**
1. Create an instance of `LeaveOneOutEncoder` from `category_encoders`.
2. Fit the encoder on `categorical_features` using the target variable.
3. Transform categorical columns into Leave-One-Out encoded values.
4. Display the updated dataset.

---

## **Main Task 3: Baseline Model Building**

---

### **Task 3-1: Split Data into Training and Testing Sets**

**Instructions:**
1. Define the feature set `X` as all preprocessed categorical and numerical features.
2. Define the target variable `y` as `Loan_Status`.
3. Use `train_test_split()` to split the dataset into:
   - `X_train`, `X_test` for features.
   - `y_train`, `y_test` for target labels.
   - `test_size=0.2` for an 80-20 split.
   - `random_state=42` for reproducibility.

---

### **Task 3-2: Train Logistic Regression Model with All Features**

**Instructions:**
1. Create an instance of `LogisticRegression`.
2. Train the model using `.fit()` with `X_train` and `y_train`.

---

### **Task 3-3: Evaluate Model and Show Feature Importance**

**Instructions:**
1. Use `.predict()` to generate predictions on `X_test`.
2. Compute classification metrics:
   - `accuracy_score()`, `precision_score()`, `recall_score()`, `f1_score()`, `roc_auc_score()`.
3. Extract model coefficients for feature importance.
4. Plot a horizontal bar chart to visualize feature importance.

---

## **Main Task 4: Categorical Encoding Impact on Model Performance**

---

### **Task 4-1: Train and Evaluate Model with One-Hot Encoding**

**Instructions:**
1. Train a logistic regression model on `X_train_ohe`, `y_train_ohe`.
2. Use `.predict()` on `X_test_ohe` to generate predictions.
3. Compute classification metrics and display the results.

---

### **Task 4-2: Train and Evaluate Model with Ordinal Encoding**

**Instructions:**
1. Train a logistic regression model on `X_train_ordinal`, `y_train_ordinal`.
2. Use `.predict()` on `X_test_ordinal` to generate predictions.
3. Compute classification metrics and display the results.

---

### **Task 4-3: Train and Evaluate Model with Mean Encoding**

**Instructions:**
1. Train a logistic regression model on `X_train_mean`, `y_train_mean`.
2. Use `.predict()` on `X_test_mean` to generate predictions.
3. Compute classification metrics and display the results.

---

### **Task 4-4: Train and Evaluate Model with Leave-One-Out Encoding**

**Instructions:**
1. Train a logistic regression model on `X_train_loo`, `y_train_loo`.
2. Use `.predict()` on `X_test_loo` to generate predictions.
3. Compute classification metrics and display the results.

---

## **Main Task 5: Final Model Selection and Summary**

---

### **Task 5-1: Compare Model Performance Across Encoding Methods**

**Instructions:**
1. Compare the classification metrics (Accuracy, Precision, Recall, F1 Score, ROC-AUC) of:
   - One-Hot Encoding
   - Ordinal Encoding
   - Mean Encoding
   - Leave-One-Out Encoding

---

### **Task 5-2: Select the Best Encoding Method Based on Performance**

**Instructions:**
1. Identify the best encoding method based on the highest **accuracy and ROC-AUC**.
2. Store the best method for final model training.

---

### **Task 5-3: Train and Evaluate Final Model with Best Encoding Method**

**Instructions:**
1. Select the dataset using the best encoding method.
2. Train a final logistic regression model using the selected encoding method.
3. Use `.predict()` on the final test set.
4. Compute and display final classification metrics.

---