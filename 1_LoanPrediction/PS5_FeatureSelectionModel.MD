### **Instructions for Tasks**

---

#### **Main Task 1: Data Preparation**

---

##### **1-1: Import Libraries and Load the Dataset**
1. Import `pandas` for data handling.
2. Import `matplotlib.pyplot` and `seaborn` for visualizations.
3. Import `SimpleImputer`, `train_test_split`, `LogisticRegression`, and metrics from `sklearn`.
4. Use `pd.read_csv()` to load the dataset into a DataFrame named `data`.
5. Preview the dataset using `.head()`.

---

##### **1-2: Identify Numerical Features**
1. Use `.select_dtypes(include=['int64', 'float64'])` on the DataFrame to identify numerical columns.
2. Remove the ID column (e.g., `Loan_ID`) from the list of numerical columns.
3. Save the resulting list as `numerical_features`.

---

##### **1-3: Check Missing Values**
1. Use `.isnull()` on the numerical columns to find missing values.
2. Use `.sum()` to count missing values for each column.
3. Save the result as `missing_values`.

---

##### **1-4: Treat Missing Values Using SimpleImputer**
1. Create an instance of `SimpleImputer` with `strategy='median'`.
2. Use `.fit_transform()` to impute missing values in numerical columns.
3. Convert the output back to a DataFrame and save it as `numerical_data`.

---

#### **Main Task 2: Preprocessing**

---

##### **2-1: Check for Outliers in Numerical Features**
1. For each numerical column, calculate the first (Q1) and third quartile (Q3).
2. Compute the Interquartile Range (IQR) as `Q3 - Q1`.
3. Identify outliers as values below `Q1 - 1.5 * IQR` or above `Q3 + 1.5 * IQR`.
4. Display the number of outliers for each column.

---

##### **2-2: Remove Outliers Using the IQR Method**
1. Remove rows containing outliers in numerical columns based on the IQR calculation.
2. Save the cleaned DataFrame as `cleaned_data`.

---

##### **2-3: Check for Normal Distribution of Numerical Features**
1. Use `.plot(kind='hist')` or seaborn's `sns.histplot()` to visualize the distribution of each numerical column.
2. Check if the distributions are approximately normal (bell-shaped).

---

##### **2-4: Scale Numerical Features Using StandardScaler**
1. Import `StandardScaler` from `sklearn.preprocessing`.
2. Create an instance of `StandardScaler`.
3. Use `.fit_transform()` on numerical columns to scale them.
4. Convert the output back to a DataFrame and save it as `scaled_data`.

---

#### **Main Task 3: Model Building and Evaluation**

---

##### **3-1: Split Data into Training and Testing Sets**
1. Use `train_test_split()` to split `scaled_data` (features) and the target variable into training and testing sets.
2. Use `test_size=0.2` for an 80-20 split.
3. Set `random_state=42` for reproducibility.
4. Save the splits as `X_train`, `X_test`, `y_train`, and `y_test`.

---

##### **3-2: Train Logistic Regression Model**
1. Create an instance of `LogisticRegression`.
2. Use `.fit()` to train the model on `X_train` and `y_train`.

---

##### **3-3: Predict on the Testing Set**
1. Use `.predict()` on the trained logistic regression model with `X_test`.
2. Save the predictions as `y_pred`.

---

##### **3-4: Calculate Accuracy**
1. Use `accuracy_score()` with `y_test` and `y_pred` to compute accuracy.
2. Save the result as `accuracy`.

---

##### **3-5: Calculate Precision, Recall, and F1 Score**
1. Use `precision_score()` with `y_test` and `y_pred` to calculate precision.
2. Use `recall_score()` with `y_test` and `y_pred` to calculate recall.
3. Use `f1_score()` with `y_test` and `y_pred` to calculate the F1 score.
4. Save each result in corresponding variables: `precision`, `recall`, and `f1`.

---

##### **3-6: Calculate ROC-AUC**
1. Use `roc_auc_score()` with `y_test` and the predicted probabilities (`.predict_proba()` on `X_test`).
2. Save the result as `roc_auc`.

---

#### **Main Task 4: Metrics Compilation**

---

##### **4-1: Compile Metrics into a Summary Table**
1. Create a dictionary with keys as metric names (`'Accuracy'`, `'Precision'`, `'Recall'`, `'F1 Score'`, `'ROC-AUC'`).
2. Use their corresponding computed values as dictionary values.
3. Convert the dictionary into a pandas DataFrame named `metrics_summary`.
4. Print `metrics_summary` to display the compiled metrics.

---