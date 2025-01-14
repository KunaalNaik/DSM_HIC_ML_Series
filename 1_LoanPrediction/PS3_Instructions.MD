### Main Task 1: Data Preparation

---

### Task 1-1: Import Libraries and Load the Dataset (leave code)

**Instructions:**
1. Import the following libraries:
   - `pandas` for data handling.
   - `SimpleImputer` from `sklearn.impute` for handling missing values.
   - `train_test_split` from `sklearn.model_selection` for splitting the dataset.
   - `LogisticRegression` from `sklearn.linear_model` for the logistic regression model.
   - `StandardScaler` from `sklearn.preprocessing` for scaling numerical features.
   - Metrics (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`) from `sklearn.metrics` for evaluation.
   - `matplotlib.pyplot` and `seaborn` for visualizations.
2. Use `pd.read_csv()` to load the dataset into a DataFrame named `data`.
3. Use `.head()` to preview the dataset.

---

### Task 1-2: Identify Numerical Features

**Instructions:**
1. Use `.select_dtypes(include=['int64', 'float64'])` on the DataFrame to filter numerical columns.
2. Remove the ID column (`Loan_ID`) from the list of numerical features.
3. Save the resulting list as `numerical_features`.

---

### Task 1-3: Check Missing Values

**Instructions:**
1. Use `.isnull()` on the numerical columns to find missing values.
2. Use `.sum()` to count missing values for each column.
3. Save the result as `missing_values`.

---

### Task 1-4: Treat Missing Values Using SimpleImputer

**Instructions:**
1. Create an instance of `SimpleImputer` with `strategy='median'`.
2. Use `.fit()` to fit the imputer on the numerical features.
3. Use `.transform()` to replace missing values in the numerical features.
4. Replace the original numerical columns in the `data` DataFrame with the transformed values.

---

### Main Task 2: Preprocessing

---

### Task 2-1: Check for Outliers in Numerical Features 

**Instructions:**
1. Initialize a dictionary named `outliers` to store the count of outliers for each numerical feature.
2. For each numerical feature:
   - Compute Q1 (25th percentile) and Q3 (75th percentile).
   - Compute the IQR as `Q3 - Q1`.
   - Calculate lower and upper bounds as `Q1 - 1.5 * IQR` and `Q3 + 1.5 * IQR`, respectively.
   - Count the number of rows with values outside these bounds and add them to the `outliers` dictionary.

---

### Task 2-2: Remove Outliers Using the IQR Method

**Instructions:**
1. For each numerical feature:
   - Compute Q1, Q3, and IQR.
   - Calculate lower and upper bounds.
   - Filter the DataFrame to include only rows within the bounds for the feature.

---

### Task 2-3: Check for Normal Distribution of Numerical Features

**Instructions:**
1. For each numerical feature:
   - Use `sns.histplot()` or `.plot(kind='hist')` to create a histogram.
   - Use `plt.title()` to add a title with the feature's name.
   - Use `plt.show()` to display the plot.

---

### Task 2-4: Scale Numerical Features Using StandardScaler

**Instructions:**
1. Create an instance of `StandardScaler`.
2. Use `.fit()` to fit the scaler on the numerical features.
3. Use `.transform()` to scale the numerical features.
4. Replace the original numerical columns in the `data` DataFrame with the scaled values.

---

### Main Task 3: Model Building and Evaluation

---

### Task 3-3: Predict on Both Training and Testing Sets

**Instructions:**
1. Use `.predict()` on the trained logistic regression model with `X_train` to predict the target variable for the training set. Save the result as `y_train_pred`.
2. Use `.predict()` on the model with `X_test` to predict the target variable for the testing set. Save the result as `y_test_pred`.

---

### Task 3-4: Calculate Metrics for Both Training and Testing Sets

**Instructions:**
1. Compute metrics for the training set:
   - Use `accuracy_score()` with `y_train` and `y_train_pred` to compute accuracy.
   - Use `precision_score()`, `recall_score()`, and `f1_score()` to compute precision, recall, and F1 score.
2. Repeat the same steps for the testing set with `y_test` and `y_test_pred`.
3. Save the metrics as separate variables for both training and testing sets.

---

### Task 3-5: Calculate and Plot ROC-AUC for Both Training and Testing Sets (leave code)

**Instructions:**
1. Compute the predicted probabilities for `X_train` and `X_test` using `.predict_proba()`.
2. Use `roc_curve()` to calculate the false positive rate (FPR), true positive rate (TPR), and thresholds for both training and testing sets.
3. Use `auc()` to compute the area under the curve (AUC) for both sets.
4. Plot the ROC curves using `plt.plot()` for both training and testing sets on the same graph.
5. Add labels, a legend, and a title to the plot using `plt.xlabel()`, `plt.ylabel()`, `plt.legend()`, and `plt.title()`.

---

### Main Task 4: Metrics Compilation

---

### Task 4-1: Compile Metrics into a Summary Table

**Instructions:**
1. Create a dictionary with metric names (`'Accuracy'`, `'Precision'`, `'Recall'`, `'F1 Score'`, `'ROC-AUC'`) as keys.
2. Assign the corresponding values for training and testing metrics to the dictionary.
3. Convert the dictionary into a pandas DataFrame named `metrics_summary`.
4. Use `.head()` or print the DataFrame to display the metrics summary.