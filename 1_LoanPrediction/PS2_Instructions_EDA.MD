### **Main Task 1: Import and Prepare the Dataset**

---

#### **Sub-Task 1-1: Import Libraries and Load the Dataset**

##### **Instructions**
1. Use `import pandas as pd` to import the pandas library.
2. Use `import matplotlib.pyplot as plt` to import the matplotlib library.
3. Use `import seaborn as sns` to import the seaborn library.
4. Use `pd.read_csv()` to load the dataset into a variable named `data`.
5. Use `.head()` on `data` to display the first few rows of the dataset.

---

#### **Sub-Task 1-2: Check for Missing Values**

##### **Instructions**
1. Use `.isnull()` on `data` to check for missing values in the dataset.
2. Use `.sum()` to count the total number of missing values for each column.
3. Assign the result to a variable named `missing_values`.
4. Print `missing_values` to see the output.

---

#### **Sub-Task 1-3: Analyze the Target Variable (`Loan_Status`)**

##### **Instructions**
1. Use `.value_counts()` on `data['Loan_Status']` to count the occurrences of each class in the target variable.
2. Assign the result to a variable named `target_counts`.
3. Use `.plot(kind='bar')` on `target_counts` to create a bar plot.
4. Add a title to the plot using `plt.title()`.
5. Label the x-axis and y-axis using `plt.xlabel()` and `plt.ylabel()`.
6. Use `plt.show()` to display the plot.

---

### **Main Task 2: Numerical Feature Analysis**

---

#### **Sub-Task 2-1: Check Missing Values for `ApplicantIncome`**

##### **Instructions**
1. Use `.isnull()` on `data['ApplicantIncome']` to check for missing values in the column.
2. Use `.sum()` to count the total number of missing values.
3. Assign the result to a variable named `applicant_income_missing`.
4. Print `applicant_income_missing` to see the output.

---

#### **Sub-Task 2-2: Plot Distribution for `ApplicantIncome`**

##### **Instructions**
1. Use `.plot(kind='hist')` on `data['ApplicantIncome']` to create a histogram.
2. Add a title to the plot using `plt.title()`.
3. Use `plt.show()` to display the plot.

---

#### **Sub-Task 2-3: Detect Outliers for `ApplicantIncome`**

##### **Instructions**
1. Use `.plot(kind='box')` on `data['ApplicantIncome']` to create a box plot.
2. Add a title to the plot using `plt.title()`.
3. Use `plt.show()` to display the plot.

---

#### **Sub-Task 2-4: Compute Average `ApplicantIncome` for Each Target Class**

##### **Instructions**
1. Use `.groupby('Loan_Status')` on `data` to group the dataset by `Loan_Status`.
2. Use `['ApplicantIncome']` to select the column.
3. Use `.mean()` to compute the average for each group.
4. Assign the result to a variable named `applicant_income_avg`.
5. Print `applicant_income_avg` to see the output.

---

#### **Sub-Task 2-5 to 2-9**: Repeat Sub-Tasks 2-1 to 2-4 for `CoapplicantIncome` and `LoanAmount`  
Change the column name and variable name accordingly for each numerical feature.

---

### **Main Task 3: Categorical Feature Analysis**

---

#### **Sub-Task 3-1: Check Missing Values for Categorical Features**

##### **Instructions**
1. Create a list named `categorical_features` containing the categorical column names: `['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']`.
2. Use `.isnull()` on `data[categorical_features]` to check for missing values in these columns.
3. Use `.sum()` to count the total number of missing values for each column.
4. Assign the result to a variable named `categorical_missing`.
5. Print `categorical_missing` to see the output.

---

#### **Sub-Task 3-2: Plot Value Counts for `Gender`**

##### **Instructions**
1. Use `.value_counts()` on `data['Gender']` to count the occurrences of each category.
2. Use `.plot(kind='bar')` to create a bar plot.
3. Add a title to the plot using `plt.title()`.
4. Use `plt.show()` to display the plot.

---

#### **Sub-Task 3-3 to 3-4**: Repeat Sub-Task 3-2 for `Married` and `Education`  
Change the column name accordingly.

---

#### **Sub-Task 3-5: Analyze Relationship Between `Gender` and `Loan_Status`**

##### **Instructions**
1. Use `pd.crosstab(data['Gender'], data['Loan_Status'])` to create a crosstab.
2. Use `.plot(kind='bar', stacked=True)` on the crosstab to create a stacked bar plot.
3. Add a title to the plot using `plt.title()`.
4. Use `plt.show()` to display the plot.

---

#### **Sub-Task 3-6 to 3-7**: Repeat Sub-Task 3-5 for `Married` and `Education`  
Change the column name accordingly.

---

### **Main Task 4: Summary and Insights**

---

#### **Sub-Task 4-1: Summarize Insights for Numerical Features**

##### **Instructions**
1. Create a dictionary named `numerical_insights`.
2. Include keys for each numerical feature (e.g., `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`).
3. For each key, include:
   - The number of missing values using `.isnull().sum()`.
   - The distribution type (e.g., 'Positively skewed').
   - Whether outliers were detected (e.g., 'Yes' or 'No').
   - The average values grouped by `Loan_Status` using `.groupby().mean()`.
4. Print `numerical_insights` to see the summary.

---

#### **Sub-Task 4-2: Summarize Insights for Categorical Features**

##### **Instructions**
1. Create a dictionary named `categorical_insights`.
2. Include keys for each categorical feature (e.g., `Gender`, `Married`, `Education`).
3. For each key, include:
   - The number of missing values using `.isnull().sum()`.
   - The category distribution using `.value_counts().to_dict()`.
   - The observed relationship with `Loan_Status` (e.g., 'Significant' or 'Weak').
4. Print `categorical_insights` to see the summary.

---

#### **Sub-Task 4-3: Conclude Features Impacting the Target Variable**

##### **Instructions**
1. Create a dictionary named `impactful_features`.
2. Add two keys: `numerical_features` and `categorical_features`.
3. List the most impactful numerical and categorical features under each key.
4. Print `impactful_features` to see the conclusion.

---