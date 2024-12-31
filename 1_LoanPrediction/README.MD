### **Main Task 1: Loading and Previewing the Dataset**

---

#### **Sub-Task 1-1: Import Libraries and Load the Dataset**

**Description**:  
The objective is to import the `pandas` library and load the dataset `train_loan_prediction.csv` into a DataFrame for further analysis.

---

**Instructions**:
1. Import the `pandas` library.
2. Use `pd.read_csv()` to load the dataset.
3. Assign the dataset to a variable named `data`.

---

**Code**:
```python
# Step 1: Import the pandas library
import pandas as pd  # pandas is a library for data manipulation and analysis

# Step 2: Load the dataset into a DataFrame
data = pd.read_csv('/mnt/data/train_loan_prediction.csv')  # Replace with your file path

# Variable 'data' now holds the dataset
data  # Display the dataset for validation
```

---

#### **Sub-Task 1-2: Preview the Dataset**

**Description**:  
Display the first few rows of the dataset to inspect its structure and understand its contents.

---

**Instructions**:
1. Use the `.head()` method to display the first five rows of the dataset.
2. Save the result in a variable named `data_preview`.

---

**Code**:
```python
# Step 1: Preview the dataset
data_preview = data.head()  # Get the first 5 rows of the dataset

# Variable 'data_preview' stores the result
data_preview  # Output the result for validation
```

---

#### **Sub-Task 1-3: Get Dataset Index Labels**

**Description**:  
Retrieve the index labels of the dataset to understand how rows are indexed.

---

**Instructions**:
1. Use the `.index` attribute to get the index labels.
2. Save the result in a variable named `index_labels`.

---

**Code**:
```python
# Step 1: Get index labels of the dataset
index_labels = data.index  # Retrieve the index labels

# Variable 'index_labels' stores the result
index_labels  # Output the result for validation
```

---

### **Main Task 2: Basic Dataset Information**

---

#### **Sub-Task 2-1: Find the Total Number of Rows**

**Description**:  
Determine the total number of rows in the dataset.

---

**Instructions**:
1. Use the `.shape` attribute to get the number of rows.
2. Save the result in a variable named `num_rows`.

---

**Code**:
```python
# Step 1: Find the total number of rows
num_rows = data.shape[0]  # Shape[0] returns the number of rows

# Variable 'num_rows' stores the result
num_rows  # Output the result for validation
```

---

#### **Sub-Task 2-2: Find the Total Number of Columns**

**Description**:  
Determine the total number of columns in the dataset.

---

**Instructions**:
1. Use the `.shape` attribute to get the number of columns.
2. Save the result in a variable named `num_columns`.

---

**Code**:
```python
# Step 1: Find the total number of columns
num_columns = data.shape[1]  # Shape[1] returns the number of columns

# Variable 'num_columns' stores the result
num_columns  # Output the result for validation
```

---

#### **Sub-Task 2-3: Get Data Types of All Columns**

**Description**:  
Retrieve the data types of all columns in the dataset to understand what kind of data each column contains.

---

**Instructions**:
1. Use the `.dtypes` attribute to get the data types.
2. Save the result in a variable named `data_types`.

---

**Code**:
```python
# Step 1: Get data types of all columns
data_types = data.dtypes  # Retrieve the data types of all columns

# Variable 'data_types' stores the result
data_types  # Output the result for validation
```

---

#### **Sub-Task 2-4: List All Column Names**

**Description**:  
List all the column names in the dataset for reference.

---

**Instructions**:
1. Use the `.columns` attribute to get the column names.
2. Save the result in a variable named `columns_list`.

---

**Code**:
```python
# Step 1: List all column names
columns_list = data.columns.tolist()  # Get column names as a list

# Variable 'columns_list' stores the result
columns_list  # Output the result for validation
```

---

#### **Sub-Task 2-5: Get Dataset Information**

**Description**:  
Retrieve metadata about the dataset, including column names, data types, and non-null counts.

---

**Instructions**:
1. Use the `.info()` method to display dataset information.
2. Save the result in a variable named `data_info`.

---

**Code**:
```python
# Step 1: Get dataset information
data_info = data.info()  # Display metadata about the dataset

# Variable 'data_info' stores the result
data_info  # Output the result for validation
```

---

### **Main Task 3: Column-Level Analysis**

---

#### **Sub-Task 3-1: Identify Numerical Columns**

**Description**:  
Identify all the numerical columns in the dataset based on their data types (`int64` or `float64`).

---

**Instructions**:
1. Use `.select_dtypes(include=[])` to filter columns with numerical data types.
2. Save the list of numerical columns in a variable named `numerical_columns`.

---

**Code**:
```python
# Step 1: Identify numerical columns
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Variable 'numerical_columns' stores the result
numerical_columns  # Output the result for validation
```

---

#### **Sub-Task 3-2: Identify Categorical Columns**

**Description**:  
Identify all the categorical columns in the dataset based on their data types (`object` or `category`).

---

**Instructions**:
1. Use `.select_dtypes(include=[])` to filter columns with categorical data types.
2. Save the list of categorical columns in a variable named `categorical_columns`.

---

**Code**:
```python
# Step 1: Identify categorical columns
categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()

# Variable 'categorical_columns' stores the result
categorical_columns  # Output the result for validation
```

---

#### **Sub-Task 3-3: Count Unique Values in Each Column**

**Description**:  
Count the number of unique values in each column to understand the variety of data in the dataset.

---

**Instructions**:
1. Use `.nunique()` to find the count of unique values in each column.
2. Save the result in a variable named `unique_values`.

---

**Code**:
```python
# Step 1: Count unique values in each column
unique_values = data.nunique()

# Variable 'unique_values' stores the result
unique_values  # Output the result for validation
```

---

#### **Sub-Task 3-4: Count Non-Null Values in Each Column**

**Description**:  
Determine the number of non-null (valid) values in each column.

---

**Instructions**:
1. Use `.count()` to count non-null values in each column.
2. Save the result in a variable named `non_null_counts`.

---

**Code**:
```python
# Step 1: Count non-null values in each column
non_null_counts = data.count()

# Variable 'non_null_counts' stores the result
non_null_counts  # Output the result for validation
```

---

#### **Sub-Task 3-5: Check for Missing Values**

**Description**:  
Identify columns with missing values and count the number of missing entries in each column.

---

**Instructions**:
1. Use `.isnull().sum()` to calculate the number of missing values in each column.
2. Save the result in a variable named `missing_values`.

---

**Code**:
```python
# Step 1: Check for missing values
missing_values = data.isnull().sum()

# Variable 'missing_values' stores the result
missing_values  # Output the result for validation
```

---

### **Main Task 4: Data Statistics and Memory Usage**

---

#### **Sub-Task 4-1: Get Summary Statistics for Numerical Columns**

**Description**:  
Generate summary statistics (e.g., mean, median, min, max) for all numerical columns in the dataset.

---

**Instructions**:
1. Use `.describe()` to generate summary statistics for numerical columns.
2. Save the result in a variable named `data_summary`.

---

**Code**:
```python
# Step 1: Get summary statistics for numerical columns
data_summary = data.describe()

# Variable 'data_summary' stores the result
data_summary  # Output the result for validation
```

---

#### **Sub-Task 4-2: Get Descriptive Statistics for All Columns**

**Description**:  
Generate descriptive statistics for all columns, including categorical ones.

---

**Instructions**:
1. Use `.describe(include='all')` to get descriptive statistics for all columns.
2. Save the result in a variable named `full_summary`.

---

**Code**:
```python
# Step 1: Get descriptive statistics for all columns
full_summary = data.describe(include='all')

# Variable 'full_summary' stores the result
full_summary  # Output the result for validation
```

---

#### **Sub-Task 4-3: Check Memory Usage**

**Description**:  
Analyze the memory usage of the dataset to understand its resource footprint.

---

**Instructions**:
1. Use `.memory_usage()` to check memory usage for each column in bytes.
2. Save the result in a variable named `memory_usage`.

---

**Code**:
```python
# Step 1: Check memory usage of the dataset
memory_usage = data.memory_usage()

# Variable 'memory_usage' stores the result
memory_usage  # Output the result for validation
```

---

### **Main Task 5: Sampling and Value Distribution**

---

#### **Sub-Task 5-1: Get Random Samples from the Dataset**

**Description**:  
Extract random rows from the dataset to inspect its contents and get a better understanding of the data.

---

**Instructions**:
1. Use `.sample()` to randomly sample rows from the dataset.
2. Save the result in a variable named `random_sample`.

---

**Code**:
```python
# Step 1: Get random samples from the dataset
random_sample = data.sample(5)  # Replace 5 with the number of rows you want to sample

# Variable 'random_sample' stores the result
random_sample  # Output the result for validation
```

---

#### **Sub-Task 5-2: Get Value Counts for a Column**

**Description**:  
Find the distribution of unique values for a specific column.

---

**Instructions**:
1. Choose a column of interest (e.g., `'ColumnName'`).
2. Use `.value_counts()` to count the frequency of each unique value in that column.
3. Save the result in a variable named `value_counts_example`.

---

**Code**:
```python
# Step 1: Get value counts for a specific column
# Replace 'ColumnName' with the name of the column you want to analyze
value_counts_example = data['ColumnName'].value_counts()

# Variable 'value_counts_example' stores the result
value_counts_example  # Output the result for validation
```

---