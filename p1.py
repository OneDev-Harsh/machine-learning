import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('insurance.csv')

# Exploratory Data Analysis (EDA)

df.shape    #returns a tuple: (no. of rows, no. of columns) of the data
df.head()   #returns the first 5 rows of the data
df.info()   #returns the columns and their datatypes in the dataframe
df.describe()   #returns a dataframe containing count, mean, std, min, max, 25%, 50%, 75% of each column containg numeric data
df.isnull().sum()   #returns the total number of null values in each column

# plotting all numeric columns in a histogram
numeric_columns = ['age', 'bmi', 'children', 'charges']
for col in numeric_columns:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, bins=20)
plt.show()

# histogram plot for other data

sns.countplot(x=df['children'])
plt.show()

sns.countplot(x=df['sex'])
plt.show()

sns.countplot(x=df['smoker'])
plt.show()

# boxplot for data

for col in numeric_columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])

# heatmap to see how much columns are co-related

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True)

plt.show()

# Data Cleaning & Preprocessing

df_cleaned = df.copy()

df_cleaned = df.drop_duplicates(inplace=True)   # returns data after removal of duplicate rows

df_cleaned['sex'] = df_cleaned['sex'].map({"male":0, "female":1})   # label encoding the sex column male->0 & female->1

df_cleaned['smoker'] = df_cleaned['smoker'].map({"yes": 1, "no":0}) # label encoding the smoker column no->0 & yes->1

df_cleaned.rename(columns={
    'sex': 'is_female',
    'smoker': 'is_smoker'
}, inplace=True)    # renames the specified columns to the given one

df_cleaned = pd.get_dummies(df_cleaned, columns=['region'], drop_first=True)    # removes the region columns and creates new columns in one hot encoding
df_cleaned.astype(int)  # converts the datatypes of all columns to int

# Feature Engineering and Extraction

# creating a new column named 'bmi_category' and the assigning values depending on bmi range
df_cleaned['bmi_category'] = pd.cut(
    df_cleaned['bmi'],
    bins=[0,18.5,24.9,29.9,float('inf')],
    labels=['Underweight', 'Normal', 'Overweight', 'Obese']
)

df_cleaned = pd.get_dummies(df_cleaned, columns=['bmi_category'], drop_first=True)
df_cleaned = df_cleaned.astype(int)

# scaling the specified columns

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
cols=['age', 'bmi', 'children']
df_cleaned[cols] = scaler.fit_transform(df_cleaned[cols])

# calculating correlation of different columns with the target column using pearson's correlation

from scipy.stats import pearsonr
 
selected_features = [
    'age', 'is_female', 'bmi', 'children', 'is_smoker',
    'region_northwest', 'region_southeast', 'region_southwest',
    'bmi_category_Normal', 'bmi_category_Overweight', 'bmi_category_Obese'
]

correlations = {
    feature: pearsonr(df_cleaned[feature], df_cleaned['charges'])[0]
    for feature in selected_features
}

correlation_df = pd.DataFrame(list(correlations.items()), columns=["Feature", "Pearson Correlation"])
correlation_df.sort_values(by="Pearson Correlation", ascending=False)

from scipy.stats import chi2_contingency

cat_features = [
    'is_female', 'is_smoker',
    'region_northwest', 'region_southeast', 'region_southwest',
    'bmi_category_Normal', 'bmi_category_Overweight', 'bmi_category_Obese'
]

alpha=0.05

df_cleaned['charges_bin'] = pd.qcut(df_cleaned['charges'], q=4, labels=False)
chi2_results = {}

for col in cat_features:
    contingency = pd.crosstab(df_cleaned[col], df_cleaned['charges_bin'])
    chi2_stat, p_val, _, _ = chi2_contingency(contingency)
    decision = 'Reject Null (Keep Feature)' if p_val<alpha else 'Accept Null (Reject Feature)'
    chi2_results[col] = {
        'chi2_statistic': chi2_stat,
        'p_value': p_val,
        'Decision': decision
    }

chi2_df = pd.DataFrame(chi2_results).T
chi2_df = chi2_df.sort_values(by='p_value')

final_df = df_cleaned['age', 'is_Female', 'bmi', 'children', 'is_smoker', 'charges', 'region_southeast', 'bmi_category_Obese']