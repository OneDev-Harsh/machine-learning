import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('heart.csv')

df.head()
df.columns
df.shape
df.info()
df.describe()

df.duplicated().sum()
df['HeartDisease'].value_counts().plot(kind='bar')

df.isnull().sum()

def plotting(var, num):
    plt.subplot(2,2,num)
    sns.histplot(df[var], kde=True)

plotting('Age',1)
plotting('RestingBP',2)
plotting('Cholesterol',3)
plotting('MaxHR',4)

plt.tight_layout()
plt.show()

df['Cholesterol'].value_counts()

cholesterol_mean = df.loc[df['Cholesterol'] != 0, 'Cholesterol'].mean()

df['Cholesterol'] = df['Cholesterol'].replace(0,round(cholesterol_mean))

plt.subplot(1,2,1)
sns.histplot(df['Cholesterol'], kde=True)

resting_bp_mean = df.loc[df['RestingBP'] != 0, 'RestingBP'].mean()

df['RestingBP'] = df['RestingBP'].replace(0,round(resting_bp_mean))

plt.subplot(1,2,2)
sns.histplot(df['RestingBP'], kde=True)

plt.show()

sns.countplot(x=df['Sex'], hue=df['HeartDisease'])
plt.show()

sns.countplot(x=df['ChestPainType'], hue=df['HeartDisease'])
plt.show()

sns.countplot(x=df['FastingBS'], hue=df['HeartDisease'])
plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()

df_encode = pd.get_dummies(df, drop_first=True)
df_encode = df_encode.astype(int)

from sklearn.preprocessing import StandardScaler
scaler  = StandardScaler()

numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
df_encode[numeric_cols] = scaler.fit_transform(df_encode[numeric_cols])

print(df_encode.head(10))