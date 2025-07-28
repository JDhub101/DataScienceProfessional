Hi, I'm JD. Thanks for visiting my page.

## 🤵 About Me

I have a background in:
- creating and embedding data transformation projects.
- analysing data to provide assurance that processes are compliant.
- creating interactive dashboards to track operations.

I'm currently undertaking a Data Science Apprenticeship to further enhance my technical skills.
<br>
<br>
<br>

## 🥇 My Skills

- **Tools**: Python, Power BI, Excel
- **Python Libraries**: Pandas, NumPy, matplotlib
- **Techniques**: Building Data Pipelines, Statistical Analysis, Data Visualisaition, Linear & Logistic Regression
- **Qualifications**: Data Analytics Apprenticeship, Compliance & Risk Apprenticeship
<br>
<br>
<br>

## 💻 My Public Projects

Currently, I have only one public project, which is a linear regression using python to predict car price (see below).
<br>
<br>
<br>

## Prediciting Car Price (using a kaggle public dataset)

<br>

#### Executive Summary

People want to pay a price for a car that reflects the quality and design. This analysis is to understand what factors influence the price of a car and if these can be relied upon to predict future car prices. This means that distributors can value cars appropriately and customers receive a fair price

The hypothesis was that the condition, brand and style of a car will influence the price, so these variables can be used to predict the price of a car. Linear Regression was chosen as the appropriate model because it can use numerical and categorical data, once encoded, to predict a numerical value.

The data was exported from Kaggle. It includes details about the cars sold in 2014 and 2015 including brand, body type, year, odometer count, condition score. Kaggle 

In summary, the model had a R squared of X and p-value of y meaning **there is a pattern between independent and dependent variables and the results are not due to randomness.**

> Linear Regression predicts a numerical continious value. Categorical columns can be encoded.

<br>
<br>
<br>

#### Data Visualisation

Visualising data is crucial to fully understand it.

I plotted a column & line chart to X

![chart 1!](/assets/Picture1.png "chart 1")
<br>
<br>
<br>

#### Copy of Python Code:

Here is the python code I used in this analysis. I used a juypter notebook in Visual Studio Code Editor.
<br>
```
# Load the packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as met

# Load the data
file_path = 'car_data.xlsx'
df = pd.read_excel(file_path, sheet_name='data'

# Convert columns to numeric values
df['sellingprice'] = pd.to_numeric(df['sellingprice'], errors='coerce')
df['odometer'] = pd.to_numeric(df['odometer'], errors='coerce')
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Remove rows with missing selling price
df = df[df['sellingprice'].notna()]
df = df[df['make'].notna()]

# Creating means
odometer_mean = df['odometer'].mean()
condition_mean = df['condition'].mean()
year_mean = df['year'].mean()

# Replace odometer outliers (values > 180000) with the mean
df.loc[df['odometer'] > 180000, 'odometer'] = odometer_mean

# Replace blanks with mean
df['odometer'].fillna(odometer_mean, inplace=True)
df['condition'].fillna(condition_mean, inplace=True)
df['year'].fillna(year_mean, inplace=True)

# create a mean encoded column 
df['make_model_trim'] = df[['make', 'model', 'trim']].astype(str).agg(' '.join, axis=1)

# calculate the mean price using other columns
mean_prices = df.groupby('make_model_trim')['sellingprice'].mean()
df['mean_sellingprice'] = df['make_model_trim'].map(mean_prices)

# See the correlation
df.corr()

# See the blank rows
df.isnull().sum()

# Split the data into test & train
train_df, test_df = train_test_split(df1, test_size=0.2, random_state=1234)

# See the split of the data
print(df1.shape)
print(train_df.shape)
print(test_df.shape)

# Drop columns in datasets that are not required
X_train = train_df.drop('sellingprice', axis=1)
X_test = test_df.drop('sellingprice', axis=1)
y_train = train_df['sellingprice']
y_test = test_df['sellingprice']

# Run the model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# See results
r2_score = met.r2_score(y_test, y_pred)
rmse = float(format(np.sqrt(met.mean_squared_error(y_test,y_pred)),'.3f'))
r2_score
rmse

# See intercept
print(model.intercept_)
pd.DataFrame(zip(X_train.columns, model.coef_))

# See p-value
import statsmodels.formula.api as smf
f = "sellingprice ~ year + condition + odometer + mean_sellingprice"
model_ols = smf.ols(formula=f, data= train_df)
model_fit = model_ols.fit()
model_fit.summary()

# Calculate the mean price for a BMW 1 Series
filtered = df[df['make_model_trim'].str.contains("bmw 1 series", case=False, na=False)]
mean_price_bmw_series1 = filtered['sellingprice'].mean()
mean_price_bmw_series1

# Calculate the predicted price (using the model) for a BWM 1 Series;
condition = 30
odometer = 30000
mean_sellingprice = 19342
year = 2015
predicted_Value = -560962 + (44 * condition) + (-0.05 * odometer) + (0.83 * mean_sellingprice) + (281 * year)

# See predicted value
print(f"The predicted value of a 2015 plate BMW 1 Series with a 30 condition and 30,000 miles is:  {predicted_Value}")

```
<br>
