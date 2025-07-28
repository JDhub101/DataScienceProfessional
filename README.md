Welcome to my data science portfolio.

## 🤵 About Me

I currently work for a UK Financial Services company as a Data Analyst. My role mainly involves:
- Project managing data transformation projects.
- Analysing data to provide assurance that processes are followed and reporting insights to management.
- Creating interactive dashboards to understand trends and give people easy access to information.

I'm undertaking a Level 6 Data Science Apprenticeship to further enhance my skills.
<br>
<br>
<br>

## 🥇 My Skills

- **Tools**: Python, PowerBI, PowerQuery, Excel
- **Python Libraries**: Pandas, NumPy, matplotlib, seaborn
- **Technical skills**: Building Data Pipelines, Statistical Analysis, Data Visualisaition, Linear & Logistic Regression
- **General skills**: Project management, UK regulatory environment and rules.
- **Expierence**: 10 years expierence in the financial services across a variety of customer service and consultancy roles. And 3 years in a Data Analyst role.
<br>
<br>
<br>

## 💻 My Public Data Science Projects

I currently have one public project; a linear regression model to predict car price, in which I used python. See below.
<br>
<br>
<br>
<br>
<br>
<br>

## 🚙 Project 1: Linear Regression to Predict Car Price

<br>

#### Executive Summary

People want to pay a price for a car that reflects the quality and design. This analysis is to understand what factors influence the price of a car and if these can be relied upon to predict future car prices. This means that distributors can value cars appropriately and customers receive a fair price

The hypothesis was that the condition, brand and style of a car will influence the price, so these variables can be used to predict the price of a car. Linear Regression was chosen as the appropriate model because it can use numerical and categorical data, once encoded, to predict a numerical value.

The data was exported from Kaggle. The dataset includes details about the cars sold in 2014 and 2015 including brand, body type, year, odometer count, condition score. Kaggle 

In summary, The model’s results show that there is a reliable pattern between independent and dependent variables and ***the results are not due to randomness; meaning this model could be used to predict car prices.***

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

#### Data Analysis

| Item | Result | What does this mean? |
| ----------- | ----------- | ----------- |
| R Squared | 0.803 | 80% of the variance can be explained by the model
| Adjusted R Squared | 0.803  | 80% of variables are contributing to the model
| RSME | 4304 | On average predicted values are off by £4,304
| Condition coefficient | 44 | Every one condition increase, increases the price by £44
| Odometer coefficient | -0.05 | Every one mile increase, decreases the price by 5p
| Year coefficient | 281 | Every one year increase, increases the price by £281
| Mean Selling Price coefficient | 0.83 | Every one pound increase, increases the price by 83p
| P values | 0.00 | Below 0.05 which means the results are not random

Together these results suggest that the model is statistically significant and that these predictor variables can be used to predict car price, despite an individual correlation. The RSME is an acceptable tolerance because the price ranges from £100 to £200,000. 

<br>
<br>
<br>

#### Copy of Python Code:

This is a copy of the python code I used in this analysis. I used a juypter notebook in Visual Studio Code Editor.
<br>
```
# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load the data
file_path = 'car_data.xlsx'
df = pd.read_excel(file_path, sheet_name='data')

# Choose columns
df = df[['year', 'make', 'model', 'trim', 'condition', 'odometer', 'sellingprice']]

# Check for blanks
df.isnull().sum()

# Convert columns to numerical values
df['sellingprice'] = pd.to_numeric(df['sellingprice'], errors='coerce')
df['odometer'] = pd.to_numeric(df['odometer'], errors='coerce')
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Remove rows with missing selling price
df = df[df['sellingprice'].notna()]
df = df[df['make'].notna()]

# See the outliers in selling price
plt.figure(figsize=(6, 8))
plt.boxplot(df['odometer'])
plt.title('Odometer')
plt.ylabel('Odometer')
plt.grid(True)
plt.tight_layout()
plt.show()

# See the distribution of data
sns.histplot(data=df, x='sellingprice').set(title='sellingprice distribution')

# See linear relationship
sns.scatterplot(x ="year", y ="sellingprice", data = df)

# Creating means & medians
odometer_mean = df['odometer'].mean()
condition_mean = df['condition'].mean()
year_mean = df['year'].mean()

# Replace odometer outliers (values > 180000) with the mean
df.loc[df['odometer'] > 180000, 'odometer'] = odometer_mean

# Replace blanks with mean
df['odometer'].fillna(odometer_mean, inplace=True)
df['condition'].fillna(condition_mean, inplace=True)
df['year'].fillna(year_mean, inplace=True)

# Create a new column for the average price of a car using the make, model and trim
df['make_model_trim'] = df[['make', 'model', 'trim']].astype(str).agg(' '.join, axis=1)
mean_prices = df.groupby('make_model_trim')['sellingprice'].mean()
df['mean_sellingprice'] = df['make_model_trim'].map(mean_prices)

# Calculate the average price of a BWM 1 series
filtered = df[df['make_model_trim'].str.contains("bmw 1 series", case=False, na=False)]
mean_price_bmw_series1 = filtered['sellingprice'].mean()
print("The average price of a BMW 1 Series is: £",round(mean_price_bmw_series1))

# Create a copy of the dataframe with the columns needed
df1 = df[['condition', 'odometer', 'year', 'mean_sellingprice', 'sellingprice']]

# Look at correlation 
df1.corr()

# Check there are no blanks
df1.isnull().sum()

# Split the data into train and test
train_df, test_df = train_test_split(df1, test_size=0.2, random_state=1234)

# See the split
print(df1.shape)
print(train_df.shape)
print(test_df.shape)

# Drop columns that aren't needed in each dataset
X_train = train_df.drop('sellingprice', axis=1)
X_test = test_df.drop('sellingprice', axis=1)

y_train = train_df['sellingprice']
y_test = test_df['sellingprice']

# Create model and see results
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("The R Squared is:", model.score(X_test, y_test))
print("The Intercept is:", model.intercept_)
rmse = float(format(np.sqrt(met.mean_squared_error(y_test,y_pred)),'.3f'))
print("The Root Squared Mean Error (RSME) is:",rmse)

# See coefficients
pd.DataFrame(zip(X_train.columns, model.coef_))

# See results including P Value
import statsmodels.formula.api as smf
f = "sellingprice ~ year + condition + odometer + mean_sellingprice"
model_ols = smf.ols(formula=f, data= train_df)
model_fit = model_ols.fit()
print(model_fit.summary())

# Predict the price of a 2015 BMW 1 series using the model
condition = 30
odometer = 30000
mean_sellingprice = 19342
year = 2015
predicted_Value = -560962 + (44 * condition) + (-0.05 * odometer) + (0.83 * mean_sellingprice) + (281 * year)

# See predicted value
print(f"The predicted value of a 2015 BMW 1 Series with 30000 miles and a condition of 30 is: £{round(predicted_Value)}")

```
<br>
