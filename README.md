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


!MMR image

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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics as met
from sklearn.linear_model import LinearRegression

```
<br>
