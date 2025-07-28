Hi, I'm JD. Thanks for visiting my page.

<img src="/assets/MyPhoto.jpg" alt="Me" style="max-width: 150px; border-radius: 8px;">

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

In summary, the model had a R squared of X and p-value of y meaning <mark>there is a pattern between independent and dependent variables and the results are not due to randomness.<mark>

> Making a point here
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
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  import os
  import seaborn as sns
  from sklearn.model_selection import train_test_split
  
  file_path = 'car_data.xlsx'
  df = pd.read_excel(file_path, sheet_name='data')
```
<br>
