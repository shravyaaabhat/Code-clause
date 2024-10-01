#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas scikit-learn matplotlib seaborn


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset
# Assuming your dataset is a CSV file. Replace 'house_prices.csv' with the correct file path.
df = pd.read_csv('Housing.csv')


# In[3]:


# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values or fill them
df = df.dropna()  # Alternatively, you can fill missing values with df.fillna()


# In[5]:


# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Scatter plot between square footage and price
sns.scatterplot(x='area', y='Price', data=df)
plt.show()

# Scatter plot between number of bedrooms and price
sns.scatterplot(x='Bedrooms', y='Price', data=df)
plt.show()


# In[9]:


# Define the features and target variable
X = df[['bedrooms', 'area']]  # Features
y = df['price']  # Target variable

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# In[10]:


# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)


# In[11]:


# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


# In[12]:


# Scatter plot for actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()


# In[ ]:




