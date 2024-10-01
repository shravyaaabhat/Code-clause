#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# Creating the dataset
data = {
    'customer_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    'channel': ['referral', 'paid advertising', 'email marketing', 'social media', 'referral', 'paid advertising', 
                'social media', 'email marketing', 'social media', 'social media', 'email marketing', 'referral', 
                'social media', 'email marketing', 'referral', 'paid advertising', 'referral', 'paid advertising', 
                'social media', 'referral', 'paid advertising'],
    'cost': [8.32, 30.45, 5.25, 9.55, 8.32, 30.45, 9.55, 5.25, 9.55, 9.55, 5.25, 8.32, 9.55, 5.25, 8.32, 30.45, 8.32, 
             30.45, 9.55, 8.32, 30.45],
    'conversion_rate': [0.12, 0.016, 0.044, 0.168, 0.12, 0.016, 0.168, 0.044, 0.168, 0.168, 0.044, 0.12, 0.168, 0.044, 
                        0.12, 0.016, 0.12, 0.016, 0.168, 0.12, 0.016],
    'revenue': [4199, 3410, 3164, 1520, 2419, 3856, 1172, 700, 2137, 982, 3003, 1455, 3388, 3562, 1147, 1303, 1456, 
                4549, 2054, 4439, np.nan]  # np.nan to simulate missing revenue for customer 21
}

df = pd.DataFrame(data)

# Display the first few rows of the dataset
print(df.head())


# In[2]:


# Drop rows with missing revenue
df.dropna(subset=['revenue'], inplace=True)


# In[3]:


# One-hot encoding for the 'channel' column
df_encoded = pd.get_dummies(df, columns=['channel'], drop_first=True)

# Display the encoded DataFrame
print(df_encoded.head())


# In[4]:


from sklearn.model_selection import train_test_split

# Define features and target variable
X = df_encoded.drop(columns=['customer_id', 'revenue'])  # Features (excluding customer_id and revenue)
y = df_encoded['revenue']  # Target variable (revenue)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the first few rows of the training data
print(X_train.head())


# In[5]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)


# In[6]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


# In[7]:


import matplotlib.pyplot as plt

# Scatter plot for actual vs predicted revenue
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Actual vs Predicted Revenue")
plt.show()


# In[ ]:




