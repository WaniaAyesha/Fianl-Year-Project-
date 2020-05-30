# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('OLX_Car_Data.csv', encoding= 'latin')
dataset = dataset.dropna() # Dropping all data points with one or more vlues missing
x_df = dataset.drop('Price', axis=1) # Dropping coloumn 5 i.e. price becasue it is output in our model
y_df = dataset['Price'] #Selecting prices for output


"""
# Converting input back to numpy array for further processing
X = x_df.to_numpy() # Converting input back to numpy array for further processing
y = y_df.to_numpy()
"""

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
#obj_df = x_df.select_dtypes(include=['object']).copy()
#obj_df["Brand"].value_counts()
x_df = pd.get_dummies(x_df, columns=["Brand"])
x_df = pd.get_dummies(x_df, columns=["Condition"])
x_df = pd.get_dummies(x_df, columns=["Fuel"])
x_df = pd.get_dummies(x_df, columns=["Model"])
x_df = pd.get_dummies(x_df, columns=["Registered City"])
x_df = pd.get_dummies(x_df, columns=["Transaction Type"])
x_df = pd.get_dummies(x_df, columns=["Year"])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size = .1, random_state = 0)

#Training a simple linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#testing routine
from sklearn.metrics import mean_squared_error
y_pred = regressor.predict(X_train)
mean_squared_error(y_train, y_pred)