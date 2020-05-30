# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('OLX_Car_Data.csv', encoding= 'latin')
X = dataset.iloc[:, :].values # Importing all coloumns and rows
df = pd.DataFrame(X) # Converting to DaatFrame for inspecting elements in spyder
df = df.drop(5, axis=1) # Dropping coloumn 5 i.e. price becasue it is output in our model
X = df.to_numpy() #converting input back to numparray for further processing
y = dataset.iloc[:, -4].values
