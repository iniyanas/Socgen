from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn import linear_model
import math
from google.colab import drive
import matplotlib.pyplot as plt
drive.mount('/content/drive')
from sklearn.metrics import mean_absolute_error, mean_squared_error

FilePath="/content/drive/MyDrive/socgen.xlsx"
df=pd.read_excel(FilePath)
print(df)

dataset=df.values
dataset
print(dataset.shape)

x=dataset[2:2068,7:10]
y=dataset[2:2068,0:7]
print(x)
print(y)
print(x.shape)
print(y.shape)

reg=linear_model.LinearRegression()

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20)

reg.fit(xtrain,ytrain)

print(reg.coef_)

ypred=reg.predict(xtest)
print(ypred)

input=[[0,1,2]]    # [task_type, task_priority, task_status]
a=reg.predict(input)
print("cpu_usage:",a[0][0],"%")
print("memory_usage",a[0][1],"%")
print("network_traffic",a[0][2],"kb/sec")
print("power _consumption",a[0][3],"watt")
print("no of instruction executed",a[0][4])
print("execution time",a[0][5],"sec")
print("energy efficient",a[0][6]*100,"%")

mae = mean_absolute_error(ytest, ypred)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(ytest, ypred)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
