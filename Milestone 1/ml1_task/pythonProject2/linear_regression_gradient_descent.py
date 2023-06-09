import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
#Loading data
data = pd.read_csv('Movies_training.csv')
#print(data.describe())
#X=data['Rotten Tomatoes']
#Y=data['IMDb']
#print(X.shape)



#Drop the rows that contain missing values
data.dropna(how='any',inplace=True)
movie_data= data.iloc[:, :]
#Features
X=data.iloc[:,1]
#Label
Y=data['IMDb']
L = 0.0000001  # The learning Rate
epochs = 100  # The number of iterations to perform gradient descent
m=0
c=0
n = float(len(X)) # Number of elements in X
for i in range(epochs):
    Y_pred = m*X + c  # The current predicted value of Y
    D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
    D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c

prediction = m*X + c


print('Mean Square Error', metrics.mean_squared_error(Y, prediction))


input_to_prediction=int(input('Enter your year : '))

out=m*input_to_prediction+c
print('Your predicted IMDb is ' + str(out))


#predictedIMDb = prediction[0]
#print('Predicted : ' + str(predictedIMDb))