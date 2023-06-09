import numpy as np
import pandas as xx
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from Pre_processing import *


def featureScaling(X, a, b):
    Normalized_X = np.zeros((X.shape[0], X.shape[1]));
    for i in range(X.shape[1]):
        Normalized_X[:, i] = ((X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))) * (b - a) + a;
    return Normalized_X




#Load players data



data = xx.read_csv('Movies_training.csv')

#Drop the rows that contain missing values

data.dropna(axis=0, how='any', inplace=True)

data['Rotten Tomatoes'] = list(map(int, data['Rotten Tomatoes'].str.replace('%', '')))
data = data.sort_values(by=['Title'])
movie_data = data.iloc[:, :]
X = data.iloc[:, [1, 3, 9, 13]]
hal = data.iloc[:, [0, 9]]
Y = data['IMDb']  # Label
cleaned = hal.set_index('Title').Directors.str.split(',', expand=True).stack()
w = xx.get_dummies(cleaned, prefix='D').groupby(level=0).sum()

dire= []
w = np.array(w)
for i in range(w.shape[0]):
    c = np.sum(w[i], axis=0)
    dire.append(c)
X['Directors'] = dire

X = featureScaling(np.array(X), 0, 1)

#cols=('Age', 'Rotten Tomatoes')
#X=Feature_Encoder(X,cols);

#to Split the data to training and testing sets we use

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30)

#to Get the correlation between the features

corr = movie_data.corr()
#Top 50% Correlation training features

top_feature = corr.index[abs(corr['IMDb']>0.2)]

#Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = movie_data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()

poly_features = PolynomialFeatures(degree=3)

# transform
X_train_poly = poly_features.fit_transform(X_train)

# fit the transform

poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)

# predicting on training dataset
y_train_predicted = poly_model.predict(X_train_poly)

# predicting on test dataset
prediction = poly_model.predict(poly_features.fit_transform(X_test))


print('Co-efficient of linear regression', poly_model.coef_)
print('Intercept of linear regression model', poly_model.intercept_)
print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))

trueIMDb = np.asarray(y_test)[0]
predictedIMDb = prediction[0]

print('real : ' + str(trueIMDb))
print('Predicted : ' + str(predictedIMDb))


