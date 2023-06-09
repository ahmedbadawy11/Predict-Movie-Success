# importing necessary libraries
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import pickle
#iris = datasets.load_iris()

def featureScaling(X, a, b):
    Normalized_X = np.zeros((X.shape[0], X.shape[1]));
    for i in range(X.shape[1]):
        Normalized_X[:, i] = ((X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))) * (b - a) + a;
    return Normalized_X

def Feature_Encoder(X):

    lbl = LabelEncoder()
    lbl.fit(list(X.values))
    X = lbl.transform(list(X.values))
 #   OneHotEncoder_X=OneHotEncoder(categories=[c])
   #  X=OneHotEncoder.fit_transform(X).toarray()


    return X




df = pd.read_csv('Movies_testing_classification.csv')

df = df.sort_values(by=['Title'])


df.dropna(axis=0, subset=['rate'], inplace=True)


df=df.fillna(0)
rep= []
for i in df['Rotten Tomatoes']:
    if(i!=0):
        rep.append(int(i.replace('%', '')))

    else:
        rep.append(i)

df['Rotten Tomatoes'] = rep
rep.clear()


dec=[]
for i in df['Directors']:
    if(i==0):
        dec.append(str(i))
    else:
        dec.append(i)

df['Directors'] = dec
dec.clear()

dire= []
hal = df.iloc[:, [0, 9]]
cleaned = hal.set_index('Title').Directors.str.split(',', expand=True).stack()
w = pd.get_dummies(cleaned, prefix='D').groupby(level=0).sum()
w = np.array(w)
for i in range(w.shape[0]):
    c = np.sum(w[i], axis=0)
    dire.append(c)
df['Directors'] = dire
dire.clear()


X = df.iloc[:, [1,3,9,  13]]  # we only take the first two features.
y = df['rate']









y=Feature_Encoder(y)
# Plot the samples
plt.scatter(X['Rotten Tomatoes'], X['Runtime'], c=y, cmap=plt.cm.coolwarm)
plt.xlabel('Rotten Tomatoes')
plt.ylabel('Runtime')
plt.show()

# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0,shuffle=True)



# training a linear SVM classifier
svm_model_linear_ovr = OneVsRestClassifier(SVC(kernel='linear', C=1)).fit(X_train, y_train)



with open ("model_pickle","wb") as f:
    pickle.dump(svm_model_linear_ovr,f)

with open ("model_pickle","rb") as f:
    ms1=pickle.load(f)
svm_predictions1 = ms1.predict(X_test)



# model accuracy for X_test
accuracy = svm_model_linear_ovr.score(X_test, y_test)
print('One VS Rest SVM accuracy_Linear: ' + str(accuracy))
#
svm_model_linear_ovo = SVC(kernel='linear', C=3).fit(X_train, y_train)



with open ("model_pickle","wb") as f:
    pickle.dump(svm_model_linear_ovo,f)

with open ("model_pickle","rb") as f:
    ms2=pickle.load(f)

svm_predictions2 = ms2.predict(X_test)

# model accuracy for X_test
accuracy = svm_model_linear_ovo.score(X_test, y_test)
print('One VS One SVM accuracy _Linear: ' + str(accuracy))




svm_model_rbf_ovo = SVC(kernel='rbf', gamma=0.8, C=5).fit(X_train, y_train)


with open ("model_pickle","wb") as f:
    pickle.dump(svm_model_rbf_ovo,f)

with open ("model_pickle","rb") as f:
    ms3=pickle.load(f)

svm_predictions3 = ms3.predict(X_test)

# model accuracy for X_test
accuracy = svm_model_rbf_ovo.score(X_test, y_test)
print('One VS One SVM accuracy Gaussian: ' + str(accuracy))


h = .02  # step size in the mesh
#feature1
x_min, x_max = X['Rotten Tomatoes'].min() - 1, X['Rotten Tomatoes'].max() + 1
#feature2
y_min, y_max = X['Runtime'].min() - 1, X['Runtime'].max() + 1

#generate f1,f2 from min f1,f2 to max f1,f2 with step size h
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

#predictions = svm_model_linear_ovr.predict(np.c_[xx.ravel(), yy.ravel()])







# Put the result into a color plot
#change predictions to a meshgrid of the size of the inputs
svm_predictions1 = svm_predictions1.reshape(xx.shape)
plt.contourf(xx, yy, svm_predictions1, cmap=plt.cm.coolwarm, alpha=0.45)

# Plot also the training points
plt.scatter(X['Rotten Tomatoes'], X['Runtime'], c=y, cmap=plt.cm.coolwarm)
plt.xlabel('Rotten Tomatoes')
plt.ylabel('Runtime')
plt.show()