from turtle import pd

from sklearn.preprocessing import LabelEncoder,OneHotEncoder




def Feature_Encoder(X,cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].IMDb))
        X[c] = lbl.fit_transform(list(X[c].values))
        OneHotEncoder_X=OneHotEncoder(categories=[c])
        X=OneHotEncoder.fit_transform(X).toarray()


    return X




