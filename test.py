import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import warnings

warnings.filterwarnings("ignore")


def GaussianPropability(X, mean, sd):
    d = len(X)
    prob = ((2*3.14)**(-d/2))
    return prob


def gaussianSOF(x, mu, cov):
    d = len(x)
    p_a = (2.*np.pi)**(-d/2)
    print(p_a)
    p_b = (np.linalg.det(cov))**(-1/2)
    print(p_b)
    p_c = np.exp(
        -1./2*(np.dot(np.dot(np.transpose(x-mu), np.linalg.inv(cov)), x-mu))
    )
    print(p_c)
    return p_a * p_b * p_c


data = datasets.load_breast_cancer()
X, X_test, Y, Y_test = train_test_split(data.data, data.target, test_size=0.3)

scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
X_test = scaler.transform(X_test)

X_c0 = X[Y == 0]
Y_c0 = Y[Y == 0]

X_c1 = X[Y == 1]
Y_c1 = Y[Y == 1]

mean_c0 = np.mean(X_c0, axis=0)
mean_c1 = np.mean(X_c1, axis=0)

cov_c0 = np.cov(np.transpose(X_c0))
cov_c1 = np.cov(np.transpose(X_c1))

print(gaussianSOF(X_c0[0], mean_c0, cov_c0))

# Y_test_pred = predict(X_test)
# f1 = metrics.f1_score(Y_test,Y_test_pred)
# RatioAcc.append([ratio,f1])
