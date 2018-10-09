import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn import preprocessing
import numpy.linalg as m

import warnings
warnings.filterwarnings("ignore")


def show_report(Y, Y_pred, name_data):
    # Prediction evaluate score
    conf_matrix = confusion_matrix(Y, Y_pred)
    acc_score = accuracy_score(Y, Y_pred)
    print('=================================\n')
    print(name_data, " %s Score: %.2f%%" % ('accuracy', acc_score*100))
    print(name_data, " : Confusion Matrix: \n", conf_matrix)
    print('=================================\n')
    return acc_score, Y_pred


def plot_guassian(mu, sigma, idx, nclass):
    x = []
    for i in range(0, nclass):
        x[i] = np.linspace(mu[i][idx] - 3*sigma[i][idx][idx],
                           mu[i][idx] + 3*sigma[i][idx][idx], 100)
        plt.plot(x[i], mlab.normpdf(x[i], mu[i][idx], sigma[i][idx][idx]))
        plt.show()


def GaussianPropability(x, mean, sd):
    d = len(x.T)
    detSD = m.det(sd)
    invSD = m.inv(sd)
    prob = (((2*3.14)**(-d/2))*detSD**-0.5) * \
        np.exp(-0.5*np.dot(np.dot((x-mean).T, invSD), (x-mean)))
    return prob


def predict(X, mu, sd, nclass):
    y_pred = np.zeros(len(X))

    for i in range(0, len(X)):
        x = X[i, :]
        idx_mx = 0
        for c in range(1, nclass):
            if GaussianPropability(x, mu[c], sd[c]) > GaussianPropability(x, mu[idx_mx], sd[idx_mx]):
                idx_mx = c
        y_pred[i] = idx_mx
    return y_pred


# Preparing DataSet
data = datasets.load_iris()
print(list(data.target_names))
X, X_test, Y, Y_test = train_test_split(data.data, data.target, test_size=0.3)
nclass = 3

scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
X_test = scaler.transform(X_test)

mu = []
sd = []
X_train = []
Y_train = []
for i in range(0, nclass):
    X_train.append(X[Y == i])
    Y_train.append(Y[Y == i])
    a = np.mean(X_train[i], axis=0)
    mu.append(np.mean(X_train[i], axis=0))
    sd.append(np.cov(X_train[i].T))

Y_train_pred = predict(X, mu, sd, nclass)
Y_test_pred = predict(X_test, mu, sd, nclass)

show_report(Y_train_pred, Y, 'training')
show_report(Y_test_pred, Y_test, 'test')


plot_guassian(mu, sd, 0, nclass)
# plot_guassian(mu,sd,1,nclass)
# plot_guassian(mu,sd,2,nclass)
