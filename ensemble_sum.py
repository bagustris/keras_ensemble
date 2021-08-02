# model averaging ensemble and a study of ensemble size on test accuracy
from sklearn.datasets import make_blobs
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.metrics import accuracy_score
from matplotlib import pyplot


def fit_model(trainX, trainy):
    '''fit model on dataset'''
    # define model
    model = Sequential()
    model.add(Dense(15, input_dim=2, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    # fit model
    model.fit(trainX, trainy, epochs=200, verbose=0)
    return model


def ensemble_predictions(members, testX):
    '''
    make an ensemble prediction for multi-class classification
    '''
    # make predictions
    yhats = [model.predict(testX) for model in members]
    yhats = np.array(yhats)
    # sum across ensemble members (soft voting)
    summed = np.sum(yhats, axis=0)
    # argmax across classes (hard voting)
    result = np.argmax(summed, axis=1)
    return result


def evaluate_n_members(members, n_members, testX, testy):
    '''evaluate a specific number of members in an ensemble'''
    # select a subset of members
    subset = members[:n_members]
    print(len(subset))
    # make prediction
    yhat = ensemble_predictions(subset, testX)
    # calculate accuracy
    return accuracy_score(testy, yhat)


# generate 2d classification dataset
X, y = make_blobs(n_samples=500, centers=3, n_features=2,
                  cluster_std=2, random_state=2)

# split into train and test
n_train = int(0.3 * X.shape[0])
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
trainy = to_categorical(trainy)

# fit all models
n_members = 20
members = [fit_model(trainX, trainy) for _ in range(n_members)]

# evaluate different numbers of ensembles
scores = list()
for i in range(1, n_members+1):
    score = evaluate_n_members(members, i, testX, testy)
    print('> %.3f' % score)
    scores.append(score)

# plot score vs number of ensemble members
x_axis = [i for i in range(1, n_members+1)]
pyplot.plot(x_axis, scores)
pyplot.show()
