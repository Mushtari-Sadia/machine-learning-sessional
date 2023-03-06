import pandas as pd
import numpy as np
def load_dataset():
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class
    :return:
    """
    # todo: implement
    # df = pd.read_csv('data_banknote_authentication.csv')
    # df = df.to_numpy()
    # samples, features = df.shape
    # X = df[:,0:features-1]
    # y = df[:,features-1]


    #online
    df = pd.read_csv('parkinsons.data')
    cols = list(df.columns.values)  # Make a list of all of the columns in the df
    cols.pop(cols.index('status'))  # Remove x from list
    df = df[cols + ['status']]
    df = df.drop(['name'], axis=1)
    df = df.to_numpy()
    samples, features = df.shape
    X = df[:,0:features-1]
    y = df[:,features-1]

    return X, y


def split_dataset(X, y, test_size=0.2, shuffle=False):
    """
    function for spliting dataset into train and test
    :param X:
    :param y:
    :param float test_size: the proportion of the dataset to include in the test split
    :param bool shuffle: whether to shuffle the data before splitting
    :return:
    """
    # todo: implement.
    test_size = int(test_size * X.shape[0])
    samples, features = X.shape
    # print("features",features)
    # print(X.shape)
    # print(y.shape)
    if shuffle:
        c = np.hstack((X, [[el] for el in y]))
        np.random.shuffle(c)
        # print("after shuffle")
        X = c[:, 0:features]
        y = c[:,features]
        # print(X.shape)
        # print(y.shape)
    X_train, y_train, X_test, y_test = X[0:samples-test_size,:], y[0:samples-test_size], X[samples-test_size:,:], y[samples-test_size:]

    #online
    min, max = min_max(X_train)
    X_train = normalization(X_train, min, max)
    X_test = normalization(X_test, min, max)

    return X_train, y_train, X_test, y_test


def bagging_sampler(X, y):
    """
    Randomly sample with replacement
    Size of sample will be same as input data
    :param X:
    :param y:
    :return:
    """
    # todo: implement
    samples, features = X.shape
    c = np.hstack((X, [[el] for el in y]))
    np.random.seed(1)
    c = c[np.random.choice(c.shape[0],c.shape[0], True),:]
    # print("i,c",i,c)
    X_sample = c[:, 0:features]
    y_sample = c[:, features]
    assert X_sample.shape == X.shape
    assert y_sample.shape == y.shape
    return X_sample, y_sample

def min_max(X):
    min = np.min(X, axis=0)
    max = np.max(X, axis=0)
    return min, max

def normalization(X, min, max):
    X_normal = (X - min) / (max - min)
    return X_normal

# X,y = load_dataset()
# X_train, y_train, X_test, y_test = split_dataset(X, y, 0.2, True)
#
# print("X_train")
# print(X_train.shape)
# # print(X_train)
#
# print("X_test")
# print(X_test.shape)
# # print(X_test)
#
# print("y_train")
# print(y_train.shape)
# # print(y_train)
#
# print("y_test")
# print(y_test.shape)
# # print(y_test)