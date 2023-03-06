import numpy as np
class LogisticRegression:
    def __init__(self, params):
        """
        figure out necessary params to take as input
        :param params:
        """
        # todo: implement
        self.epoch = params['epoch']
        self.alpha = params['alpha']
        self.costs = []


    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement
        y = np.reshape(y,(y.shape[0], 1))
        w = np.zeros((X.shape[1],1))
        b = 0
        m = X.shape[0]
        # print("m",m)

        for e in range(self.epoch):
            H = self.g(np.dot(X,w) + b)
            # print("H.shape",H.shape)
            # print("y.shape", y.shape)
            cost = -np.sum(np.dot(y.T,np.log(H)) + np.dot((1 - y.T),np.log(1 - H)))*(1.0/m)
            self.costs.append(cost)
            dw = np.dot(X.T, (H - y))/m
            # print("dw.shape",dw.shape)
            db = np.sum(H - y)/m
            w = w-(self.alpha * dw)
            b = b-(self.alpha * db)
        self.w = w
        self.b = b
        # print("W",w)
        # print("b",b)



    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        :param X:
        :return:
        """
        # todo: implement
        y_pred = np.zeros((1,X.shape[0]))
        W = self.w.reshape(X.shape[1], 1)
        hypothesis = self.g(np.dot(X,W) + self.b)
        # print("hypothesis shape",hypothesis.shape)
        x = hypothesis.shape[0]
        for i in range(x):
            if hypothesis[i, 0] >= 0.5:
                y_pred[0, i] = 1
            else:
                y_pred[0, i] = 0
        # print(y_pred)
        return y_pred

    def g(self,z):
        sigm = 1.0 / (1.0 + np.exp(-z))
        return sigm