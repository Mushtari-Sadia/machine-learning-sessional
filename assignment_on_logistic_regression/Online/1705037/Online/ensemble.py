from data_handler import bagging_sampler
import copy
import numpy as np

class BaggingClassifier:
    def __init__(self, base_estimator, n_estimator):
        """
        :param base_estimator:
        :param n_estimator:
        :return:
        """
        # todo: implement
        self.base_estimator = base_estimator
        self.n_estimator = n_estimator
        self.estimators = []

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement
        for i in range(self.n_estimator):
            model = copy.copy(self.base_estimator)
            X, y = bagging_sampler(X, y)
            model.fit(X,y)
            self.estimators.append(model)



    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        apply majority voting
        :param X:
        :return:
        """
        # todo: implement
        y_preds = []
        for model in self.estimators:
            y_preds.append(model.predict(X))
        y_preds_arr = np.array(y_preds)
        majority_matrix = y_preds_arr.T
        y_pred = np.zeros(X.shape[0])
        i=0
        for row in majority_matrix:
            ones = np.count_nonzero(row==1.0)
            zeros = len(row)-ones
            if ones >= zeros :
                y_pred[i]=1.0
            i+=1
        return y_pred
