from src.neural_network.metrics.Metrics import Metrics
from collections import defaultdict


class Summary:

    def __init__(self):
        self.y_real = None
        self.y_predicted = None
        self.summary = defaultdict(list)

    def fit(self, y_real, y_predicted):
        self.y_real = y_real
        self.y_predicted = y_predicted

    def add_step(self):
        """
        Adds a step into calculated metrics
        :return:
        """
        assert self.y_real is not None and self.y_predicted is not None

        # Calculates some metrics
        rmse = Metrics.rmse_loss(self.y_real, self.y_predicted)
        mse = Metrics.mse_loss(self.y_real, self.y_predicted)
        cm = Metrics.confusion_matrix(self.y_real, self.y_predicted)
        accuracy = Metrics.accuracy(cm)

        # Store them
        self.summary['rmse'].append(rmse)
        self.summary['accuracy'].append(accuracy)
        self.summary['mse'].append(mse)
        self.summary['cm'].append(cm)

    def fit_and_add_step(self, y_real, y_predicted):

        self.fit(y_real, y_predicted)
        self.add_step()

    def get_rmse(self):

        return self.summary['rmse'][-1]

    def get_mse(self):

        return self.summary['mse'][-1]

    def get_confusion_matrix(self):

        return self.summary['cm'][-1]

    def get_accuracy(self):

        return self.summary['accuracy'][-1]

    def __repr__(self):

        first_line = "Best accuracy: {:.2f}\n".format(self.summary['accuracy'][-1])
        second_line = "Best RMSE: {:.2f}\n".format(self.summary['rmse'][-1])
        third_line = "Best MSE: {:.2f}\n".format(self.summary['mse'][-1])
        return first_line + second_line + third_line

