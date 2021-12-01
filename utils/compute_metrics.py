import numpy as np
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")

class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MetricsMeter:
    """
    a class to compute evaluation metrics at each time step.

    Parameters:
    -----------
    task: str
        classification or regression
    """
    def __init__(self, task):
        self.task = task
        self.metrics_dict = {}

        if self.task == "REGRESSION":
            regression_metrics = ["mae", "mse", "rmse"]
            for metric in regression_metrics:
                self.metrics_dict.update({metric: AverageMeter()})
        elif self.task == "CLASSIFICATION":
            classification_metrics = ["accuracy", "auc", "precision", "recall", "f1_score"]
            for metric in classification_metrics:
                self.metrics_dict.update({metric: AverageMeter()})

    def compute_metrics(self, predictions, labels, n_class):
        """
        Compute several metrics depending on task

        Parameters:
        -----------
        predictions: ndarray
            array of predictions from our model
        labels: ndarray
            array of true labels to compare our predictions
        n_class: int
            number of class in our dataset

        Returns:
        --------
        metrics_dict: dictionnary
            a dict containing metrics name (key) and corresponding AverageMeter (value)
        """
        if self.task == "REGRESSION":
            mae = metrics.mean_absolute_error(labels, predictions["preds"])
            self.metrics_dict["mae"].update(mae, len(labels))
            mse = metrics.mean_squared_error(labels, predictions["preds"])
            self.metrics_dict["mse"].update(mse, len(labels))
            rmse = np.sqrt(mse)
            self.metrics_dict["rmse"].update(rmse, len(labels))
            return self.metrics_dict
            
        elif self.task == "CLASSIFICATION":
            accuracy = metrics.accuracy_score(labels, predictions["preds"])
            self.metrics_dict["accuracy"].update(accuracy, len(labels))
            auc = None
            if n_class == 2:
                try:
                    auc = metrics.roc_auc_score(labels, predictions["preds_score"])
                    self.metrics_dict["auc"].update(auc, len(labels))
                except ValueError:
                    pass

                precision = metrics.precision_score(labels, predictions["preds"])
                self.metrics_dict["precision"].update(precision, len(labels))
                recall = metrics.recall_score(labels, predictions["preds"])
                self.metrics_dict["recall"].update(recall, len(labels))
                f1_score = metrics.f1_score(labels, predictions["preds"])
                self.metrics_dict["f1_score"].update(f1_score, len(labels))
            elif n_class>2:
                precision = metrics.precision_score(labels, predictions["preds"], average="macro")
                self.metrics_dict["precision"].update(precision, len(labels))
                recall = metrics.recall_score(labels, predictions["preds"], average="macro")
                self.metrics_dict["recall"].update(recall, len(labels))
                f1_score = metrics.f1_score(labels, predictions["preds"], average="macro")
                self.metrics_dict["f1_score"].update(f1_score, len(labels))
                
            return self.metrics_dict
        else:
            raise Exception("task not supported")
