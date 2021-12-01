import numpy as np
import torch

class EarlyStopping:
    """

    Parameters:
    -----------

    Returns:
    --------
    """
    def __init__(self, patience=7, mode="max", delta=0.001):
        """
        init parameters for the early stopping object

        Parameters:
        -----------
        patience: int
            number of epochs without validation score improve before early stop
        mode: str - "min" or "max"
            min if we are trying to minimize validation loss - max otherwise
        delta: float
            minimum improve to consider in the validation loss
        Returns:
        --------
        EarlyStopping initial object
        """
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):
        """
        early stopping check to call at each epoch

        Parameters:
        -----------
        epoch_score: float or int
            validation loss or metric to check for early stopping
        model: object
            the model object that is trained
        model_path: str
            path indicating where to save trained models
        Returns:
        --------
        Save model to model_path if performance improved
        increase early stop counter by one otherwise
        """
        
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(
                self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print(
                'Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score
