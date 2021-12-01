##################
# IMPORT MODULES #
##################
import numpy as np
import torch
from tqdm import tqdm
from utils.process_outputs import process_outputs
from utils.compute_metrics import MetricsMeter, AverageMeter

import warnings
warnings.filterwarnings("ignore")
#################
# TRAINER CLASS #
#################


class TRAINER:
    """
    Trainer class to fit a model, validation and get predictions on test data.

    Parameters:
    -----------
    model: nn.Module object
        the model that we want to use for training or predictions
    task: str
        classification or regression
    device: str
        cuda for GPU training or cpu if GPU not available.
    optimizer: nn.optim object
        the optimizer used for gradient descent training
    criterion: loss function
        the loss function used for training
    n_class: int
        the number of class in your data (1 for regression)
    """
    def __init__(self, model, task, device, optimizer=None, criterion=None, n_class=2):
        self.model = model
        self.task = task
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.n_class = n_class

    #################
    # TRAINING STEP #
    #################
    def training_step(self, data_loader):
        """
        train the given model for one epoch using the training dataloader

        Parameters
        ----------
        data_loader: torch dataloader object
            the training dataloader on which we wish to train the model
           
        Returns
        -------
        self: object
            trained model
        """
        # LOSS AVERAGE
        losses = AverageMeter()
        metrics_meter = MetricsMeter(task=self.task)
        # MODEL TO TRAIN MODE
        self.model.train()
        # TRAINING LOOP
        tk0 = tqdm(data_loader, total=len(data_loader))
        for _, data in enumerate(tk0):
            model_name = self.model.__class__.__name__
            # LOADING TEXT TOKENS & LABELS
            ids = data["ids"].to(self.device)
            masks = data["masks"].to(self.device)
            labels = data["labels"].to(self.device)
            # REGRESSION loss functions expect labels to be floats
            if self.task == "REGRESSION":
                labels = labels.to(torch.float32)
            # BERT REQUIRES TOKEN_TYPE_IDS TOO
            if model_name in ["BERT"]:
                token_type_ids = data["token_type_ids"].to(self.device)
                # GETTING PREDICTION FROM MODEL
                self.model.zero_grad()
                output = self.model(ids=ids, mask=masks,
                                    token_type_ids=token_type_ids)

            elif model_name in ["DISTILBERT", "ROBERTA"]:
                # GETTING PREDICTION FROM MODEL
                self.model.zero_grad()
                output = self.model(ids=ids, mask=masks)

            # CALCULATE LOSS
            loss = self.criterion(output, labels)
            # CALCULATE GRADIENTS
            loss.backward()
            self.optimizer.step()

            #COMPUTE METRICS
            train_preds, labels = process_outputs(self.task, output, labels)
            train_metrics = metrics_meter.compute_metrics(train_preds, labels, n_class=self.n_class)

            # UPDATE LOSS
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=losses.avg)
        return losses.avg, train_metrics

    ###################
    # VALIDATION STEP #
    ###################
    def validation_step(self, data_loader):
        """
        Validate the trained model on the validation loader and compute evaluation metric

        Parameters
        ----------
        data_loader: torch dataloader object
            the validation dataloader we use to evaluate current model performance
        metric: metric from utils.metric.metric_dict (sklearn function)
            the chosen metric we use to evaluate model performance
        Returns
        -------
        loss: float
            model current validation loss
        metrics_avg.avg: float
            model current performance using metric chosen
        """
        # LOSS & METRIC AVERAGE
        losses = AverageMeter()
        metrics_meter = MetricsMeter(task=self.task)
        # MODEL TO EVAL MODE
        self.model.eval()
        # VALIDATION LOOP
        with torch.no_grad():
            tk0 = tqdm(data_loader, total=len(data_loader))
            for _, data in enumerate(tk0):
                model_name = self.model.__class__.__name__
                # LOADING TEXT TOKENS & LABELS
                ids = data["ids"].to(self.device)
                masks = data["masks"].to(self.device)
                labels = data["labels"].to(self.device)
                if self.task == "REGRESSION":
                    labels = labels.to(torch.float32)
                if model_name in ["BERT"]:
                    token_type_ids = data["token_type_ids"].to(self.device)
                    # GETTING PREDICTION FROM MODEL
                    output = self.model(ids=ids, mask=masks,
                                        token_type_ids=token_type_ids)
                elif model_name in ["DISTILBERT", "ROBERTA"]:
                    # GETTING PREDICTION FROM MODEL
                    output = self.model(ids=ids, mask=masks)

                # CALCULATE LOSS & METRICS
                loss = self.criterion(output, labels)

                #COMPUTE METRICS
                valid_preds, labels = process_outputs(self.task, output, labels)
                valid_metrics = metrics_meter.compute_metrics(valid_preds, labels, n_class=self.n_class)

                losses.update(loss.item(), ids.size(0))
                tk0.set_postfix(loss=losses.avg)
        print(f"Validation Loss = {losses.avg}")
        return losses.avg, valid_metrics

    #############
    # TEST STEP #
    #############
    def test_step(self, data_loader):
        """
        test a trained model on a testloader and output predictions

        Parameters:
        -----------
        data_loader: torch dataloader object
            test dataloader
        n_class:
            number of different class in your dataset
        Returns:
        --------
        model_preds: list
            list of model predictions on the test dataset.
        """
        # DATA LOADER LOOP
        model_preds = []
        with torch.no_grad():
            tk0 = tqdm(data_loader, total=len(data_loader))
            for _, data in enumerate(tk0):
                model_name = self.model.__class__.__name__
                # LOADING TEXT TOKENS & LABELS
                ids = data["ids"].to(self.device)
                masks = data["masks"].to(self.device)
                if model_name in ["BERT"]:
                    token_type_ids = data["token_type_ids"].to(self.device)
                    # GETTING PREDICTION FROM MODEL
                    preds = self.model(ids=ids, mask=masks, token_type_ids=token_type_ids)
                elif model_name in ["DISTILBERT", "ROBERTA"]:
                    # GETTING PREDICTION FROM MODEL
                    preds = self.model(ids=ids, mask=masks)
                
                test_preds, _ = process_outputs(self.task, preds, labels=None)
                model_preds.extend(test_preds["preds"])
            tk0.set_postfix(stage="test")
        return model_preds
