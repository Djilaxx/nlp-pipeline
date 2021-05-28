##################
# IMPORT MODULES #
##################
import numpy as np
import torch
from tqdm import tqdm
from utils.average_meter import AverageMeter
import warnings
warnings.filterwarnings("ignore")
#################
# TRAINER CLASS #
#################


class TRAINER:
    '''
    training_step train the model for one epoch
    eval_step evaluate the current model on validation data and output current loss and other evaluation metric
    '''
    def __init__(self, model, task, device, optimizer=None, criterion=None):
        self.model = model
        self.task = task
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion

    #################
    # TRAINING STEP #
    #################
    def training_step(self, data_loader):
        # LOSS AVERAGE
        losses = AverageMeter()
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
            # UPDATE LOSS
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=losses.avg)

    ###################
    # VALIDATION STEP #
    ###################
    def eval_step(self, data_loader, metric):
        # LOSS & METRIC AVERAGE
        losses = AverageMeter()
        metrics_avg = AverageMeter()
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

                # CHECK FOR REGRESSION VS CLASSIFICATION
                if self.task == "CLASSIFICATION":
                    output = output.argmax(axis=1)
                output = output.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                metric_value = metric(labels, output)

                losses.update(loss.item(), ids.size(0))
                metrics_avg.update(metric_value.item(), ids.size(0))

                tk0.set_postfix(loss=losses.avg)
        print(f"Validation Loss = {losses.avg}")
        return loss, metrics_avg.avg

    #############
    # TEST STEP #
    #############
    def test_step(self, data_loader, n_class):
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
                
                # OUTPUT PROCESSING
                if self.task == "CLASSIFICATION":
                    if n_class == 2:
                        preds = torch.sigmoid(preds)
                    elif n_class > 2:
                        preds = torch.softmax(preds, dim=0)
                preds = preds.cpu().detach().numpy()
                model_preds.extend(preds)
            tk0.set_postfix(stage="test")
        return model_preds
