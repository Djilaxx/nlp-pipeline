##################
# IMPORT MODULES #
##################
import numpy as np
import torch
from tqdm import tqdm
from utils.average_meter import AverageMeter
from utils.metrics import metrics_dict
import warnings
warnings.filterwarnings("ignore")
#################
# TRAINER CLASS #
#################
class TRAINER:
    '''
    trn_function train the model for one epoch
    eval_function evaluate the current model on validation data and output current loss and other evaluation metric
    '''
    def __init__(self, model, optimizer, device, criterion, task):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = criterion
        self.task = task
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
            if model_name in ["DISTILBERT", "ROBERTA"]:
                ids = data["ids"].to(self.device)
                masks = data["masks"].to(self.device)
                labels = data["labels"].to(self.device)
                # GETTING PREDICTION FROM MODEL
                self.model.zero_grad()
                output = self.model(ids=ids, mask=masks)

            elif model_name in ["BERT"]:
                ids = data["ids"].to(self.device)
                masks = data["masks"].to(self.device)
                token_type_ids = data["token_type_ids"].to(self.device)
                # GETTING PREDICTION FROM MODEL
                self.model.zero_grad()
                output = self.model(ids=ids, mask=masks, token_type_ids=token_type_ids)    

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
    def eval_step(self, data_loader, metric, n_class):
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
                if model_name in ["DISTILBERT", "ROBERTA"]:
                    ids = data["ids"].to(self.device)
                    masks = data["masks"].to(self.device)
                    labels = data["labels"].to(self.device)
                    # GETTING PREDICTION FROM MODEL
                    output = self.model(ids=ids, mask=masks)

                elif model_name in ["BERT"]:
                    ids = data["ids"].to(self.device)
                    masks = data["masks"].to(self.device)
                    token_type_ids = data["token_type_ids"].to(self.device)
                    # GETTING PREDICTION FROM MODEL
                    output = self.model(ids=ids, mask=masks, token_type_ids=token_type_ids)    

                # CALCULATE LOSS & METRICS
                loss = self.criterion(output, labels)

                metric_used = metrics_dict[metric]
                predictions = torch.softmax(output, dim=1)
                _, predictions = torch.max(predictions, dim=1)

                metric_value = metric_used(labels, predictions, n_class)

                losses.update(loss.item(), ids.size(0))
                metrics_avg.update(metric_value.item(), ids.size(0))

                tk0.set_postfix(loss=losses.avg)
        print(f"Validation Loss = {losses.avg}")
        return loss, metrics_avg.avg
