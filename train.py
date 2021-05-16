##################
# IMPORT MODULES #
##################
# SYS IMPORT
import os, inspect, importlib, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import gc
import pandas as pd
import numpy as np
from pathlib import Path
# ML IMPORT
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import BertTokenizer
from sklearn import metrics
from sklearn import model_selection
# MY OWN MODULES
from utils import early_stopping, folding
from utils.metrics import metrics_dict
from trainer.trainer import Trainer

##################
# TRAIN FUNCTION #
##################
def train(folds=5, project="tweet_disaster", model_name="distilbert", task="CL"):
    # CHECKING MODEL TYPE
    if model_name in ["BERT", "DISTILBERT", "ROBERTA"]:
        model_type = "TRANSFORMER"
    elif model_name in ["LSTM", "GRU"]:
        model_type = "RNN"

    complete_name = f"{model_name}_{task}"
    print(f"Training on task : {project} for {folds} folds with {complete_name} model")

    # CONFIG
    config = getattr(importlib.import_module(f"project.{project}.config"), "config")
    
    # CREATING FOLDS
    folding.create_folds(datapath=config.main.TRAIN_FILE,
                        output_path=config.main.FOLD_FILE,
                        nb_folds = folds,
                        method=config.main.FOLD_METHOD,
                        target=config.main.TARGET_VAR)

    # LOADING DATA FILE & TOKENIZER
    df = pd.read_csv(config.main.FOLD_FILE)

    # MODEL 
    # NEED TO BE ADAPTED TO BE ABLE TO RUN RNN MODELS AND TRANSFORMERS WITH LOADED CONFIG #####
    # TOKENIZER
    if model_type == "TRANSFORMER":
        tokenizer = getattr(importlib.import_module(f"models.{model_type}.{model_name}.tokenizer"), "tokenizer")
        tokenizer = tokenizer()

    # LOADING MODEL
    for name, cls in inspect.getmembers(importlib.import_module(f"models.{model_type}.{model_name}.model"), inspect.isclass):
        if name == model_name:
            # SOLVE REGRESSION VS CLASSIFICATION LOADING PROBLEM IN MODELS
            if model_type == "TRANSFORMER":
                model = cls(task = task, n_class = config.main.N_CLASS, model_config_path = f"models/{model_name}/config")

    # METRIC
    metric_selected = metrics_dict[config.train.METRIC]
    # FOLD LOOP
    for fold in range(folds):
        print(f"Starting training for fold : {fold}")

        # CREATING TRAINING AND VALIDATION SETS
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        model.to(config.main.DEVICE)
        ########################
        # CREATING DATALOADERS #
        ########################
        # TRAINING IDs & LABELS
        trn_text = df_train[config.main.TEXT_VAR].values.tolist()
        trn_labels = df_train[config.main.TARGET_VAR].values
        # VALIDATION IDs & LABELS
        valid_text = df_valid[config.main.TEXT_VAR].values.tolist()
        valid_labels = df_valid[config.main.TARGET_VAR].values
        # TRAINING DATASET
        #if model_name in ["DISTILBERT", "BERT", "ROBERTA"]:
        dataset_fct = getattr(importlib.import_module(f"datasets.{model_type}_dataset"), "NLP_DATASET")
        train_ds = dataset_fct(
            model_name = complete_name,
            text = trn_text,
            labels = trn_labels,
            max_len = config.main.MAX_LEN,
            tokenizer = config.train.tokenizer,
        )
        # TRAINING DATALOADER
        train_loader = torch.utils.data.DataLoader(
            train_ds, 
            batch_size=config.train.TRAIN_BS, 
            shuffle=True, 
            num_workers=0
        )
        # VALIDATION DATASET
        valid_ds = dataset_fct(
            model_name = complete_name,
            text = valid_text,
            labels = valid_labels,
            max_len = config.main.MAX_LEN,
            tokenizer = config.train.tokenizer
        )
        # VALIDATION DATALOADER
        valid_loader = torch.utils.data.DataLoader(
            valid_ds, 
            batch_size=config.hyper.VALID_BS, 
            shuffle=True, 
            num_workers=0
        )

        # IMPORT LOSS FUNCTION
        loss_module = importlib.import_module(f"loss.{config.train.loss}")
        criterion = loss_module.loss_function()
        # SET OPTIMIZER, SCHEDULER
        optimizer = torch.optim.Adam(model.parameters(), lr=config.train.LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        # SET EARLY STOPPING FUNCTION
        es = early_stopping.EarlyStopping(patience=2, mode="max")
        # CREATE TRAINER
        trainer_fct = getattr(importlib.import_module(f"trainer.{model_type}_trainer"), "TRAINER")
        trainer = trainer_fct(model, optimizer, config.main.DEVICE, criterion, task)

        # START TRAINING FOR N EPOCHS
        for epoch in range(config.train.EPOCHS):
            print(f"Starting epoch number : {epoch}")
            # TRAINING PHASE
            print("Training the model...")
            trainer.training_step(train_loader)
            # VALIDATION PHASE
            print("Evaluating the model...")
            val_loss, metric_value = trainer.eval_step(valid_loader, metric_selected, config.main.N_CLASS)
            scheduler.step(val_loss)
            # METRICS
            print(f"Validation {metric_selected} = {metric_value}")
            #SAVING CHECKPOINTS
            Path(os.path.join(config.main.PROJECT_PATH, "model_output/")).mkdir(parents=True, exist_ok=True)
            es(metric_value, model, model_path=os.path.join(config.main.PROJECT_PATH, "model_output/", f"model_{fold}.bin"))
            if es.early_stop:
                print("Early Stopping")
                break
            gc.collect()

##########
# PARSER #
##########
parser = argparse.ArgumentParser()
parser.add_argument("--folds", type=int, default=5)
parser.add_argument("--project", type=str, default="tweet_disaster")
parser.add_argument("--model_name", type=str, default="distilbert")
parser.add_argument("--task", type=str, default="CL")

args = parser.parse_args()
##################
# START TRAINING #
##################
if __name__ == "__main__":
    print("Training start...")
    train(
        folds=args.folds,
        task=args.project,
        model_name=args.model_name,
        task=args.model_type,
    )
