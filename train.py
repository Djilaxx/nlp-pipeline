##################
# IMPORT MODULES #
##################
# SYS IMPORT
import os, inspect, importlib, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import gc
import pandas as pd
from pathlib import Path
import datetime
# ML IMPORT
import torch
# MY OWN MODULES
from datasets.NLP_DATASET import NLP_DATASET
from trainer.TRAINER import TRAINER
from utils import early_stopping, folding
from utils.metrics import metrics_dict

##################
# TRAIN FUNCTION #
##################
def train(folds=5, project="tweet_disaster", model_name="distilbert"):
    print(f"Training on project : {project} for {folds} folds with {model_name} model")
    # CONFIG
    config = getattr(importlib.import_module(f"projects.{project}.config"), "config")
    # CREATING FOLDS
    folding.create_folds(datapath=config.main.TRAIN_FILE,
                        output_path=config.main.FOLD_FILE,
                        nb_folds = folds,
                        method=config.main.FOLD_METHOD,
                        target=config.main.TARGET_VAR)

        # LOADING DATA FILE & TOKENIZER
    df = pd.read_csv(config.main.FOLD_FILE)

    # FEATURE ENGINEERING FUNCTION
    try:
        feature_eng = getattr(importlib.import_module(f"projects.{project}.feature_eng"), "feature_engineering")
    except:
        print("No feature_engineering function")
        feature_eng = None
    # MODEL 
    # NEED TO BE ADAPTED TO BE ABLE TO RUN RNN MODELS AND TRANSFORMERS WITH LOADED CONFIG #####
    # TOKENIZER
    tokenizer = getattr(importlib.import_module(f"models.{model_name}.tokenizer"), "tokenizer")
    tokenizer = tokenizer()
    # METRIC
    metric_selected = metrics_dict[config.train.METRIC]
    # FOLD LOOP
    for fold in range(folds):
        print(f"Starting training for fold : {fold}")
        # CREATING TRAINING AND VALIDATION SETS
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        # LOADING MODEL
        for name, cls in inspect.getmembers(importlib.import_module(f"models.{model_name}.model"), inspect.isclass):
            if name == model_name:
                model = cls(task=config.main.TASK, model_config_path=f"models/{model_name}/config", n_class=config.main.N_CLASS)

        model.to(config.main.DEVICE)
        ########################
        # CREATING DATALOADERS #
        ########################
        # TRAINING IDs & LABELS
        train_text = df_train[config.main.TEXT_VAR].values.tolist()
        train_labels = df_train[config.main.TARGET_VAR].values
        # VALIDATION IDs & LABELS
        valid_text = df_valid[config.main.TEXT_VAR].values.tolist()
        valid_labels = df_valid[config.main.TARGET_VAR].values
        # TRAINING DATASET
        train_ds = NLP_DATASET(
            model_name = model_name,
            task = config.main.TASK,
            text=train_text,
            labels=train_labels,
            max_len = config.main.MAX_LEN,
            tokenizer = tokenizer,
            feature_eng=feature_eng
        )
        # TRAINING DATALOADER
        train_loader = torch.utils.data.DataLoader(
            train_ds, 
            batch_size=config.train.TRAIN_BS, 
            shuffle=True, 
            num_workers=0
        )
        # VALIDATION DATASET
        valid_ds = NLP_DATASET(
            model_name = model_name,
            task = config.main.TASK,
            text = valid_text,
            labels = valid_labels,
            max_len = config.main.MAX_LEN,
            tokenizer=tokenizer,
            feature_eng=feature_eng
        )
        # VALIDATION DATALOADER
        valid_loader = torch.utils.data.DataLoader(
            valid_ds, 
            batch_size=config.train.VALID_BS, 
            shuffle=True, 
            num_workers=0
        )

        # IMPORT LOSS FUNCTION
        loss_function = getattr(importlib.import_module(f"loss.{config.train.LOSS}"), "loss_function")
        criterion = loss_function()
        # SET OPTIMIZER, SCHEDULER
        optimizer = torch.optim.Adam(model.parameters(), lr=config.train.LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        # SET EARLY STOPPING FUNCTION
        es = early_stopping.EarlyStopping(patience=2, mode="max")
        # CREATE TRAINER
        trainer = TRAINER(model=model,
                          optimizer=optimizer,
                          device=config.main.DEVICE,
                          criterion=criterion,
                          task=config.main.TASK)

        # START TRAINING FOR N EPOCHS
        for epoch in range(config.train.EPOCHS):
            print(f"Starting epoch number : {epoch}")
            # TRAINING PHASE
            print("Training the model...")
            trainer.training_step(train_loader)
            # VALIDATION PHASE
            print("Evaluating the model...")
            val_loss, metric_value = trainer.eval_step(valid_loader, metric_selected)
            scheduler.step(val_loss)
            # METRICS
            print(f"Validation {config.train.METRIC} = {metric_value}")
            #SAVING CHECKPOINTS
            Path(os.path.join(config.main.PROJECT_PATH, "model_output/")).mkdir(parents=True, exist_ok=True)
            es(
                metric_value, 
                model, 
                model_path=os.path.join(config.main.PROJECT_PATH, "model_output/",f"model_{model_name}_{fold}_{round(metric_value, 3)}_{str(datetime.date.today().isoformat())}.bin")
            )
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

args = parser.parse_args()
##################
# START TRAINING #
##################
if __name__ == "__main__":
    print("Training start...")
    train(
        folds=args.folds,
        project=args.project,
        model_name=args.model_name    
        )
