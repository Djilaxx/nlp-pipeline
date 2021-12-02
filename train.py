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
import wandb
# ML IMPORT
import torch
# MY OWN MODULES
from datasets.NLP_DATASET import NLP_DATASET
from trainer.TRAINER import TRAINER
from utils import early_stopping, folding
from utils.memory_usage import reduce_memory_usage

##################
# TRAIN FUNCTION #
##################
def train(project="tweet_disaster", model_name="DISTILBERT", run_note="test"):
    """
    Train, validate, and log results of a model on a specified dataset

    Parameters
    ----------
    project: str
        the name of the project you wish to work on - must be the name of the project folder under projects/
    model_name: str
        the name of the model you wish to train - name of the python file under models/
    run_note: str
        An string note for your current run

    Returns
    -------
    save trained model under projects/model_saved/
    print training results and log to wandb
    """

    # LOADING PROJECT CONFIG
    config = getattr(importlib.import_module(f"projects.{project}.config"), "config")
    if config.main.SPLIT is True:
        print(f"Starting run {run_note}, training on project : {project} with {model_name} model")
    else:
        print(f"Starting run {run_note}, training on project : {project} for {config.main.FOLD_NUMBER} folds with {model_name} model")
    complete_name = f"{model_name}_{config.main.TASK}"
    # RECORD RUNS USING WANDB TOOL
    wandb.init(config=config, project=project, name=complete_name + "_" + str(run_note))
    # CREATING FOLDS
    df = pd.read_csv(config.main.TRAIN_FILE)
    df = reduce_memory_usage(df, verbose=True)
    df = folding.create_splits(df=df,
                               task=config.main.TASK,
                               n_folds=config.main.FOLD_NUMBER,
                               split=config.main.SPLIT,
                               split_size=config.main.SPLIT_SIZE,
                               target=config.main.TARGET_VAR)

    # FEATURE ENGINEERING FUNCTION
    try:
        feature_eng = getattr(importlib.import_module(f"projects.{project}.feature_eng"), "feature_engineering")
    except:
        print("No feature_engineering function")
        feature_eng = None

    # TOKENIZER
    tokenizer = getattr(importlib.import_module(f"models.{model_name}.tokenizer"), "tokenizer")
    tokenizer = tokenizer()

    # FOLD LOOP
    for fold in range(max(config.main.FOLD_NUMBER, 1)):
        print("Starting training...") if config.main.SPLIT is True else print(f"Starting training for fold {fold}")
        # CREATING TRAINING AND VALIDATION SETS
        df_train = df[df.splits != fold].reset_index(drop=True)
        df_valid = df[df.splits == fold].reset_index(drop=True)

        # LOADING MODEL
        for name, cls in inspect.getmembers(importlib.import_module(f"models.{model_name}.model"), inspect.isclass):
            if name == model_name:
                # DOWNLOADING THE MODEL CONFIG BEFOREHAND IS FASTER BUT WE CAN DOWNLOAD DIRECTLY WHEN WE START TRAINING
                if os.path.isdir(f"models/{model_name}/config"):
                    model = cls(task=config.main.TASK, model_config_path=f"models/{model_name}/config", n_class=config.main.N_CLASS)
                else:
                    model = cls(task=config.main.TASK, n_class=config.main.N_CLASS)

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

        wandb.watch(model, criterion=criterion, idx=fold)
        # START TRAINING FOR N EPOCHS
        for epoch in range(config.train.EPOCHS):
            print(f"Starting epoch number : {epoch}")
            # TRAINING PHASE
            print("Training the model...")
            train_loss, train_metrics = trainer.training_step(train_loader)
            print(f"Training loss for epoch {epoch}: {train_loss}")
            wandb.log({f"Training loss": train_loss})
            for metric_name, metric_value in train_metrics.items():
                print(f"Training {metric_name} score for epoch {epoch}: {metric_value.avg}")
                wandb.log({f"Training {metric_name} score": metric_value.avg})

            # VALIDATION PHASE
            print("Evaluating the model...")
            val_loss, valid_metrics = trainer.validation_step(valid_loader)
            print(f"Validation loss for epoch {epoch}: {val_loss}")
            wandb.log({f"Validation loss": val_loss})
            for metric_name, metric_value in valid_metrics.items():
                print(f"Validation {metric_name} score for epoch {epoch}: {metric_value.avg}")
                wandb.log({f"Validation {metric_name} score": metric_value.avg})

            scheduler.step(val_loss)

            #SAVING CHECKPOINTS
            Path(os.path.join(config.main.PROJECT_PATH, "model_output/")).mkdir(parents=True, exist_ok=True)
            es(
                valid_metrics[config.train.METRIC].avg,
                model,
                model_path=os.path.join(config.main.PROJECT_PATH, "model_output/", f"model_{model_name}_{run_note}_{fold}.bin")
            )
            if es.early_stop:
                print("Early Stopping")
                break
            gc.collect()

        # IF WE GO FOR A TRAIN - VALID SPLIT WE TRAIN ONE MODEL ONLY (folds=0 or 1)
        if config.main.SPLIT is True:
            break


##########
# PARSER #
##########
parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str, default="tweet_disaster")
parser.add_argument("--model_name", type=str, default="DISTILBERT")
parser.add_argument("--run_note", type=str, default="test")

args = parser.parse_args()
##################
# START TRAINING #
##################
if __name__ == "__main__":
    print("Training start...")
    train(
        project=args.project,
        model_name=args.model_name,
        run_note=args.run_note
    )
