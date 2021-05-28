##################
# IMPORT MODULES #
##################
# SYS IMPORT

from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
import os
import glob
import inspect
import importlib
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ML IMPORT
# MY OWN MODULES
from trainer.TRAINER import TRAINER
from datasets.NLP_DATASET import NLP_DATASET

def predict(project="tweet_disaster", model_name="DISTILBERT"):
    print(f"Predictions on project : {project} with {model_name} model")
    # CONFIG
    config = getattr(importlib.import_module(f"projects.{project}.config"), "config")

    # LOADING DATA FILE
    df_test = pd.read_csv(config.main.TEST_FILE)

    # FEATURE ENGINEERING FUNCTION
    try:
        feature_eng = getattr(importlib.import_module(f"projects.{project}.feature_eng"), "feature_engineering")
    except:
        print("No feature_engineering function")
        feature_eng = None
    
    # TOKENIZER
    tokenizer = getattr(importlib.import_module(f"models.{model_name}.tokenizer"), "tokenizer")
    tokenizer = tokenizer()

    # LOADING MODEL
    for name, cls in inspect.getmembers(importlib.import_module(f"models.{model_name}.model"), inspect.isclass):
        if name == model_name:
            model = cls(
                task=config.main.TASK, model_config_path=f"models/{model_name}/config", n_class=config.main.N_CLASS)

    model.to(config.main.DEVICE)
    ########################
    # CREATING DATALOADERS #
    ########################
    # TEST IDs & LABELS
    test_text = df_test[config.main.TEXT_VAR].values.tolist()

    # TEST DATASET
    test_ds = NLP_DATASET(
        model_name=model_name,
        task=config.main.TASK,
        text=test_text,
        labels=None,
        max_len=config.main.MAX_LEN,
        tokenizer=tokenizer,
        feature_eng=feature_eng
    )

    # TEST DATALOADER
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=config.train.TRAIN_BS, 
        shuffle=False, 
        num_workers=0
    )

    # PREDICTION LOOP
    final_preds = None
    for fold in range(config.main.PREDICTION_FOLD_NUMBER):
        print(f"Starting predictions for fold  : {fold}")
        # LOAD MODEL WITH FOLD WEIGHTS
        weights = torch.load(config.main.WEIGHTS_PATH.rsplit("_", 1)[0] + "_" + str(fold) + ".bin")
        model.load_state_dict(weights)
        model.eval()

        trainer = TRAINER(model=model, task=config.main.TASK, device=config.main.DEVICE, optimizer=None, criterion=None)
        # DATA LOADER LOOP
        predictions = trainer.test_step(data_loader=test_loader, n_class=config.main.N_CLASS)
        predictions = np.vstack(predictions)
        predictions = predictions.reshape(len(df_test), 1, config.main.N_CLASS)
        temp_preds = None
        for p in predictions:
            if temp_preds is None:
                temp_preds = p
            else:
                temp_preds = np.vstack((temp_preds, p))
        if final_preds is None:
            final_preds = temp_preds
        else:
            final_preds += temp_preds

    final_preds /= config.main.PREDICTION_FOLD_NUMBER
    if config.main.PREDICT_PROBA is False:
        final_preds = final_preds.argmax(axis=1)
    # CONDITIONAL SUBMISSION FILE DEPENDING IF WE HAVE A TEST FILE OR NOT
    test_final_data = {config.main.TEXT_VAR: df_test[config.main.TEXT_VAR].values.tolist(), config.main.TARGET_VAR: final_preds}
    test_df = pd.DataFrame(data=test_final_data, index=None)
    test_df.to_csv(os.path.join(config.main.PROJECT_PATH, "preds.csv"))

##########
# PARSER #
##########
parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str, default="tweet_disaster")
parser.add_argument("--model_name", type=str, default="DISTILBERT")

args = parser.parse_args()
##################
# START TRAINING #
##################
if __name__ == "__main__":
    print("Prediction start...")
    predict(
        project=args.project,
        model_name=args.model_name    
    )
