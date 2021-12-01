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

def predict(project="tweet_disaster", model_name="DISTILBERT", run_note="test"):
    """
    Predict on a test dataset using a trained model

    Parameters
    ----------
    project: str
        the name of the project you wish to work on - must be the name of the project folder under projects/
    model_name: str
        the name of the model you wish to use to predict - name of the python file under models/
    run_note: str
        An string note for your current run
    Returns
    -------
    Create a csv file containing the model predictions
    """
    
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
    final_preds = []
    for fold in range(max(config.main.FOLD_NUMBER, 1)):
        print(f"Starting predictions for fold  : {fold}")
        # LOAD MODEL WITH FOLD WEIGHTS
        model_weights = torch.load(os.path.join(
            config.main.PROJECT_PATH, "model_output/", f"model_{model_name}_{run_note}_{fold}.bin"))
        model.load_state_dict(model_weights)
        model.eval()

        trainer = TRAINER(model=model, task=config.main.TASK, device=config.main.DEVICE, n_class=config.main.N_CLASS)
        # DATA LOADER LOOP
        predictions = trainer.test_step(data_loader=test_loader)
        final_preds.append(predictions)

        if config.main.SPLIT is True:
            break

    final_preds = np.mean(np.column_stack(final_preds), axis=1)
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
parser.add_argument("--run_note", type=str, default="test")

args = parser.parse_args()
##################
# START TRAINING #
##################
if __name__ == "__main__":
    print("Training start...")
    predict(
        project=args.project,
        model_name=args.model_name,
        run_note=args.run_note
    )
