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
from datasets import nlp_dataset
from utils import early_stopping, folding
from trainer.trainer import Trainer

##################
# TRAIN FUNCTION #
##################
def train(folds=5, project="tweet_disaster", model_name="distilbert", model_type="CL"):
    complete_name = f"{model_name}_{model_type}"
    print(f"Training on task : {project} for {folds} folds with {complete_name} model")

    # Loading project config
    config = getattr(importlib.import_module(f"project.{project}.config"), "config")
    
    # CREATING FOLDS
    folding.create_folds(datapath=config.main.TRAIN_FILE,
                        output_path=config.main.FOLD_FILE,
                        nb_folds = folds,
                        method=config.main.FOLD_METHOD,
                        target=config.main.TARGET_VAR)

    # LOADING DATA FILE & TOKENIZER
    df = pd.read_csv(config.main.FOLD_FILE)

    # FEATURE ENGINEERING

    # MODEL & TOKENIZER

    # METRIC

    # FOLD LOOP

    
##########
# PARSER #
##########
parser = argparse.ArgumentParser()
parser.add_argument("--folds", type=int, default=5)
parser.add_argument("--project", type=str, default="tweet_disaster")
parser.add_argument("--model_name", type=str, default="distilbert")
parser.add_argument("--model_type", type=str, default="CL")

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
        model_type=args.model_type,
    )