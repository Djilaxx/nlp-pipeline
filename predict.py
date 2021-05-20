import os
import inspect
import importlib
import argparse

import pandas as pd 
import numpy as np

##########
# PARSER #
##########
parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str, default="TPS-FEV2021")
parser.add_argument("--model_name", type=str, default="LGBM")
parser.add_argument("--task", type=str, default="REG")

args = parser.parse_args()
##################
# START TRAINING #
##################
if __name__ == "__main__":
    print("Prediction start...")
    predict(
        project=args.project,
        model_name=args.model_name,
        model_task=args.model_task,
    )
