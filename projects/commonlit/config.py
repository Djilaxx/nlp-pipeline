from easydict import EasyDict as edict
import torch

config = edict()
########
# MAIN #
########
# main is the config section related to basic info on the project
# data repo, data format, folding etc... data preparation
config.main = edict()
config.main.PROJECT_PATH = "projects/commonlit/"
config.main.TRAIN_FILE = "data/commonlit/train.csv"
config.main.TEST_FILE = "data/commonlit/test.csv"
config.main.SUBMISSION = "data/commonlit/sample_submission.csv"
config.main.FOLD_FILE = "data/commonlit/train_folds.csv"
config.main.FOLD_NUMBER = 5
config.main.SPLIT_SIZE = 0.2
config.main.TASK = "REGRESSION"
config.main.TARGET_VAR = "target"
config.main.TEXT_VAR = "excerpt"
config.main.DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
config.main.N_CLASS = 1
config.main.MAX_LEN = 320
config.main.PREDICTION_FOLD_NUMBER = 5
config.main.WEIGHTS_PATH = "projects/commonlit/model_output/model_DISTILBERT_2021-05-28_0.bin"
config.main.PREDICT_PROBA = False
#######################
# TRAINING PARAMETERS #
#######################
config.train = edict()
config.train.EPOCHS = 1
config.train.TRAIN_BS = 16
config.train.VALID_BS = 8
config.train.ES = 5
config.train.LR = 1e-4
config.train.LOSS = "MSE"
config.train.METRIC = "MSE"

####################
# MODEL PARAMETERS #
####################
###################
# HYPERPARAMETERS #
###################
config.model = edict()
