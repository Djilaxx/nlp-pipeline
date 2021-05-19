from easydict import EasyDict as edict
import torch

config = edict()
########
# MAIN #
########
# main is the config section related to basic info on the project
# data repo, data format, folding etc... data preparation
config.main = edict()
config.main.PROJECT_PATH = "projects/allocine/"
config.main.TRAIN_FILE = "data/allocine/train.csv"
config.main.TEST_FILE = "data/allocine/test.csv"
config.main.SUBMISSION = "data/allocine/sample_submission.csv"
config.main.FOLD_FILE = "data/allocine/train_folds.csv"
config.main.FOLD_NUMBER = 5
config.main.FOLD_METHOD = "SKF"
config.main.TARGET_VAR = "polarity"
config.main.TEXT_VAR = "review"
config.main.DEVICE = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
config.main.N_CLASS = 2
config.main.MAX_LEN = 160
#######################
# TRAINING PARAMETERS #
#######################
config.train = edict()
config.train.EPOCHS = 1
config.train.TRAIN_BS = 64
config.train.VALID_BS = 32
config.train.ES = 50
config.train.LR = 1e-4
config.train.LOSS = "CROSS_ENTROPY"
config.train.METRIC = "ACCURACY"

####################
# MODEL PARAMETERS #
####################
###################
# HYPERPARAMETERS #
###################
config.model = edict()
