from easydict import EasyDict as edict

config = edict()
########
# MAIN #
########
# main is the config section related to basic info on the project
# data repo, data format, folding etc... data preparation
config.main = edict()
config.main.PROJECT_PATH = "project/tweet_disaster/"
config.main.TRAIN_FILE = "data/tweet_disaster/train.csv"
config.main.TEST_FILE = "data/tweet_disaster/test.csv"
config.main.SUBMISSION = "data/tweet_disaster/sample_submission.csv"
config.main.FOLD_FILE = "data/tweet_disaster/train_folds.csv"
config.main.FOLD_NUMBER = 5
config.main.FOLD_METHOD = "SKF"
config.main.TARGET_VAR = "Survived"
#######################
# TRAINING PARAMETERS #
#######################
config.train = edict()
config.train.ES = 50
config.train.METRIC = "ACCURACY"

####################
# MODEL PARAMETERS #
####################
###################
# HYPERPARAMETERS #
###################
config.model = edict()