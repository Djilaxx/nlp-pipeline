import warnings
warnings.filterwarnings("ignore")

from pytorch_lightning.metrics import functional

metrics_dict = {
    "ACCURACY" : functional.accuracy,
    "AUROC" : functional.auroc,
    "CONFUSION_MATRIX" : functional.confusion_matrix,
    "F1_SCORE" : functional.f1_score,
}