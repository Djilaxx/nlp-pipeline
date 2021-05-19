from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

metrics_dict = {
    "ACCURACY": metrics.accuracy_score,
    "AUC": metrics.roc_auc_score,
    "MSE": metrics.mean_squared_error,
    "MAE": metrics.mean_absolute_error
}
