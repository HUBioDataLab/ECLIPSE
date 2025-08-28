import copy
import yaml
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef


#----Load configuration from file or use defaults-----
def load_config(config_path):
    """Load configuration from file or use defaults."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    

#-----Convert real continous values into binary class labels for classification metrics-----
def binarization(y,thr):
    y_binary = copy.deepcopy(y)
    y_binary = binarize(y_binary.reshape(1,-1), threshold=thr, copy=False)[0]
    return y_binary


#-----Calculate model performance scores-----
def performance_scores(true_value, pred, tr_pchembl_median):
    pred_bin = binarization(pred, tr_pchembl_median)
    true_value_bin = binarization(true_value, tr_pchembl_median)

    loss = round(float(F.mse_loss(pred, true_value)), 4)
    rmse = round(np.sqrt(loss), 4)
    pearson = round(pearsonr(true_value, pred)[0], 4)
    spearman = round(spearmanr(true_value, pred)[0], 4)
 
    mcc = round(matthews_corrcoef(true_value_bin, pred_bin), 4)
    accuracy = round(accuracy_score(true_value_bin, pred_bin), 4)
    precision = round(precision_score(true_value_bin, pred_bin), 4)
    recall = round(recall_score(true_value_bin, pred_bin), 4)
    f1 = round(f1_score(true_value_bin, pred_bin), 4)

    return [loss, rmse, pearson, spearman, mcc, accuracy, precision, recall, f1]
