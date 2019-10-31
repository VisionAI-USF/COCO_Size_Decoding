import numpy as np
from sklearn.metrics import roc_auc_score
import os

def get_CV_AUC(params):
    catNames = params['category']
    for category in catNames:
        AUC = []
        for run in range(5):
            for fold in range(2):
                dir_path = params['tmp_dir'] + '/auc_src/{}/{}/{}/'.format(category,run,fold)
                cauc = auc_value(dir_path)
                AUC.append(cauc)
        fname = params['tmp_dir'] + '/auc_{}.txt'.format(category)
        with open(fname,'w') as f:
            for cauc in AUC:
                f.write("{}\n".format(cauc))







def auc_value(dir_path):

    y_true = np.load(dir_path + 'test_labels.npy')
    y_scores = np.load(dir_path + 'cnn_predictions.npy')
    
    y_true = y_true[:,0]
    y_scores = y_scores[:,0]

    auc = roc_auc_score(y_true, y_scores)
    return auc