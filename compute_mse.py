import numpy as np
from sklearn.metrics import roc_auc_score
import os

def get_CV_MSE(params):
    catNames = params['category']
    txt = ''

    for category in catNames:
        AUC = []
        for run in range(5):
            for fold in range(2):
                dir_path = params['tmp_dir'] + '/auc_src/{}/{}/{}/'.format(category,run,fold)
                mse, cc = mse_value(dir_path)
                txt = txt + '{}({}) '.format(mse,cc)
            txt = txt + '\n'
        fname = params['tmp_dir'] + '/mse_{}.txt'.format(category)
        with open(fname,'w') as f:
            f.write(txt)
                







def mse_value(dir_path):

    y_true = np.load(dir_path + 'test_labels.npy')
    y_scores = np.load(dir_path + 'cnn_predictions.npy')
    y_scores = y_scores[:,0]

    mse = 0.0
    for j in range(len(y_true)):
        mse = (y_true[j]-y_scores[j])*(y_true[j]-y_scores[j])
    mse = mse / len(y_true)
    sort_indeces = np.argsort(y_true)
    y_true = y_true[sort_indeces]
    y_scores = y_scores[sort_indeces]
    cc = np.corrcoef(y_true,y_scores)
    cc = cc[0, 1]
    return mse, cc