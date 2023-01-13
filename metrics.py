import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    np_preds = np.array(predictions)
    np_labels = np.array(labels)

    mse_valence = mean_squared_error(np_labels[:,0], np_preds[:,0])
    mae_valence = mean_absolute_error(np_labels[:,0], np_preds[:,0])   
    pearson_corr_valence = stats.pearsonr(np_labels[:,0], np_preds[:,0])[0] # The first element of pearsonr is pearson correlation

    mse_arousal = mean_squared_error(np_labels[:,1], np_preds[:,1])
    mae_arousal = mean_absolute_error(np_labels[:,1], np_preds[:,1])   
    pearson_corr_arousal = stats.pearsonr(np_labels[:,1], np_preds[:,1])[0] 

    print('\n \n')
    print("mse_valence : " + str(mse_valence) + '\n' + 
            "mae_valence : " + str(mae_valence) + '\n' + 
            "pearson_corr_valence : " + str(pearson_corr_valence) + '\n' + 
            "mse_arousal : " + str(mse_arousal) + '\n' + 
            "mae_arousal : " + str(mae_arousal) + '\n' + 
            "pearson_corr_arousal : " + str(pearson_corr_arousal))
    print('\n \n')

    return {
        "mse_valence" : mse_valence, 
        "mae_valence" : mae_valence,
        "pearson_corr_valence": pearson_corr_valence,
        "mse_arousal" : mse_arousal, 
        "mae_arousal" : mae_arousal,
        "pearson_corr_arousal": pearson_corr_arousal
        }