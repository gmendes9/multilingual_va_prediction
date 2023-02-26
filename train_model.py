import os
import sys
import json
from datetime import datetime
from signal import signal
from utils import create_prediction_tables, handle_signal
from data_loader import MyDataset
from fold1 import training_fold1
from fold2 import training_fold2



# CUDA_LAUNCH_BLOCKING make cuda report the error where it actually occurs
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

preds_dir = 'temp'
run_parameters={}

# TODO Completely separate 2 fold training

if __name__ == '__main__':
    # defining  your handler for SIGINT (2)
    signal(2, handle_signal)

    # Allowed command line arguments
    models = ['distilbert', 'xlmroberta-base', 'xlmroberta-large']
    losses = ['mse', 'ccc', 'robust', 'mse+ccc', 'robust+ccc']

    # Get command line arguments
    try:
        n = len(sys.argv)
        model = sys.argv[1]
        loss = sys.argv[2]
        if(n < 2 or model not in models or loss not in losses):
            print('\nPlease use 2 arguments. Model and Loss. Arguments accepted:')
            print('Model: distilbert | xlmroberta-base | xlmroberta-large')
            print('Loss: mse | ccc | robust | mse+ccc | robust+ccc\n')
            quit()
    except:
        print('\nPlease use 2 arguments. Model and Loss. Arguments accepted:')
        print('Model: distilbert | xlmroberta-base | xlmroberta-large')
        print('Loss: mse | ccc | robust | mse+ccc | robust+ccc\n')
        quit()
    
    if(model == 'distilbert'):
        checkpoint = "distilbert-base-multilingual-cased"
    elif(model == 'xlmroberta-base'):
        checkpoint = "xlm-roberta-base"
    elif(model == 'xlmroberta-large'):
        checkpoint = "xlm-roberta-large"
        
    run_parameters['model'] = model
    run_parameters['loss_function'] = loss
    
    # Creates directories
    now = datetime.now()
    timestamp = now.strftime("%b-%d_%H-%M-%S")
    
    try:
        preds_dir = 'Preds/' + timestamp + "_" + os.environ['COMPUTERNAME']
    except:
        preds_dir = 'Preds/' + timestamp + "_" + os.environ['HOST']

    os.makedirs(preds_dir)
    run_parameters['path'] = preds_dir
    
    # Parameters
    params = {
        'batch_size_distil' : 16,
        'batch_size_xlmrB' : 16,
        'batch_size_xlmrL' : 16,
        'lr' : 6e-6,
        'train_epochs' : 10,
        'weight_decay' : 0.01,
        'warmup_ratio': 0.1,
    }
    
    # Write parameters to json file
    for k,v in params.items():
        run_parameters[k] = v
    with open(preds_dir + '/training_parameters.json', "w") as f:
        json.dump(run_parameters,f)
        
    # Dataset
    data_dir = "data" 
    filename_1 = os.path.join(data_dir, "full_dataset_fold1.csv")
    filename_2 = os.path.join(data_dir, "full_dataset_fold2.csv")

    
    split_1 = MyDataset(filename=filename_1, checkpoint=checkpoint, maxlen=200)
    split_2 = MyDataset(filename=filename_2, checkpoint=checkpoint, maxlen=200)
    dataset = [[split_1, split_2], [split_2, split_1]]

    # Train and Prediction
    training_fold1(model, loss, timestamp, params, dataset, preds_dir, checkpoint)
    print('\n\n\n------------ NOW ON FOLD 2 -------------- \n\n\n')
    training_fold2(model, loss, timestamp, params, dataset, preds_dir, checkpoint)
    
    create_prediction_tables(preds_dir)

    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    