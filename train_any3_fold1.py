from transformers import RobertaForSequenceClassification, TrainingArguments, DataCollatorWithPadding, Trainer
from models import DistilBertForSequenceClassificationSig, XLMRobertaForSequenceClassificationSig
from data_loader import MyDataset
from custom_trainer import CustomTrainerMSE, CustomTrainerCCC, CustomTrainerRobust, CustomTrainerMSE_CCC, CustomTrainerRobustCCC
from metrics import compute_metrics
import torch
import pandas as pd
import numpy as np
import json
from utils import create_prediction_tables


    
def training_fold1(model, loss, timestamp, params, dataset, preds_dir, checkpoint):
    output_dir1 = "Output Directory/" + timestamp + "/fold1"
    
    model_dir = "model/" + timestamp + "/fold1"
    log_dir = "runs/" + timestamp + "/fold1"
    
    # Chooses the model
    if(model == 'distilbert'):
        checkpoint = checkpoint
        model = DistilBertForSequenceClassificationSig.from_pretrained(checkpoint, num_labels=2)
        batch_size = params['batch_size_distil']
    elif(model == 'xlmroberta-base'):
        checkpoint = checkpoint
        model = XLMRobertaForSequenceClassificationSig.from_pretrained(checkpoint, num_labels=2)
        batch_size = params['batch_size_xlmrB']
    elif(model == 'xlmroberta-large'):
        checkpoint = checkpoint
        model = XLMRobertaForSequenceClassificationSig.from_pretrained(checkpoint, num_labels=2)
        batch_size = params['batch_size_xlmrL']
        
    training_args = TrainingArguments(
        output_dir=output_dir1,
        logging_dir='logs/logs1',
        logging_steps=200,
        per_device_train_batch_size=batch_size, 
        per_device_eval_batch_size=batch_size, 
        num_train_epochs=params['train_epochs'],
        learning_rate=params['lr'], 
        weight_decay=params['weight_decay'],
        group_by_length=True,
        evaluation_strategy="epoch", 
        save_strategy="epoch",
        load_best_model_at_end=True,
        warmup_ratio=params['warmup_ratio'],
        # report_to="wandb"
        ) 
        
    
    print("Starting fold 1")
    
    train_data = dataset[0][0]
    val_data = dataset[0][1]
    
    data_collator = DataCollatorWithPadding(train_data.tokenizer)
    
    if(loss == 'mse'):
        trainer1 = CustomTrainerMSE(
        model,
        training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=val_data,    
        tokenizer=train_data.tokenizer,
        compute_metrics=compute_metrics,
        # optimizers = torch.optim.AdamW
        #optimizers=(optimizer, self.lr_scheduler)
        )
    elif(loss == 'ccc'):
        trainer1 = CustomTrainerCCC(
        model,
        training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=val_data,    
        tokenizer=train_data.tokenizer,
        compute_metrics=compute_metrics,
        )
    elif(loss == 'robust'):
        trainer1 = CustomTrainerRobust(
        model,
        training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=val_data,    
        tokenizer=train_data.tokenizer,
        compute_metrics=compute_metrics,
        )
        # for Loss
        # adaptive = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
        #     num_dims=1, float_dtype=np.float32, device=0
        # )
        # params = list(model.parameters()) + list(adaptive.parameters())
        # optimizer = torch.optim.Adam(params, lr=0.01) # No TrainingArguments tenho lr= 2e-5
    elif(loss == 'mse+ccc'): 
        trainer1 = CustomTrainerMSE_CCC(
        model,
        training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=val_data,    
        tokenizer=train_data.tokenizer,
        compute_metrics=compute_metrics,
        )
    elif(loss == 'robust+ccc'): 
        trainer1 = CustomTrainerRobustCCC(
        model,
        training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=val_data,    
        tokenizer=train_data.tokenizer,
        compute_metrics=compute_metrics,
        )
        
    trainer1.train()
    
    # eval
    preds2 = trainer1.predict(val_data)
    
    
    preds_df2 = pd.DataFrame(preds2.predictions)
    run_metrics = preds2.metrics
    
    preds_df2.to_csv(preds_dir +  "/predictions_fold2.csv")      # Write file with predictions on fold2 data
    with open(preds_dir + '/fold2_metrics.csv', 'w') as fa:     # Write run metrics
        for key in run_metrics.keys():
            fa.write("%s,%s\n"%(key,run_metrics[key]))
    fa.close()
    
    
    trainer1.save_model(model_dir)  
    

