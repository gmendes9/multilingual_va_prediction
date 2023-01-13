import os
import pandas as pd
import numpy as np
from datasets import load_dataset, dataset_dict, DatasetDict
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer, RobertaTokenizer, AutoTokenizer, XLMRobertaTokenizer, RobertaTokenizer
import torch
import csv

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MyDataset(Dataset):
    def __init__(self, filename, checkpoint, maxlen):
        if(checkpoint == 'distilbert-base-multilingual-cased'):
            self.tokenizer = DistilBertTokenizer.from_pretrained(checkpoint)
        elif(checkpoint == 'xlm-roberta-large'): 
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(checkpoint)
        elif(checkpoint == 'xlm-roberta-base'): 
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(checkpoint)

        df = pd.read_csv(filename,sep='\t',
                quotechar='"',
                engine='python', 
                quoting=csv.QUOTE_NONE,
                escapechar='\\',
                keep_default_na=False,
                dtype={'index':np.int32,'text':str,'valence':np.float64, 'arousal':np.float64})
    
        self.index = df['index'].to_list()
        self.texts = df['text'].to_list()
        self.valence = df['valence'].to_list()
        self.arousal = df['arousal'].to_list()
        self.maxlen = maxlen

    def __getitem__(self, idx):
        item = { }
        aux = self.tokenizer(self.texts[idx], max_length=self.maxlen, truncation=True, padding=False)
        item['input_ids'] = torch.tensor(aux['input_ids'])
        item['attention_mask'] = torch.tensor(aux['attention_mask'])
        item['labels'] = torch.tensor( [ self.valence[idx], self.arousal[idx] ] )

        return item

    def __len__(self):
        return len(self.texts)
    


        

    