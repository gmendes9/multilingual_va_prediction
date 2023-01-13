from transformers import DistilBertForSequenceClassification, XLMRobertaForSequenceClassification
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification
from transformers.models.xlm_roberta.configuration_xlm_roberta import XLMRobertaConfig
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Set, Tuple, Union
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    
    
)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


class DistilBertForSequenceClassificationSig(DistilBertForSequenceClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.sigmoid = torch.sigmoid
        # self.sigmoid = torch.nn.functional.hardsigmoid # HardSigmoid

        # self.sigmoid = self.customSigmoid # CUSTOM SIGMOID
        self.sigmoid = self.customHardSigmoid # CUSTOM HARD SIGMOID
        self.threshold = torch.nn.Threshold(-1,-1) 

        
    def customSigmoid(self, x):
        return torch.sigmoid(5*x)  
    def customHardSigmoid(self, x):
        return torch.nn.functional.hardsigmoid(3*x)
  
            
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
        
        # ret = super.forward(input_ids, attention_mask, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
        ret = super(DistilBertForSequenceClassificationSig, self).forward(input_ids, attention_mask, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
        ret.logits = self.sigmoid(ret.logits) # Uncomment to use any sigmoid
        
        # ret.logits = torch.relu(ret.logits) # Uncomment to use ReLu w threshold
        # ret.logits = -self.threshold(-ret.logits) # Uncomment to use ReLu w threshold
        

        return ret
        
    
        
class RobertaForSequenceClassificationSig(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        # self.sigmoid = torch.sigmoid
        self.sigmoid = torch.nn.functional.hardsigmoid # HardSigmoid
        self.threshold = torch.nn.Threshold(-1,-1)

    def customHardSigmoid(self, x):
        return torch.nn.functional.hardsigmoid(3*x)
        
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        
        # ret = super.forward(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
        ret = super(RobertaForSequenceClassificationSig, self).forward(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
        ret.logits = self.sigmoid(ret.logits) # Uncomment to use any sigmoid
        
        # ret.logits = torch.relu(ret.logits) # Uncomment to use ReLu w threshold
        # ret.logits = -self.threshold(-ret.logits) # Uncomment to use ReLu w threshold
        

        
        return ret
    
        
        
class XLMRobertaForSequenceClassificationSig(RobertaForSequenceClassificationSig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    """
    This class overrides [`RobertaForSequenceClassification`]. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig