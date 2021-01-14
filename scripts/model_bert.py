import torch
from torch import nn
from transformers import BertForTokenClassification
from model_crf import CRF


class BERT(nn.Module):
    def __init__(self, num_labels, use_crf, tag_pad_idx, pad_token, tag_names):
        '''

        bert sequence classifier

        num_labels: number of output classes
        use_crf: switch for using conditional random field (reduces probability of invalid tagging sequences)
        tag_pad_idx: index for tag padding token
        pad_token: pad token
        tag_names: the names of all of the tags in the tag field

        '''
        super().__init__()
        self.num_labels = num_labels
        self.use_crf = use_crf
        self.tag_pad_idx, self.pad_token, self.tag_names = tag_pad_idx, pad_token, tag_names
        self.build_model_layers()
        self.init_weights()



    def build_model_layers(self):
        ''' builds the layers in the model '''
        self.bert = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=self.num_labels, 
                                                               output_attentions=False, output_hidden_states=False)
        if self.use_crf:
            self.crf = CRF(self.tag_pad_idx, self.pad_token, self.tag_names)
    

    def forward(self, sentence, attention_mask, tags):
        ''' forward operation for network '''
        outputs = self.bert(sentence, token_type_ids=None, attention_mask=attention_mask, labels=tags)
        loss, logits = outputs[0], outputs[1]
        if self.use_crf:
            # remove first token id in each sentence (to make crf mask work)
            # crf_out, crf_loss = self.crf(logits, tags)
            crf_out, crf_loss = self.crf(logits[:, 1:], tags[:, 1:])
            return crf_out, crf_loss
        else:
            return logits, loss
    

    def init_weights(self):
        ''' initializes model weights '''
        # param_initializer = list(self.bert.classifier.named_parameters())
        # if self.crf:
        #     param_initializer += list(self.crf.named_parameters())
        # for name, param in param_initializer:
        #     nn.init.normal_(param.data, mean=0, std=0.1)
        
        # only initialize conditional random field weights
        if self.crf:
            for name, param in self.crf.named_parameters():
                nn.init.normal_(param.data, mean=0, std=0.1)
        

    def count_parameters(self):
        ''' counts model parameters '''
        return sum(p.numel() for p in self.parameters() if p.requires_grad)