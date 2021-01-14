import torch
from torch import nn


class NERModel(nn.Module):
    def __init__(self, input_dim, embedding_dim,
                 char_input_dim, char_embedding_dim,
                 char_filter, char_kernel, 
                 hidden_dim, output_dim,
                 embedding_dropout_ratio, cnn_dropout_ratio, fc_dropout_ratio,
                 tag_names, text_pad_idx, text_unk_idx,
                 char_pad_idx, tag_pad_idx, pad_token,
                 pretrained_embeddings):
        '''

        basic class for named entity recognition models. inherits from neural network module.
        layers and forward function will be defined by a child class.

        input_dim: input dimension (size of text vocabulary)
        embedding_dim: embedding dimension (size of word vectors)
        hidden_dim: hidden dimension
        output_dim: output dimension
        embedding_dropout_ratio: dropout for embedding layer
        fc_dropout_ratio: dropout for fully connected layer
        use_crf: switch for using conditional random field (reduces probability of invalid tagging sequences)
        tag_names: the names of all of the tags in the tag field
        text_pad_idx: index for text padding token
        text_unk_idx: indices for text unknown tokens
        tag_pad_idx: index for tag padding token
        pad_token: pad token
        pretrained_embeddings: the pretrained word vectors for the dataset

        '''
        # initialize the superclass
        super().__init__()
        # dimensions
        self.input_dim, self.embedding_dim = input_dim, embedding_dim
        self.char_input_dim, self.char_embedding_dim = char_input_dim, char_embedding_dim
        self.char_filter, self.char_kernel = char_filter, char_kernel
        self.hidden_dim, self.output_dim = hidden_dim, output_dim
        # dropout ratios
        self.embedding_dropout_ratio, self.cnn_dropout_ratio, self.fc_dropout_ratio = embedding_dropout_ratio, cnn_dropout_ratio, fc_dropout_ratio
        # tagging format
        self.tag_names = tag_names
        # indices for padding and unknown tokens
        self.text_pad_idx, self.text_unk_idx, self.char_pad_idx, self.tag_pad_idx = text_pad_idx, text_unk_idx, char_pad_idx, tag_pad_idx
        self.pad_token = pad_token
        # pretrained word embeddings
        self.pretrained_embeddings = pretrained_embeddings
    

    def init_weights(self):
        ''' initializes model weights '''
        for name, param in self.named_parameters():
            nn.init.normal_(param.data, mean=0, std=0.1)
    

    def init_embeddings(self):
        ''' initializes model embeddings  '''
        for idx in (self.text_unk_idx, self.text_pad_idx):
            self.embedding.weight.data[idx] = torch.zeros(self.embedding_dim)
        if self.pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(embeddings=torch.as_tensor(self.pretrained_embeddings), padding_idx=self.text_pad_idx, freeze=True)            


    def count_parameters(self):
        ''' counts model parameters '''
        return sum(p.numel() for p in self.parameters() if p.requires_grad)