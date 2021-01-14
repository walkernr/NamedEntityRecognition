import math
import torch
from torch import nn
from model_ner import NERModel
from model_crf import CRF


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        ''' forward operation for network '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer(NERModel):
    def __init__(self, input_dim, embedding_dim,
                 char_input_dim, char_embedding_dim,
                 char_filter, char_kernel, 
                 hidden_dim, output_dim,
                 trf_layers, attn_heads, use_crf,
                 embedding_dropout_ratio, cnn_dropout_ratio, trf_dropout_ratio,
                 fc_dropout_ratio,
                 tag_names, text_pad_idx, text_unk_idx,
                 char_pad_idx, tag_pad_idx, pad_token,
                 pretrained_embeddings):
        '''

        Transformer model for named entity recognition. inherits from neural network module

        input_dim: input dimension (size of text vocabulary)
        embedding_dim: embedding dimension (size of word vectors)
        char_input_dim: input dimension for characters (size of character vocabulary)
        char_embedding_dim: chatracter embedding dimension
        char_filter: number of filters for the character convolutions
        char_kernel: kernel size for character convolutions
        hidden_dim: hidden dimension
        output_dim: output dimension
        trf_layers: number of trf layers
        attn_heads: number of attention heads for attention component (set to zero or None to ignore)
        use_crf: switch for using conditional random field (reduces probability of invalid tagging sequences)
        embedding_dropout_ratio: dropout for embedding layer
        cnn_dropout_ratio: dropout for convolutions over characters
        trf_dropout_ratio: dropout for trf layers
        fc_dropout_ratio: dropout for fully connected layer
        tag_names: the names of all of the tags in the tag field
        text_pad_idx: index for text padding token
        text_unk_idx: indices for text unknown tokens
        char_pad_idx: indices for character unknown tokens
        tag_pad_idx: index for tag padding token
        pretrained_embeddings: the pretrained word vectors for the dataset

        '''
        # initialize the superclass
        super().__init__(input_dim, embedding_dim,
                         char_input_dim, char_embedding_dim,
                         char_filter, char_kernel, 
                         hidden_dim, output_dim,
                         embedding_dropout_ratio, cnn_dropout_ratio, fc_dropout_ratio,
                         tag_names, text_pad_idx, text_unk_idx,
                         char_pad_idx, tag_pad_idx, pad_token,
                         pretrained_embeddings)
        # network structure settings
        self.trf_layers, self.attn_heads, self.use_crf = trf_layers, attn_heads, use_crf
        # dropout ratios
        self.trf_dropout_ratio = trf_dropout_ratio
        # build model layers
        self.build_model_layers()
        # initialize model weights
        self.init_weights()
        # initialize model embeddings
        self.init_embeddings()


    def build_model_layers(self):
        ''' builds the layers in the model '''
        # embedding layer
        self.embedding = nn.Embedding(num_embeddings=self.input_dim, embedding_dim=self.embedding_dim,
                                      padding_idx=self.text_pad_idx)
        # dropout for embedding layer
        self.embedding_dropout = nn.Dropout(self.embedding_dropout_ratio)
        # character cnn
        if self.char_embedding_dim:
            self.char_embedding = nn.Embedding(num_embeddings=self.char_input_dim,
                                               embedding_dim=self.char_embedding_dim,
                                               padding_idx=self.char_pad_idx)
            self.char_cnn = nn.Conv1d(in_channels=self.char_embedding_dim,
                                      out_channels=self.char_embedding_dim*self.char_filter,
                                      kernel_size=self.char_kernel,
                                      groups=self.char_embedding_dim)
            self.cnn_dropout = nn.Dropout(self.cnn_dropout_ratio)
            # lstm layers with dropout
            all_embedding_dim = self.embedding_dim+(self.char_embedding_dim*self.char_filter)
        else:
            all_embedding_dim = self.embedding_dim
        # transformer encoder layers with attention and dropout
        self.position_encoder = PositionalEncoding(d_model=all_embedding_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=all_embedding_dim, nhead=self.attn_heads, activation='relu', dropout=self.trf_dropout_ratio)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=self.trf_layers)
        # fully connected layer with gelu activation
        self.fc1 = nn.Linear(in_features=all_embedding_dim, out_features=self.hidden_dim)
        self.fc1_gelu = nn.GELU()
        # layer norm
        self.fc1_norm = nn.LayerNorm(self.hidden_dim)
        # dropout for fully connected layer
        self.fc2_dropout = nn.Dropout(self.fc_dropout_ratio)
        # fully connected layer
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        # use crf layer if it is switched on
        if self.use_crf:
            self.crf = CRF(self.tag_pad_idx, self.pad_token, self.tag_names)
    

    def forward(self, sentence, characters, tags):
        ''' forward operation for network '''
        # the output of the embedding layer is dropout(embedding(input))
        embedding_out = self.embedding_dropout(self.embedding(sentence))
        key_padding_mask = torch.as_tensor(sentence == self.text_pad_idx).permute(1, 0)
        if self.char_embedding_dim:
            char_embedding_out = self.embedding_dropout(self.char_embedding(characters))
            batch_size, sentence_len, word_len, char_embedding_dim = char_embedding_out.shape
            char_cnn_max_out = torch.zeros(batch_size, sentence_len, self.char_cnn.out_channels)
            # iterate over sentences
            for sentence_i in range(sentence_len):
                # character field of sentence i
                sentence_char_embedding = char_embedding_out[:, sentence_i, :, :]
                # channels last
                sentence_char_embedding_p = sentence_char_embedding.permute(0, 2, 1)
                char_cnn_sentence_out = self.char_cnn(sentence_char_embedding_p)
                char_cnn_max_out[:, sentence_i, :], _ = torch.max(char_cnn_sentence_out, dim=2)
            char_cnn = self.cnn_dropout(char_cnn_max_out)
            # concatenate word and character embeddings
            word_features = torch.cat((embedding_out, char_cnn), dim=2)
            # positional encoding
            pos_out = self.position_encoder(word_features)
        else:
            # positional encoding
            pos_out = self.position_encoder(embedding_out)
        # encoding
        enc_out = self.encoder(pos_out, src_key_padding_mask=key_padding_mask)
        # fully connected layers
        fc1_out = self.fc1_norm(self.fc1_gelu(self.fc1(enc_out)))
        fc2_out = self.fc2(self.fc2_dropout(fc1_out))
        if self.use_crf:
            crf_out, crf_loss = self.crf(fc2_out, tags)
            return crf_out, crf_loss
        else:
            return fc2_out
