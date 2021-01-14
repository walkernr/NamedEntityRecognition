import torch
from torch import nn
from model_ner import NERModel
from model_crf import CRF


class BiLSTM(NERModel):
    def __init__(self, input_dim, embedding_dim,
                 char_input_dim, char_embedding_dim,
                 char_filter, char_kernel, 
                 hidden_dim, output_dim,
                 lstm_layers, attn_heads, use_crf,
                 embedding_dropout_ratio, cnn_dropout_ratio, lstm_dropout_ratio,
                 attn_dropout_ratio, fc_dropout_ratio,
                 tag_names, text_pad_idx, text_unk_idx,
                 char_pad_idx, tag_pad_idx, pad_token,
                 pretrained_embeddings):
        '''

        BiLSTM model for named entity recognition. inherits from named recognition model

        input_dim: input dimension (size of text vocabulary)
        embedding_dim: embedding dimension (size of word vectors)
        char_input_dim: input dimension for characters (size of character vocabulary)
        char_embedding_dim: chatracter embedding dimension
        char_filter: number of filters for the character convolutions
        char_kernel: kernel size for character convolutions
        hidden_dim: hidden dimension
        output_dim: output dimension
        lstm_layers: number of lstm layers
        attn_heads: number of attention heads for attention component (set to zero or None to ignore)
        use_crf: switch for using conditional random field (reduces probability of invalid tagging sequences)
        embedding_dropout_ratio: dropout for embedding layer
        cnn_dropout_ratio: dropout for convolutions over characters
        lstm_dropout_ratio: dropout for lstm layers
        attn_dropout_ratio: dropout for attention layer
        fc_dropout_ratio: dropout for fully connected layer
        tag_names: the names of all of the tags in the tag field
        text_pad_idx: index for text padding token
        text_unk_idx: indices for text unknown tokens
        char_pad_idx: indices for character unknown tokens
        tag_pad_idx: index for tag padding token
        pad_token: pad_token
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
        self.lstm_layers, self.attn_heads, self.use_crf = lstm_layers, attn_heads, use_crf
        # dropout ratios
        self.lstm_dropout_ratio, self.attn_dropout_ratio = lstm_dropout_ratio, attn_dropout_ratio
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
            all_embedding_dim = self.embedding_dim+(self.char_embedding_dim*self.char_filter)
            # lstm layers with dropout
        else:
            all_embedding_dim = self.embedding_dim
        # lstm layers with dropout
        self.lstm = nn.LSTM(batch_first=True, input_size=all_embedding_dim,
                            hidden_size=self.hidden_dim, num_layers=self.lstm_layers,
                            bidirectional=True, dropout=self.lstm_dropout_ratio if self.lstm_layers > 1 else 0)
        # use multihead attention if there are attention heads
        if self.attn_heads:
            self.attn = nn.MultiheadAttention(embed_dim=self.hidden_dim*2, num_heads=self.attn_heads, dropout=self.attn_dropout_ratio)
        # dropout for fully connected layer
        self.fc_dropout = nn.Dropout(self.fc_dropout_ratio)
        # fully connected layer
        self.fc = nn.Linear(self.hidden_dim*2, self.output_dim)
        # use crf layer if it is switched on
        if self.use_crf:
            self.crf = CRF(self.tag_pad_idx, self.pad_token, self.tag_names)            
    

    def forward(self, sentence, characters, tags):
        ''' forward operation for network '''
        # the output of the embedding layer is dropout(embedding(input))
        embedding_out = self.embedding_dropout(self.embedding(sentence))
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
            # lstm of embedding output
            lstm_out, _ = self.lstm(word_features)
        else:
            # lstm of embedding output
            lstm_out, _ = self.lstm(embedding_out)
        # attention layer
        if self.attn_heads:
            # masking using the text padding index
            key_padding_mask = torch.as_tensor(sentence == self.text_pad_idx).permute(1, 0)
            # attention outputs
            attn_out, attn_weight = self.attn(lstm_out, lstm_out, lstm_out, key_padding_mask=key_padding_mask)
            # fully connected layer as function of attention output
            fc_out = self.fc(self.fc_dropout(attn_out))
        else:
            # fully connected layer as function of lstm output
            fc_out = self.fc(self.fc_dropout(lstm_out))
        if self.use_crf:
            crf_out, crf_loss = self.crf(fc_out, tags)
            return crf_out, crf_loss
        else:
            return fc_out