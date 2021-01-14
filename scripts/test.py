import os
import subprocess
import numpy as np
from pathlib import Path
import torch
from torch import nn
from torch.optim import Adam
from data_convert import split_dataset_to_tsv
from data_corpus import Corpus
from model_crf import CRF
from model_bilstm import BiLSTM
from model_transformer import Transformer
from model_trainer import NERTrainer
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rc('font', family='sans-serif')
ftsz = 28
figw = 16
pparam = {'figure.figsize': (figw, figw),
          'lines.linewidth': 4.0,
          'legend.fontsize': ftsz,
          'axes.labelsize': ftsz,
          'axes.titlesize': ftsz,
          'axes.linewidth': 2.0,
          'xtick.labelsize': ftsz,
          'xtick.major.size': 20,
          'xtick.major.width': 2.0,
          'ytick.labelsize': ftsz,
          'ytick.major.size': 20,
          'ytick.major.width': 2.0,
          'font.size': ftsz}
plt.rcParams.update(pparam)
cm = plt.get_cmap('plasma')

# clear console
# os.system('clear')
# console dimensions
n, m = subprocess.check_output(['stty', 'size']).decode().split()
n, m = int(n), int(m)

# run in cpu mode if gpu not available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# deterministic run
seed = 256
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# parameters for corpus
embedding_dim = 300
min_word_freq = 3
max_vocab_size = 25_000
# dataset selection
test = False
if test:
    # udpos or conll2000
    prefix = 'conll2000'
    glove6b = True
    data_path = Path(__file__).parent / '../data'
    vector_path = Path(__file__).parent / '../model/vector_cache'
    batch_size = 64
else:
    # so far, there have been random batch mismatches during training/evaluation that have not been solved yet
    prefix = 'gmb_dataset'
    data_path = Path(__file__).parent / '../data/'
    glove6b = True
    # if not all([os.path.isfile(data_path.resolve().as_posix()+'/{}/{}.tsv'.format(prefix, f)) for f in ['train', 'dev', 'test']]):
    split_dataset_to_tsv(data_path.resolve().as_posix(), prefix, seed)
    vector_path = Path(__file__).parent / '../model/vector_cache'
    batch_size = 64

# switch for training from scratch or loading previously saved weights from file
new_calculation = True

# corpus class for interacting with data
corpus = Corpus(data_path=data_path.resolve().as_posix(), vector_path=vector_path.resolve().as_posix(), glove6b=glove6b,
                embedding_dim=embedding_dim, min_word_freq=min_word_freq, max_vocab_size=max_vocab_size,
                batch_size=batch_size, device=device, test=test, prefix=prefix)

# size of vocabs for texts and tags
text_vocab_size = len(corpus.text_field.vocab)
char_vocab_size = len(corpus.char_field.vocab)
tag_vocab_size = len(corpus.tag_field.vocab)
tag_names = corpus.tag_field.vocab.itos

# print information about vocabularies
print(m*'-')
print('vocabularies built')
print('unique tokens in text vocabulary: {}'.format(text_vocab_size))
print('unique tokens in char vocabulary: {}'.format(char_vocab_size))
print('unique tokens in tag vocabulary: {}'.format(tag_vocab_size))
print('10 most frequent words in text vocabulary: '+(10*'{} ').format(*corpus.text_field.vocab.freqs.most_common(10)))
print('tags: '+(tag_vocab_size*'{} ').format(*tag_names))
print(m*'-')

# print information about datasets
print('train set: {} sentences'.format(len(corpus.train_set)))
print('valid set: {} sentences'.format(len(corpus.valid_set)))
print('test set: {} sentences'.format(len(corpus.valid_set)))
print(m*'-')

# parameters from corpus
text_pad_idx = corpus.text_pad_idx
text_unk_idx = corpus.text_unk_idx
char_pad_idx = corpus.char_pad_idx
tag_pad_idx = corpus.tag_pad_idx
pad_token = corpus.pad_token
pretrained_embeddings = corpus.text_field.vocab.vectors

try:
    CRF(tag_pad_idx, pad_token, tag_names)
    use_crf = True
    print('using crf for models')
except:
    use_crf = False
    print('not using crf for models (incompatible tagging format)')
print(m*'-')

# shared cnn parameters
char_embedding_dim = 37
char_filter = 4
char_kernel = 3
# shared dropouts
embedding_dropout_ratio = 0.5
char_embedding_dropout_ratio = 0.25
cnn_dropout_ratio = 0.25
fc_dropout_ratio = 0.25
# shared attention parameters
attn_heads = 16
attn_dropout_ratio = 0.25

# parameters for bilstm model
# parameters for char cnn

# lstm parameters
hidden_dim = 64
lstm_layers = 2
lstm_dropout_ratio = 0.1

# initialize bilstm
bilstm = BiLSTM(input_dim=text_vocab_size, embedding_dim=embedding_dim,
                char_input_dim=char_vocab_size, char_embedding_dim=char_embedding_dim,
                char_filter=char_filter, char_kernel=char_kernel,
                hidden_dim=hidden_dim, output_dim=tag_vocab_size,
                lstm_layers=lstm_layers, attn_heads=attn_heads, use_crf=use_crf,
                embedding_dropout_ratio=embedding_dropout_ratio, cnn_dropout_ratio=cnn_dropout_ratio, lstm_dropout_ratio=lstm_dropout_ratio,
                attn_dropout_ratio=attn_dropout_ratio, fc_dropout_ratio=fc_dropout_ratio,
                tag_names=tag_names, text_pad_idx=text_pad_idx, text_unk_idx=text_unk_idx,
                char_pad_idx=char_pad_idx, tag_pad_idx=tag_pad_idx, pad_token=pad_token,
                pretrained_embeddings=pretrained_embeddings)
# print bilstm information
print('BiLSTM model initialized with {} trainable parameters'.format(bilstm.count_parameters()))
print(bilstm)
print(m*'-')

# parameters for transformer model
hidden_dim = 256
trf_layers = 1

# initialize transformer
transformer = Transformer(input_dim=text_vocab_size, embedding_dim=embedding_dim,
                          char_input_dim=char_vocab_size, char_embedding_dim=char_embedding_dim,
                          char_filter=char_filter, char_kernel=char_kernel,
                          hidden_dim=hidden_dim, output_dim=tag_vocab_size,
                          trf_layers=trf_layers, attn_heads=attn_heads, use_crf=use_crf,
                          embedding_dropout_ratio=embedding_dropout_ratio, cnn_dropout_ratio=cnn_dropout_ratio,
                          trf_dropout_ratio=attn_dropout_ratio, fc_dropout_ratio=fc_dropout_ratio,
                          tag_names=tag_names, text_pad_idx=text_pad_idx, text_unk_idx=text_unk_idx,
                          char_pad_idx=char_pad_idx, tag_pad_idx=tag_pad_idx, pad_token=pad_token,
                          pretrained_embeddings=pretrained_embeddings)
# print transformer information
print('Transformer model initialized with {} trainable parameters'.format(transformer.count_parameters()))
print(transformer)
print(m*'-')

# initialize trainer class for bilstm
bilstm_trainer = NERTrainer(model=bilstm, data=corpus, optimizer_cls=Adam, criterion_cls=nn.CrossEntropyLoss, device=device)
# initialize trainer class for transformer
transformer_trainer = NERTrainer(model=transformer, data=corpus, optimizer_cls=Adam, criterion_cls=nn.CrossEntropyLoss, device=device)

# parameters for trainer
n_epoch = 256

# bilstm paths
bilstm_train_path = Path(__file__).parent / '../model/history/{}_{}_hist_train.pt'.format(prefix, 'bilstm')
bilstm_valid_path = Path(__file__).parent / '../model/history/{}_{}_hist_valid.pt'.format(prefix, 'bilstm')
bilstm_model_path = Path(__file__).parent / '../model/{}_{}_model.pt'.format(prefix, 'bilstm')

# transformer paths
transformer_train_path = Path(__file__).parent / '../model/history/{}_{}_hist_train.pt'.format(prefix, 'transformer')
transformer_valid_path = Path(__file__).parent / '../model/history/{}_{}_hist_valid.pt'.format(prefix, 'transformer')
transformer_model_path = Path(__file__).parent / '../model/{}_{}_model.pt'.format(prefix, 'transformer')

# if new_calculation, then train and save the model. otherwise, just load everything from file
if new_calculation:
    print('training BiLSTM model')
    print(m*'-')
    if os.path.isfile(bilstm_model_path):
        print('loading model checkpoint')
        bilstm_trainer.load_model(model_path=bilstm_model_path)
        bilstm_trainer.load_history(train_path=bilstm_train_path, valid_path=bilstm_valid_path)
    bilstm_trainer.train(n_epoch=n_epoch)
    bilstm_trainer.save_model(model_path=bilstm_model_path)
    bilstm_trainer.save_history(train_path=bilstm_train_path, valid_path=bilstm_valid_path)
    print(m*'-')
    print('training Transformer model')
    print(m*'-')
    if os.path.isfile(transformer_model_path):
        print('loading model checkpoint')
        transformer_trainer.load_model(model_path=transformer_model_path)
        transformer_trainer.load_history(train_path=transformer_train_path, valid_path=transformer_valid_path)
    transformer_trainer.train(n_epoch=n_epoch)
    transformer_trainer.save_model(model_path=transformer_model_path)
    transformer_trainer.save_history(train_path=transformer_train_path, valid_path=transformer_valid_path)
    print(m*'-')
else:
    print('loading BiLSTM model')
    print(m*'-')
    bilstm_trainer.load_model(model_path=bilstm_model_path)
    bilstm_trainer.load_history(train_path=bilstm_train_path, valid_path=bilstm_valid_path)
    print('loading Transformer model')
    print(m*'-')
    transformer_trainer.load_model(model_path=transformer_model_path)
    transformer_trainer.load_history(train_path=transformer_train_path, valid_path=transformer_valid_path)
    print(m*'-')

# evaluate test set
print('testing BiLSTM')
bilstm_trainer.test()
print(m*'-')
print('testing Transformer')
transformer_trainer.test()
print(m*'-')

valid_bilstm_history = np.array(bilstm_trainer.get_history()[1])[:, 2, :]
valid_transformer_history = np.array(transformer_trainer.get_history()[1])[:, 2, :]
# initialize figure and axes
fig, ax = plt.subplots()
# remove spines on top and right
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# set axis ticks to left and bottom
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
# plot losses
ax.plot(valid_bilstm_history.mean(1), color=cm(0.3), label='BiLSTM')
ax.plot(valid_transformer_history.mean(1), color=cm(0.7), label='Transformer')
ax.legend(loc='upper left')
ax.set_xlabel('Epoch')
ax.set_ylabel('F1 Score')
# save figure
fig.savefig('bilstm_transformer_f1_history.png')
plt.close()