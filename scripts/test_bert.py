import os
import subprocess
import numpy as np
from pathlib import Path
import torch
from transformers import AdamW
from data_convert import split_dataset_to_tsv
from data_bert_corpus import BERTCorpus
from model_crf import CRF
from model_bert import BERT
from model_bert_trainer import BERTTrainer
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
    data_path = Path(__file__).parent / '../data'
    max_sequence_length = 64
    batch_size = 64
else:
    # so far, there have been random batch mismatches during training/evaluation that have not been solved yet
    prefix = 'gmb_dataset'
    data_path = Path(__file__).parent / '../data/'
    # if not all([os.path.isfile(data_path.resolve().as_posix()+'/{}/{}.tsv'.format(prefix, f)) for f in ['train', 'dev', 'test']]):
    split_dataset_to_tsv(data_path.resolve().as_posix(), prefix, seed)
    max_sequence_length = 64
    batch_size = 64

# switch for training from scratch or loading previously saved weights from file
new_calculation = True

# corpus class for interacting with data
bert_corpus = BERTCorpus(data_path=data_path.resolve().as_posix(), max_sequence_length=max_sequence_length,
                         batch_size=batch_size, device=device, test=test, prefix=prefix)

# size of vocabs for texts and tags
tag_vocab_size = len(bert_corpus.tag_field.vocab)
tag_names = bert_corpus.tag_field.vocab.itos

# print information about vocabularies
print(m*'-')
print('tag vocabulary built')
print('unique tokens in tag vocabulary: {}'.format(tag_vocab_size))
print('tags: '+(tag_vocab_size*'{} ').format(*tag_names))
print(m*'-')

# print information about datasets
print('train set: {} sentences'.format(len(bert_corpus.train_set)))
print('valid set: {} sentences'.format(len(bert_corpus.valid_set)))
print('test set: {} sentences'.format(len(bert_corpus.valid_set)))
print(m*'-')

tag_pad_idx = bert_corpus.tag_pad_idx
pad_token = bert_corpus.pad_token

try:
    CRF(tag_pad_idx, pad_token, tag_names)
    use_crf = True
    print('using crf for models')
except:
    use_crf = False
    print('not using crf for models (incompatible tagging format)')
print(m*'-')

optimizer_cls = AdamW
full_finetuning = False
max_grad_norm = 1.0

bert = BERT(num_labels=tag_vocab_size, use_crf=use_crf, tag_pad_idx=tag_pad_idx, pad_token=pad_token, tag_names=tag_names)

print('BERTForTokenClassification model initialized with {} trainable parameters'.format(bert.count_parameters()))
print(bert)
print(m*'-')

bert_trainer = BERTTrainer(model=bert, data=bert_corpus, optimizer_cls=optimizer_cls, full_finetuning=full_finetuning, max_grad_norm=max_grad_norm, device=device)

n_epoch = 128
bert_train_path = Path(__file__).parent / '../model/history/{}_{}_hist_train.pt'.format(prefix, 'bert')
bert_valid_path = Path(__file__).parent / '../model/history/{}_{}_hist_valid.pt'.format(prefix, 'bert')
bert_model_path = Path(__file__).parent / '../model/{}_{}_model.pt'.format(prefix, 'bert')

if new_calculation:
    print('training BERTForTokenClassificatio model')
    print(m*'-')
    if os.path.isfile(bert_model_path):
        print('loading model checkpoint')
        bert_trainer.load_model(model_path=bert_model_path)
        bert_trainer.load_history(train_path=bert_train_path, valid_path=bert_valid_path)
    bert_trainer.train(n_epoch=n_epoch)
    bert_trainer.save_model(bert_model_path)
    bert_trainer.save_history(bert_train_path, bert_valid_path)
else:
    bert_trainer.load_model(bert_model_path)
    bert_trainer.load_history(bert_train_path, bert_valid_path)
bert_trainer.test()

valid_bert_history = np.array(bert_trainer.get_history()[1])[:, 2, :]
# initialize figure and axes
fig, ax = plt.subplots()
# remove spines on top and right
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# set axis ticks to left and bottom
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
# plot losses
ax.plot(valid_bert_history.mean(1), color=cm(0.5), label='BERT')
ax.legend(loc='upper left')
ax.set_xlabel('Epoch')
ax.set_ylabel('F1 Score')
# save figure
fig.savefig('bert_f1_history.png')
plt.close()