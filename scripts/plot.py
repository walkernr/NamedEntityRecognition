import numpy as np
from pathlib import Path
import torch
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

prefix = 'gmb_dataset'

bilstm_train_path = Path(__file__).parent / '../model/history/{}_{}_hist_train.pt'.format(prefix, 'bilstm')
bilstm_valid_path = Path(__file__).parent / '../model/history/{}_{}_hist_valid.pt'.format(prefix, 'bilstm')

transformer_train_path = Path(__file__).parent / '../model/history/{}_{}_hist_train.pt'.format(prefix, 'transformer')
transformer_valid_path = Path(__file__).parent / '../model/history/{}_{}_hist_valid.pt'.format(prefix, 'transformer')

bert_train_path = Path(__file__).parent / '../model/history/{}_{}_hist_train.pt'.format(prefix, 'bert')
bert_valid_path = Path(__file__).parent / '../model/history/{}_{}_hist_valid.pt'.format(prefix, 'bert')

valid = {'BiLSTM': (0.3, np.array(torch.load(bilstm_train_path)), np.array(torch.load(bilstm_valid_path))),
         'Transformer': (0.5, np.array(torch.load(transformer_train_path)), np.array(torch.load(transformer_valid_path))),
         'BERT': (0.7, np.array(torch.load(bert_train_path)), np.array(torch.load(bert_valid_path)))}
quant = ['Loss', 'Accuracy Score', 'F1 Score']
phase = ['Training', 'Validation']
# initialize figure and axes
fig, axs = plt.subplots(3, 2)
# plot losses
for i in range(axs.shape[0]):
    for j in range(axs.shape[1]):
        ax = axs[i, j]
        # remove spines on top and right
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # set axis ticks to left and bottom
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        for key, val in valid.items():
            ax.plot(val[j+1][:, i, :].mean(1), color=cm(val[0]), label=key)
        if 'Loss' in quant[i]:
            loc = 'upper right'
        else:
            loc = 'lower right'
        ax.legend(loc=loc, fontsize=ftsz/2)
        if i == 2:
            ax.set_xlabel('Epoch')
        if j == 0:
            ax.set_ylabel(quant[i])
            if i == 0:
                ax.set_title(phase[j])
        if j == 1 and i == 0:
            ax.set_title(phase[j])
# save figure
fig.savefig('history.png')
plt.close()