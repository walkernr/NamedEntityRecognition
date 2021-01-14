import csv
import random
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset_to_tsv(data_path, prefix, seed):
    ''' splits data to tsv files '''
    if prefix == 'ner_dataset':
        # annotated corpus dataset - https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus
        # read data into pandas dataframe
        data = pd.read_csv(data_path+'/{}/ner_dataset.csv'.format(prefix), encoding='latin1', dtype=str)
        # drop position column
        data = data.drop('POS', 1)
        # fill nan in 'sentence #' column by feeding the non-nan values forward
        data = data.fillna(method='ffill')
    if prefix == 'gmb_dataset':
        # annotated gmb corpus - https://www.kaggle.com/shoumikgoswami/annotated-gmb-corpus/home
        # read data into pandas dataframe
        data = pd.read_csv(data_path+'/{}/GMB_dataset.txt'.format(prefix), sep='\t', header=None, encoding='latin1', dtype=str)
        # fix header (unlabelled column)
        data.columns = data.iloc[0]
        data = data[1:]
        data.columns = ['Index', 'Sentence #', 'Word', 'POS', 'Tag']
        # drop index
        data = data.reset_index(drop=True)
    # check for any gaps in data
    data.info()

    # collects the (text, tag) values into a list
    agg_func = lambda s:[(w, t) for w, t in zip(s["Word"].values.tolist(), s["Tag"].values.tolist())]
    # group the data by sentence and collect the (text, tag) values into a list
    grouped = data.groupby("Sentence #").apply(agg_func)
    
    # construct list from generator
    sentences = [s for s in grouped]
    # separate text and tag data in list
    text = [[s[0] for s in sent] for sent in sentences]
    tag = [[s[1] for s in sent] for sent in sentences]

    # split the data into train, dev (validation), and test sets
    train_text, test_text, train_tag, test_tag = train_test_split(text, tag, test_size=0.5, random_state=seed)
    dev_text, test_text, dev_tag, test_tag = train_test_split(test_text, test_tag, test_size=0.5, random_state=seed)
    
    # write train data to file in tsv format (same directory)
    with open(data_path+'/{}/train.tsv'.format(prefix), 'w') as f:
        for txt, tg in zip(train_text, train_tag):
            for w, t in zip(txt, tg):
                f.write('{}\t{}\n'.format(w, t))
            f.write('\n')
    # write dev data to file in tsv format (same directory)
    with open(data_path+'/{}/dev.tsv'.format(prefix), 'w') as f:
        for txt, tg in zip(dev_text, dev_tag):
            for w, t in zip(txt, tg):
                f.write('{}\t{}\n'.format(w, t))
            f.write('\n')
    # write test data to file in tsv format (same directory)
    with open(data_path+'/{}/test.tsv'.format(prefix), 'w') as f:
        for txt, tg in zip(test_text, test_tag):
            for w, t in zip(txt, tg):
                f.write('{}\t{}\n'.format(w, t))
            f.write('\n')
