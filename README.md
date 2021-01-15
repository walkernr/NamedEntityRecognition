# NamedEntityRecognition

A collection of models used for named entity recognition

## data_convert.py

This is used to split datasets and save them in a .tsv format. Currently supports:

- https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus (extract to data directory with the subdirectory "ner_dataset")

- https://www.kaggle.com/shoumikgoswami/annotated-gmb-corpus/home (extract to data directory in the subdirectory "gmb_dataset")

## data_corpus.py

This contains a class is used to interact with a dataset using the torchtext library. This does use classes that are deprecated and soon to be moved to legacy, including the Field, NestedField, Example, and Iterator classes.

Inputs:

- data_path: root path for dataset directory

- vector_path: path for vector_cache

- glove6b: switch for using glove.6b pretrained embeddings

- embedding_dim: dimension of embedding (50, 100, 200, or 300 for glove.6b)

- min_word_freq: ignore words that don't meet the frequency threshold in the text field

- max_vocab_size: maximum size of the vocabulary of the text field

- batch_size: batch size for data iterators

- device: torch device

- test: switch for whether the dataset is a test (torchtext) set

- prefix: prefix to be appended to data path (subdirectory)

The spacy tokenizer will need to download English language data before usage with the command `python -m spacy download en`. Note that using GloVe embedding vectors will incur a large download to the "vector_path" directory. Additionally, the test sets provided through torchtext can be used by providing the prefixes "udpos" or "conll2000." These will also incur large downloads.

## data_bert_corpus.py

This contains a class for interacting with a dataset using the torchtext library that is made specifically for BERT models.

## model_crf.py

A class for a conditional random field layer.

## model_ner.py

A basic class for the BiLSTM and Transformer models to inherit from.

## model_bilstm.py

The BiLSTM NER model class.

## model_transformer.py

The Transformer NER model class.

## model_bert.py

The BERT NER model class. This will incur a large download.

## model_trainer.py

The classes for training the BiLSTM and Transformer NER models.

## model_bert_trainer.py

A class specifically for training the BERT NER model.

## test.py

An example test run for the BiLSTM and Transformer NER models.

## test_bert.py

An example test run for the BERT NER model.

## plot.py

A script for plotting the losses and metrics for the training and validation of the BiLSTM, Transformer, and BERT NER models using the saved training histories.