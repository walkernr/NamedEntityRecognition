# import gensim
from collections import Counter
import torch
from torchtext.data import Field, NestedField, BucketIterator
from torchtext.datasets import SequenceTaggingDataset, UDPOS, CoNLL2000Chunking
from torchtext.vocab import Vocab


class Corpus(object):
    def __init__(self, data_path, vector_path, glove6b, embedding_dim,
                 min_word_freq, max_vocab_size, batch_size,
                 device, test, prefix):
        '''

        class for interacting with dataset

        data_path: root path for dataset directory
        vector_path: path for vector_cache
        glove6b: switch for using glove.6b pretrained embeddings
        embedding_dim: dimension of embedding (50, 100, 200, or 300 for glove.6b)
        min_word_freq: ignore words that don't meet the frequency threshold in the text field
        max_vocab_size: maximum size of the vocabulary of the text field
        batch_size: batch size for data iterators
        device: torch device
        test: switch for whether the dataset is a test (torchtext) set that is hopefully more likely to work
        prefix: prefix to be appended to data path
        
        '''
        # set all of the attributes
        self.data_path, self.vector_path, self.glove6b = data_path, vector_path, glove6b
        self.embedding_dim, self.min_word_freq, self.max_vocab_size = embedding_dim, min_word_freq, max_vocab_size
        self.batch_size = batch_size
        self.device, self.test, self.prefix = device, test, prefix
        # initialize text and tag fields
        self.initialize_fields()
        # load dataset
        self.load_data()
        # build vocabularies from text and tag data
        self.build_vocabularies()
        # build iterators for batches of train, dev, and test sets
        self.initialize_iterators()
        # initialize indices of padding and unknown tokens
        self.init_idxs()
    

    def initialize_fields(self):
        ''' initializes fields '''
        # initialize the text field with the spacy tokenizer and no casing
        self.text_field = Field(tokenize='spacy', lower=True, batch_first=True)
        # initialize the tag field without an unknown token (hopefully the train set contains all of the tags)
        self.tag_field = Field(unk_token=None, batch_first=True)
        # initialize the character field
        char_nesting_field = Field(tokenize=list, batch_first=True)
        self.char_field = NestedField(char_nesting_field)
        self.pad_token = self.text_field.pad_token
    

    def load_data(self):
        ''' load data from file using torchtext '''
        if self.test:
            # built-in datasets
            if self.prefix == 'udpos':
                self.train_set, self.valid_set, self.test_set = UDPOS.splits(fields=((('text', 'char'), (self.text_field, self.char_field)),
                                                                                     ('tag', self.tag_field), ('pos', None)),
                                                                             root=self.data_path)
            if self.prefix == 'conll2000':
                self.train_set, self.valid_set, self.test_set = CoNLL2000Chunking.splits(fields=((('text', 'char'), (self.text_field, self.char_field)),
                                                                                                 ('pos', None), ('tag', self.tag_field)),
                                                                                         root=self.data_path)
        else:
            # load datasets from pre-prepared tsv files
            self.train_set, self.valid_set, self.test_set = SequenceTaggingDataset.splits(fields=((('text', 'char'), (self.text_field, self.char_field)),
                                                                                                  ('tag', self.tag_field)),
                                                                                          path=self.data_path+'/{}'.format(self.prefix),
                                                                                          train='train.tsv', validation='dev.tsv', test='test.tsv')
    

    def build_vocabularies(self):
        ''' builds vocabularies for the text and tag data '''
        # if a vector path is provided, then we have to make sure that the word vectors are handled
        if self.vector_path:
            if self.glove6b:
                # the way to do this is built-in with glove.6b
                self.text_field.build_vocab(self.train_set.text, max_size=self.max_vocab_size, min_freq=self.min_word_freq,
                                            vectors='glove.6B.{}d'.format(self.embedding_dim), vectors_cache=self.vector_path,
                                            unk_init=torch.Tensor.normal_)
            else:
                # not sure if this is working
                self.text_field.build_vocab(self.train_set.text, max_size=self.max_vocab_size, min_freq=self.min_word_freq, vectors_cache=self.vector_path)
            ###########################################################################
            # not currently working due to conflict between gensim and python version #
            ###########################################################################
            #     self.wv_model = gensim.models.word2vec.Word2Vec.load(wv_file)
            #     self.embedding_dim = self.wv_model.vector_size
            #     word_freq = {word: self.wv_model.wv.vocab[word].count for word in self.wv_model.wv.vocab}
            #     word_counter = Counter(word_freq)
            #     self.word_field.vocab = Vocab(word_counter, min_freq=min_word_freq)
            #     vectors = []
            #     for word, idx in self.word_field.vocab.stoi.items():
            #         if word in self.wv_model.wv.vocab.keys():
            #             vectors.append(torch.as_tensor(self.wv_model.wv[word].tolist()))
            #         else:
            #             vectors.append(torch.zeros(self.embedding_dim))
            #     self.word_field.vocab.set_vectors(stoi=self.word_field.vocab.stoi, vectors=vectors, dim=self.embedding_dim)
        else:
            # no vectors required 
            self.text_field.build_vocab(self.train_set.text, max_size=self.max_vocab_size, min_freq=self.min_word_freq)
        # build vocabulary for the tags (nothing fancy needed)
        self.char_field.build_vocab(self.train_set.char)
        self.tag_field.build_vocab(self.train_set.tag)
    

    def initialize_iterators(self):
        ''' build iterators for data (by batches) using the bucket iterator '''
        self.train_iter, self.valid_iter, self.test_iter = BucketIterator.splits(datasets=(self.train_set, self.valid_set, self.test_set),
                                                                                 batch_size=self.batch_size, device=self.device, random_state=seed)
    

    def init_idxs(self):
        ''' saves indices for padding and unknown tokens '''
        self.text_pad_idx = self.text_field.vocab.stoi[self.text_field.pad_token]
        self.char_pad_idx = self.char_field.vocab.stoi[self.char_field.pad_token]
        self.tag_pad_idx = self.tag_field.vocab.stoi[self.tag_field.pad_token]
        self.text_unk_idx = self.text_field.vocab.stoi[self.text_field.unk_token]
