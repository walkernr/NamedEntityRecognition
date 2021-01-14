from transformers import BertTokenizer
from torchtext.data import Field, BucketIterator, Iterator
from torchtext.datasets import SequenceTaggingDataset, UDPOS, CoNLL2000Chunking


class BERTCorpus(object):
    def __init__(self, data_path, max_sequence_length, batch_size, device, test, prefix):
        self.data_path = data_path
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self.device = device
        self.test = test
        self.prefix = prefix
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        self.pad_token = self.tokenizer.pad_token
        self.unk_token = self.tokenizer.unk_token
        self.initialize_fields()
        self.load_data()
        self.build_tag_vocabulary()
        self.initialize_iterators()


    def insert_initial_pad_tag(self, tag):
        tag.insert(0, self.pad_token)
        return tag
    

    def initialize_fields(self):
        self.text_field = Field(use_vocab=False, preprocessing=self.tokenizer.encode, lower=False,
                                include_lengths=False, batch_first=True, fix_length=self.max_sequence_length,
                                pad_token=self.tokenizer.convert_tokens_to_ids(self.pad_token),
                                unk_token=self.tokenizer.convert_tokens_to_ids(self.unk_token))
        self.tag_field = Field(batch_first=True, fix_length=self.max_sequence_length,
                               preprocessing=self.insert_initial_pad_tag, pad_token=self.pad_token, unk_token=None)
    

    def load_data(self):
        ''' load data from file using torchtext '''
        if self.test:
            # built-in datasets
            if self.prefix == 'udpos':
                self.train_set, self.valid_set, self.test_set = UDPOS.splits(fields=(('text', self.text_field), ('tag', self.tag_field), ('pos', None)),
                                                                             root=self.data_path)
            if self.prefix == 'conll2000':
                self.train_set, self.valid_set, self.test_set = CoNLL2000Chunking.splits(fields=(('text', self.text_field), ('pos', None), ('tag', self.tag_field)),
                                                                                         root=self.data_path)
        else:
            # load datasets from pre-prepared tsv files
            self.train_set, self.valid_set, self.test_set = SequenceTaggingDataset.splits(fields=(('text', self.text_field), ('tag', self.tag_field)),
                                                                                          path=self.data_path+'/{}'.format(self.prefix),
                                                                                          train='train.tsv', validation='dev.tsv', test='test.tsv')
    

    def build_tag_vocabulary(self):
        self.tag_field.build_vocab(self.train_set.tag)
        self.tag_pad_idx = self.tag_field.vocab.stoi[self.tag_field.pad_token]


    def initialize_iterators(self):
        ''' build iterators for data (by batches) using the bucket iterator '''
        self.train_iter, self.valid_iter, self.test_iter = BucketIterator.splits(datasets=(self.train_set, self.valid_set, self.test_set),
                                                                                 batch_size=self.batch_size, device=self.device)
