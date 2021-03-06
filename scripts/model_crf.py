import torch
from torch import nn
import torchcrf

class CRF(nn.Module):
    def __init__(self, tag_pad_idx, pad_token, tag_names):
        super().__init__()
        # tag pad index and tag names
        self.tag_pad_idx = tag_pad_idx
        self.pad_token = pad_token
        self.tag_names = tag_names
        # initialize CRF
        self.crf = torchcrf.CRF(num_tags=len(self.tag_names), batch_first=True)
        # construct definitions of invalid transitions
        self.define_invalid_crf_transitions()
        # initialize transitions
        self.init_crf_transitions()
    

    def define_invalid_crf_transitions(self):
        ''' function for establishing valid tagging transitions, assumes BIO or BILUO tagging '''
        self.prefixes = set([tag_name[0] for tag_name in self.tag_names if tag_name != self.pad_token])
        if self.prefixes == set(['B', 'I', 'O']):
            # (B)eginning (I)nside (O)utside
            # cannot begin sentence with I (inside), only B (beginning) or O (outside)
            self.invalid_begin = ('I',)
            # cannot end sentence with B (beginning) or I (inside) - assumes data ends with O (outside), such as punctuation
            self.invalid_end = ('B', 'I')
            # prevent B (beginning) going to P - B must be followed by B, I, or O
            # prevent I (inside) going to P - I must be followed by B, I, or O
            # prevent O (outside) going to I (inside) - O must be followed by B or O
            self.invalid_transitions_position = {'B': 'P',
                                                 'I': 'P',
                                                 'O': 'I'}
            # prevent B (beginning) going to I (inside) of a different type
            # prevent I (inside) going to I (inside) of a different type
            self.invalid_transitions_tags = {'B': 'I',
                                             'I': 'I'}
        if self.prefixes == set(['B', 'I', 'L', 'U', 'O']):
            # (B)eginning (I)nside (L)ast (U)nit (O)utside
            # cannot begin sentence with I (inside) or L (last)
            self.invalid_begin = ('I', 'L')
            # cannot end sentence with B (beginning) or I (inside)
            self.invalid_end = ('B', 'I')
            # prevent B (beginning) going to B (beginning), O (outside), U (unit), or P - B must be followed by I or L
            # prevent I (inside) going to B (beginning), O (outside), U (unit), or P - I must be followed by I or L
            # prevent L (last) going to I (inside) or L(last) - U must be followed by B, O, U, or P
            # prevent U (unit) going to I (inside) or L(last) - U must be followed by B, O, U, or P
            # prevent O (outside) going to I (inside) or L (last) - O must be followed by B, O, U, or P
            self.invalid_transitions_position = {'B': 'BOUP',
                                                 'I': 'BOUP',
                                                 'L': 'IL',
                                                 'U': 'IL',
                                                 'O': 'IL'}
            # prevent B (beginning) from going to I (inside) or L (last) of a different type
            # prevent I (inside) from going to I (inside) or L (last) of a different tpye
            self.invalid_transitions_tags = {'B': 'IL',
                                             'I': 'IL'}
        if self.prefixes == set(['B', 'I', 'E', 'S', 'O']):
            # (B)eginning (I)nside (E)nd (S)ingle (O)utside
            # cannot begin sentence with I (inside) or E (end)
            self.invalid_begin = ('I', 'E')
            # cannot end sentence with B (beginning) or I (inside)
            self.invalid_end = ('B', 'I')
            # prevent B (beginning) going to B (beginning), O (outside), S (single), or P - B must be followed by I or E
            # prevent I (inside) going to B (beginning), O (outside), S (single), or P - I must be followed by I or E
            # prevent E (end) going to I (inside) or E (end) - U must be followed by B, O, U, or P
            # prevent S (single) going to I (inside) or E (end) - U must be followed by B, O, U, or P
            # prevent O (outside) going to I (inside) or E (end) - O must be followed by B, O, U, or P
            self.invalid_transitions_position = {'B': 'BOSP',
                                                 'I': 'BOSP',
                                                 'E': 'IE',
                                                 'S': 'IE',
                                                 'O': 'IE'}
            # prevent B (beginning) from going to I (inside) or E (end) of a different type
            # prevent I (inside) from going to I (inside) or E (end) of a different tpye
            self.invalid_transitions_tags = {'B': 'IE',
                                             'I': 'IE'}
    

    def init_crf_transitions(self, imp_value=-100):
        num_tags = len(self.tag_names)
        # penalize bad beginnings and endings
        for i in range(num_tags):
            tag_name = self.tag_names[i]
            if tag_name[0] in self.invalid_begin or tag_name == self.pad_token:
                torch.nn.init.constant_(self.crf.start_transitions[i], imp_value)
            if tag_name[0] in self.invalid_end:
                torch.nn.init.constant_(self.crf.end_transitions[i], imp_value)
        # build tag type dictionary
        tag_is = {}
        for tag_position in self.prefixes:
            tag_is[tag_position] = [i for i, tag in enumerate(self.tag_names) if tag[0] == tag_position]
        tag_is['P'] = [i for i, tag in enumerate(self.tag_names) if tag == 'tag']
        # penalties for invalid consecutive tags by position
        for from_tag, to_tag_list in self.invalid_transitions_position.items():
            to_tags = list(to_tag_list)
            for from_tag_i in tag_is[from_tag]:
                for to_tag in to_tags:
                    for to_tag_i in tag_is[to_tag]:
                        torch.nn.init.constant_(self.crf.transitions[from_tag_i, to_tag_i], imp_value)
        # penalties for invalid consecutive tags by tag
        for from_tag, to_tag_list in self.invalid_transitions_tags.items():
            to_tags = list(to_tag_list)
            for from_tag_i in tag_is[from_tag]:
                for to_tag in to_tags:
                    for to_tag_i in tag_is[to_tag]:
                        if self.tag_names[from_tag_i].split('-')[1] != self.tag_names[to_tag_i].split('-')[1]:
                            torch.nn.init.constant_(self.crf.transitions[from_tag_i, to_tag_i], imp_value)        
    

    def forward(self, fc_out, tags):
        # mask ignores pad index
        mask = tags != self.tag_pad_idx
        # compute output and loss
        crf_out = self.crf.decode(fc_out, mask=mask)
        crf_loss = -self.crf(fc_out, tags=tags, mask=mask)
        return crf_out, crf_loss