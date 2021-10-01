import copy
import re
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import Dataset
from tqdm import tqdm


class RegexTokenizer():
    
    def __init__(self, filters=r'\s+', trunc_num=False):
        self.filters = filters
        self.trunc_num = trunc_num
        
        self.cls_token = "[CLS]"
        self.num_token = "[NUM]"
        self.mask_token = "[MASK]"
        self.unk_token = "[UNK]"
        self.pad_token = "[PAD]"
        
        self.word2index = {self.pad_token: 0, self.cls_token: 1, self.mask_token: 2, self.unk_token: 3, self.num_token: 4}
        self.index2word = {0: self.pad_token, 1: self.cls_token, 2: self.mask_token, 3: self.unk_token, 4: self.num_token}

        self.tokenized = []
        
                    
    def fit(self, sentences):
        for line in tqdm(sentences):
            tokens = self.tokenize(line)
            self.tokenized.append(tokens)

    def tokenize(self, text):
        text = str(text)
        text = text.replace('\'', '')
        regex_split = re.split(self.filters, text)
        tokens = []
        for rs in regex_split:
            if rs and rs != '':
                word = self.addWord(rs)
                tokens.append(self.word2index[word])
        return [self.word2index[self.cls_token]] + tokens

    def addWord(self, word):
        n = 0
        if self.trunc_num:
            if self._num_there(word):
                word = self._trunc_num(word)
        if word not in self.word2index:
            c_words = len(self.word2index)
            self.index2word[c_words] = word
            self.word2index[word] = c_words
        return word
    
    @staticmethod
    def _num_there(word):
        digits = [w.isdigit() for w in word]
        return np.sum(digits) > 0.0
    
    def _trunc_num(self, num_word):
        if num_word.startswith('0x'):
            word = num_word[0:3]
        elif len(num_word) < 2:
            word = num_word[0]
        else:
            word = self.num_token
        return word
    
    def vocab_size(self):
        return len(self.word2index)
    

class TextTokenizationDataset(Dataset):

    def __init__(self, data, pad_len, labels=None, transforms=None, pad=0, weights=None):
        self.c = copy.deepcopy
        self.data = data
        self.transforms = transforms
        self.pad = pad
        self.max_token_len = pad_len
        self.weights = weights
        self.labels = labels

    def _get_padded_data(self, data):
        padded = pad_sequences(data, maxlen=self.max_token_len, dtype="long", truncating="post", padding="post")
        return padded

    def __getitem__(self, index):
        token_ids = self.data[index]
        if self.max_token_len > 0:
            token_ids_padded = self._get_padded_data([token_ids])[0]
        else:
            token_ids_padded = token_ids
        
        if self.labels is not None:
            label = self.labels[index]
        else:
            label = -1

        return token_ids_padded, index, 1, 1, label

    def __len__(self):
        return len(self.data)