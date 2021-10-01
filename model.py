import pandas as pd
import numpy as np
import os
import re
import pandas as pd
from itertools import islice
import collections
import time
from tqdm import tqdm
from torchvision import transforms
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import DataLoader, SequentialSampler, WeightedRandomSampler, Dataset
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import copy as c
import copy
from torch.autograd import Variable
from sklearn import metrics



MODEL_LOCATION = os.path.join(os.getcwd(), 'trained_models')
MODEL_NAME = '{}.pt'

class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg
            self.trg_y = trg
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    def _subsequent_mask(self, size):
        """Mask out subsequent positions."""
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0

    def make_std_mask(self, tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(self._subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask



def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class Generator(nn.Module):

    def __init__(self, d_model, vocab, centroid):
        super(Generator, self).__init__()
        self.d_model = d_model
        self.proj = nn.Linear(self.d_model, vocab)
        self.centroid = centroid

    def forward(self, x):
        out = self.proj(x[:, 0, :])
        return out


class TransformerModel(nn.Module):

    def __init__(self, encoder, src_embed, tgt_embed, generator):
        super(TransformerModel, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.centroid = generator.centroid
        self.threshold = 0

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if attention_mask is not None:
            if len(attention_mask.shape) == 2:
                attention_mask = torch.unsqueeze(attention_mask, dim=1)
        if input_ids.shape[1] % 2 != 0:
            input_ids = input_ids[:, :(input_ids.shape[1] - 1)]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :, :(attention_mask.shape[2] - 1)]

        out = self.encode(input_ids, attention_mask)
        return out

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)


class Encoder(nn.Module):

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


class EncoderLayer(nn.Module):

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)




def run_classification_train(dataloader, model, loss_compute, step_size=30, device=None):
    "Standard Training and Logging Function"
    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    start = time.time()
    total_tokens = 0
    total_loss, total_loss_bce = 0, 0
    tokens = 0
    for i, batch in enumerate(dataloader):

        b_input, _, _, _, b_labels = batch
        

        out = model.forward(b_input.type(torch.LongTensor).to(device), None, None)
        dist = torch.sum((out[:,0,:] - model.centroid) ** 2, dim=1)
        loss, _, _ = loss_compute(out, b_labels.to(device), dist)
        #total_loss_bce += loss_bce
        total_loss += loss

        if i % step_size == 0:
            elapsed = time.time() - start
            print("Epoch Step: %d / %d Loss: %f" % (i, len(dataloader), total_loss))
            #print("Epoch Step: %d / %d BCE Loss: %f" % (i, len(dataloader), total_loss_bce))
            start = time.time()
            tokens = 0
            #total_loss = 0
    
    #logging.info("Loss: {}".format(total_loss))
    return model, total_loss


def run_classification_test(dataloader, model, loss_compute, step_size=10, device=None):
    "Standard Training and Logging Function"
    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    preds = []
    distances = []
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            b_input, _, _, _, b_labels = batch

            out = model.forward(b_input.type(torch.LongTensor).to(device), None, None)
            out_p = model.generator(out)
            dist = torch.sum((out[:,0,:] - model.centroid) ** 2, dim=1)
            loss, _, _ = loss_compute(out, b_labels.to(device), dist)
            total_loss += loss 
            if i % step_size == 0:
                print("Epoch Step: %d / %d Loss: %f" %
                    (i, len(dataloader), total_loss / step_size))
                total_loss = 0
            #tmp = out_p.cpu().numpy()
            #preds += list(np.argmax(tmp,axis=1))
            distances += list(dist.cpu().numpy())
            #print(dist.cpu().numpy())

    return distances


def make_model(src_vocab, tgt_vocab, n=3, d_model=512, d_ff=2048, h=8, dropout=0.1, max_len=2000, centroid=0):
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout, max_len)
    model = TransformerModel(
        Encoder(EncoderLayer(d_model, c.deepcopy(attn), c.deepcopy(ff), dropout), n),
        nn.Sequential(Embeddings(d_model, src_vocab), c.deepcopy(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c.deepcopy(position)),
        Generator(d_model, tgt_vocab, centroid))

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model



def save_model(model, name="nulog", model_location=MODEL_LOCATION):
    model_name = MODEL_NAME.format(name)
    torch.save(model, os.path.join(model_location, model_name))
    
def load_model(name="nulog", model_location=MODEL_LOCATION):
    model_name = MODEL_NAME.format(name)
    model_path = os.path.join(model_location, model_name)
    if os.path.exists(model_path):
        model = torch.load(model_path, map_location=lambda storage, loc: storage)
        return model
    return None


class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, model, opt=None, is_test=False, numerator=1):
        self.model = model
        self.opt = opt
        self.is_test = is_test
        self.numerator = numerator

    def __call__(self, x, y, dist):
        loss = torch.mean((1-y)*(dist*dist) + (y)*(self.numerator/torch.abs(dist)))
        if not self.is_test:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if self.opt is not None:
                self.opt.step()
                self.opt.zero_grad()

        return loss.item(), None, None
    
def train_model(model, dataset, loss_compute, device, epochs=45, trained_condition=0.01):
    count_data = len(dataset)
    model.to(device)
    model.train()
    for epoch in range(epochs):
        print("Running epoch {} / {}".format(epoch + 1, epochs))
        model, loss = run_classification_train(dataset, model, loss_compute, step_size=1000, device=device)
        print('loss: {} cond: {}'.format(loss/count_data, trained_condition))
        if loss/count_data < trained_condition:
            break