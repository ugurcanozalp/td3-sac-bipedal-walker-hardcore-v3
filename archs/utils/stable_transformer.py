from torch.nn.modules import TransformerEncoderLayer, TransformerEncoder
import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.autograd import Variable
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len=16, ratio=None):
        super(PositionalEncoding, self).__init__()
        self.embedding_scale = d_model**0.5
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        _log10000 = 9.21034037198
        #_log1000 = 6.907755278982137
        #_log100 = 4.605170185988092
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-_log10000 / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        if ratio is not None:
            self.ratio = torch.nn.Parameter(torch.tensor(ratio, dtype=torch.float), requires_grad=True)
        else:
            self.ratio = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float), requires_grad=False)

    def forward(self, x):
        x = x*self.embedding_scale + torch.flip(self.ratio*self.pe[:, :x.size(1), :], dims=[1])
        #x = x + self.pe[:, :x.size(1), :]
        return x

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len=16):
        super(LearnablePositionalEncoding, self).__init__()
        self.embedding = nn.Parameter(torch.zeros(seq_len, d_model))
        torch.nn.init.orthogonal_(self.embedding, gain=1.0)
    def forward(self, x):
        return x + self.embedding[:x.size(1)].unsqueeze(0)
    
class StableTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        #fill in reordering of operations as done in https://arxiv.org/pdf/1910.06764.pdf
        #d_model: dimension of embedding for each input
        super(StableTransformerLayer,self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        nn.init.xavier_uniform_(self.linear1.weight, gain=1.0)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        nn.init.xavier_uniform_(self.linear2.weight, gain=1.0)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()
        self.relu = nn.GELU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None, only_last_state=False):

        '''
        #ORIGINAL TRANSFORMER ORDERING
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        '''

        #HOW SHOULD THE DROPOUT BE APPLIED, it seems like dropout is completely missing the original source that residually connects?
        #This doesn't perfectly correspond to dropout used in TransformerXL I believe. (to do: read their code)


        src2 = self.norm1(src)
        if only_last_state:
            src2 = self.self_attn(src2[-1:], src2, src2, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)[0]
        else:
            src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)[0]

        if only_last_state:
            src2 = src[-1:] + self.dropout1(src2)
        else:
            src2 = src + self.dropout1(src2) # src2 = src + self.relu(self.dropout1(src2))
        
        src3 = self.norm2(src2)
        src3 = self.linear2(self.dropout(self.activation(self.linear1(src3))))
        src3 = self.dropout2(src3) + src2 # src3 = self.dropout2(self.relu(src3)) + src2

        return src3
        