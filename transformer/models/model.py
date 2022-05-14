import torch
from torch import nn
import torch.nn.functional as F
import math
from .config import HP

class PositionEncoding(nn.Module):

    def __init__(self,d_model ,max_len = 10000):

        super(PositionEncoding,self).__init__()
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.)/d_model))
        position = torch.arange(max_len).unsqueeze(1)
        pe = torch.zeros(1,max_len,d_model)
        pe[0,:,0::2] = torch.sin(position*div_term)
        pe[0,:,1::2] = torch.cos(position*div_term)

        #梯度不会更新，全局变量，自动追踪GPU
        self.register_buffer('pe',pe)


    def forward(self,x):
        x = x + self.pe[:,:x.size(1),:]
        return x

#encoder
class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.token_embedding = nn.Embedding(HP.grapheme_size,HP.encoder_dim)
        #注入位置信息
        self.pe = PositionEncoding(d_model=HP.encoder_dim,max_len= HP.encoder_max_input)
        #6层
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(HP.encoder_layer)])
        #
        self.drop = nn.Dropout(HP.encoder_drop_prob)
        #注入位置信息之前会进行缩放
        self.register_buffer('scale',torch.sqrt(torch.tensor(HP.encoder_dim).float()))

    def forward(self,inputs,inputs_mask):

        # inputs: [N , max_seq_len]
        token_emb = self.token_embedding(inputs)  #[N, max_seq_len,en_dim]
        inputs = self.pe(token_emb * self.scale) #注入位置信息

        inputs = self.drop(inputs) #[N, max_seq_len,en_dim]

        #encoder layer 6 层
        for idx , layer in enumerate(self.layers):
            inputs = layer(inputs,inputs_mask)

        return inputs



class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        # [N, max_seq_len,en_dim]

        self.self_att_layer_norm = nn.LayerNorm(HP.encoder_dim)
        self.pff_layer_norm = nn.LayerNorm(HP.encoder_dim)

        self.self_att = MultiHeadAttentionLayer(HP.encoder_dim,HP.nhead)
        self.pff = PointWiseFeedForwardLayer(HP.encoder_dim,HP.encoder_feed_forward_dim,HP.feed_forward_drop_prob)

        self.dropout = nn.Dropout(HP.encoder_drop_prob)

    def forward(self,inputs ,inputs_mask):
        #inputs shape: [N, max_seq_len,en_dim]
        _inputs ,att_res = self.self_att(inputs,inputs,inputs,inputs_mask) #[N, max_seq_len_q,hid_dim]
        inputs = self.self_att_layer_norm(inputs + self.dropout(_inputs)) #[N, max_seq_len_q,hid_dim]

        #feed forward  #[N, max_seq_len_q,hid_dim]
        _inputs = self.pff(inputs)
        inputs = self.pff_layer_norm(inputs + self.dropout(_inputs))

        return inputs

class MultiHeadAttentionLayer(nn.Module):

    def __init__(self,hid_dim ,nhead):
        super(MultiHeadAttentionLayer, self).__init__()
        self.hid_dim =  hid_dim
        self.nhead  = nhead
        assert not self.hid_dim % self.nhead
        self.head_dim = self.hid_dim // self,nhead

        #Q K V
        self.fc_q = nn.Linear(self.hid_dim,self.hid_dim)
        self.fc_k = nn.Linear(self.hid_dim,self.hid_dim)
        self.fc_v = nn.Linear(self.hid_dim,self.hid_dim)

        self.fc_o = nn.Linear(self.hid_dim,self.hid_dim)

        self.register_buffer('scale',torch.sqrt(torch.tensor(self.hid_dim).float()))


    def forward(self,query,key,value,inputs_mask = None):
        # Q K V
        # [N, max_seq_len,en_dim]
        bn = query.size(0)
        Q = self.fc_q(query)    # [N, max_seq_len_q,en_dim]
        K = self.fc_k(key)      # [N, max_seq_len_k,en_dim]
        V = self.fc_v(value)    # [N, max_seq_len_v,en_dim]

        #nhead
        ## [N, nhead,max_seq_len,head_dim]
        Q = Q.view(bn,-1,self.nhead,self.head_dim).premute((0,2,1,3))
        K = K.view(bn,-1,self.nhead,self.head_dim).premute((0,2,1,3))
        V = V.view(bn,-1,self.nhead,self.head_dim).premute((0,2,1,3))

        #Energy
        #Q  [N, nhead,max_seq_len_q,head_dim]
        #KT [N, nhead,head_dim,max_seq_len_k]
        # [N, nhead,max_seq_len_q,max_seq_len_k]  A
        energy = torch.matmul(Q,K.permute(0,1,3,2)) /self.scale

        if inputs_mask is not None:
            energy = energy.masked_fill(inputs_mask == 0 , -1.e10)

        attention = F.softmax(energy,dim=-1)

        #attention [N, nhead,max_seq_len_q,max_seq_len_k]  A
        #V [N, nhead,max_seq_len_v,head_dim]
        #out  [N, nhead,max_seq_len_q,head_dim]
        out = torch.matmul(attention,V)
        out = out.permute((0,2,1,3)).contiguous()#[N, max_seq_len_q, nhead,head_dim]
        out = out.view((bn,-1,self.hid_dim)) #[N, max_seq_len_q,hid_dim]

        out = self.fc_o(out)

        return out #[N, max_seq_len_q,hid_dim]

class PointWiseFeedForwardLayer(nn.Module):

    def __init__(self,hid_dim,pff_dim,pff_drop_out):
        super(PointWiseFeedForwardLayer, self).__init__()
        self.hid_dim  = hid_dim
        self.pff_dim = pff_dim
        self.pff_drop_out = pff_drop_out


        self.fc1 = nn.Linear(self.hid_dim,self.pff_dim)
        self.fc2 = nn.Linear(self.pff_dim,self.hid_dim)

        self.dropout = nn.Dropout(self.pff_drop_out)

    def forward(self,inputs):
        #inputs  [N, max_seq_len_q,hid_dim]
        inputs = self.dropout(F.relu(self.fc1(inputs)))  #[N, max_seq_len_q,pff_dim]
        out = self.fc2(inputs) # [N, max_seq_len_q,hid_dim]

        return out # [N, max_seq_len_q,hid_dim]

#deconder
class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.token_embedding = nn.Embedding(HP.phoneme_size,HP.decoder_dim)
        self.pe = PositionEncoding(d_model=HP.decoder_dim,max_len= HP.MAX_DECODE_STEP)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(HP.decoder_layer)])

        self.fc_out = nn.Linear(HP.decoder_dim,HP.phoneme_size) #128
        self.dropout = nn.Dropout(HP.decoder_drop_prob)

        self.register_buffer('scale',torch.sqrt(torch.tensor(HP.decoder_dim).float()))


    def forward(self,trg,enc_src,trg_mask,src_mask):
        # trg [N, max_seq_len]
        token_emb = self.token_embedding(trg)# [N , max_seq_len ,de_dim ]
        pos_emb = self.pe(token_emb * self.scale)
        trg = self.dropout(pos_emb)  # [N , max_seq_len ,de_dim ]

        for idx,layer in enumerate(self.layers):
            trg ,attention = layer(trg ,enc_src ,trg_mask ,src_mask)

        out = self.fc_out(trg)  # [N , max_seq_len ,phoneme_size ]

        return out,attention


class DecoderLayer(nn.Module):

    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.mask_self_att = MultiHeadAttentionLayer(HP.decoder_dim,HP.nhead)
        self.mask_self_norm = nn.LayerNorm(HP.decoder_dim)

        self.mha = MultiHeadAttentionLayer(HP.decoder_dim,HP.nhead)
        self.mha_norm = nn.LayerNorm(HP.decoder_dim)


        self.pff = PointWiseFeedForwardLayer(HP.decoder_dim,HP.decoder_feed_forward_dim,HP.feed_forward_drop_prob)
        self.pff_norm = nn.LayerNorm(HP.decoder_dim)

        self.dropout = nn.Dropout(HP.decoder_drop_prob)

    def forward(self,trg ,enc_src ,trg_mask ,src_mask):
        _trg , _ = self.mask_self_att(trg,trg,trg,trg_mask)
        trg = self.mask_self_norm(trg + self.dropout(_trg))

        _trg,attention = self.mha(trg,enc_src,enc_src,src_mask)
        trg = self.mha_norm(trg + self.dropout(_trg))

        _trg = self.pff(trg)
        trg = self.pff_norm(trg + self.dropout(_trg))

        return trg ,attention


#transformer
class Transformer(nn.Module):

    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    @staticmethod
    def create_src_mask(src):

        mask = (src != 0).unsqueeze(1).unsqueeze(2)

        return mask #[N,1,1,max_seq_len]
    @staticmethod
    def create_trg_mask(trg):

        trg_len = trg.size(1)
        pad_mask = (trg != 0).unsqueeze(1).unsqueeze(2)
        sub_mask = torch.tril(torch.ones(size=(trg_len,trg_len),dtype = torch.uint8)).bool()

        trg_mask = pad_mask & sub_mask

        return trg_mask #[N,1,max_seq_len,max_seq_len]

    def forward(self,src,trg):
        #src [N,max_seq_len]  trg [N,max_seq_len]
        src_mask = self.create_src_mask(src) #src_mask  [N,1,1,max_seq_len]
        trg_mask = self.create_trg_mask(trg) #trg_mask [N,1,max_seq_len,max_seq_len]

        enc_src = self.encoder(src,src_mask)
        output ,attention = self.decoder(trg,enc_src,trg_mask,src_mask)


        return output ,attention


















































