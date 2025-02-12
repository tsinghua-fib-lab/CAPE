import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils
import copy
from mca import *

class Disentangle_attn(nn.Module):
    def __init__(self, config,pretrained_embedding,args):
        super(Disentangle_attn, self).__init__()
        self.embedding_size = config.embedding_size
        self.hidden_size = args.hidden_size
        self.feature_size = config.feature_size
        self.args = args
        self.pretrained_embedding = pretrained_embedding
        self.config = config
        self.item_size = args.item_size
        
        # embedding
        self.embed = nn.Embedding.from_pretrained(pretrained_embedding)

        # item
        self.price_proj = nn.Linear(5,self.item_size)
        self.cate_proj = nn.Linear(9,self.item_size)
        self.brand_proj = nn.Linear(4,self.item_size)
        self.sex_proj = nn.Linear(4,self.item_size)
        self.age_proj = nn.Linear(6,self.item_size)
        self.heat_proj = nn.Linear(5,self.item_size)

        # persuasion
        # word level
        self.word_GRU = nn.GRU(self.embedding_size, self.hidden_size, batch_first = True,dropout = self.args.dropout)
        self.w_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_context_vector = nn.Parameter(torch.randn([self.hidden_size, 1]).float())
        self.word_linear = nn.Linear(self.hidden_size, self.feature_size)
        self.activate = nn.Linear(self.hidden_size,self.hidden_size)

        self.CE = Context_Excitation(config,args)
        self.linear_p = FC(self.hidden_size, 1, dropout_r=0.0, use_relu=True)
        self.linear_i = FC(self.item_size, 1, dropout_r=0.0, use_relu=True)
        self.lienar_out = nn.Linear(self.hidden_size*self.feature_size+6*self.item_size,1)
        self.softmax = nn.Softmax(dim=2)
        self.predict = nn.Linear(self.feature_size*2,self.config.output_size)
        self.linear_p_out = nn.Linear(self.args.hidden_size,self.feature_size)
        self.linear_i_out = nn.Linear(self.args.item_size,self.feature_size)

    def forward(self, x, item,sen_num,sen_len,t_all):

        # batch_size * sentence_num * sentence_len
        batch_size = x.shape[0]
        sentence_num = x.shape[1]

        price = self.price_proj(item[:,:5])
        cate = self.cate_proj(item[:,5:14])
        brand = self.brand_proj(item[:,14:18])
        props_sex = self.sex_proj(item[:,18:22])
        props_age = self.age_proj(item[:,22:28])
        heat = self.heat_proj(item[:,28:])

        item_embedding = torch.cat((price,cate,brand,props_sex,props_age,heat),dim=1).view([batch_size,6,self.item_size])
        x = x.view([x.shape[0] * x.shape[1], x.shape[2]])

        sen_len = sen_len.view([sen_len.shape[0]*sen_len.shape[1],-1]).squeeze()

        x_embed_origin = self.embed(x)
        # batch_size*sentence_num * sentence_len * embedding_size

        x_embed = rnn_utils.pack_padded_sequence(x_embed_origin, sen_len, batch_first=True, enforce_sorted=False)
        
        # pad
        pad_mask_s = torch.zeros(batch_size*sentence_num,1).cuda()
        sen_len_pad = sen_len.view([batch_size*sentence_num,-1])

        pad_mask = torch.ones(x.size(0),x.size(1)).cuda()

        for index,i in enumerate(sen_len_pad.cpu().detach().numpy()):
            pad_mask[index][:int(i)] = torch.tensor([0]).cuda()
            if i>1:
                pad_mask_s[index] = 1
        
        pad_mask = pad_mask.unsqueeze(dim=2)
        pad_mask_s = pad_mask_s.view([batch_size,sentence_num]) # batch_size * sentence_num

        x_out, h = self.word_GRU(x_embed)

        x_out,_ = rnn_utils.pad_packed_sequence(x_out, batch_first=True,total_length=self.config.sentence_len)
        
        Hw = torch.tanh(self.w_proj(x_out))
        # batch_size*sentence_num * sentence_len * hidden_size
        w_s = Hw.matmul(self.w_context_vector)
        w_s.masked_fill_(pad_mask.bool(),-1e9)
        w_score = F.softmax(w_s,dim=1)
        x_out = x_out.mul(w_score)
        x_out = torch.sum(x_out, dim = 1)
        x_out = F.relu(self.activate(x_out))

        x_FC = self.word_linear(x_out)

        x_out = x_out.view([batch_size, sentence_num, x_out.shape[1]]) # batch_size * sentence_num * hidden_size

        x_prob = x_FC.view([batch_size, sentence_num, x_FC.shape[1]])
        x_prob = x_prob.permute(0,2,1) # batch_size * feature_size * sentence_num
        
        x_prob = F.softmax(x_prob,dim=1) # batch_size * feature_size * sentence_num

        pad_mask_s = pad_mask_s.unsqueeze(dim=1).repeat(1,x_prob.shape[1],1)
        
        x_prob = x_prob.mul(pad_mask_s)

        x_p_out = x_prob.matmul(x_out) 
        # x_p: batch_size * feature_size * hidden_size
        # item_embedding: batch_size * 6 * item_size

        attn_i,attn_p = self.CE(item_embedding,x_p_out)

        x_p = x_p_out.mul(attn_p) 
        item_embedding = item_embedding.mul(attn_i)

        x_p = torch.sum(x_p,dim = 1)
        item_embedding = torch.sum(item_embedding,dim = 1)

        x_p = F.relu(self.linear_p_out(x_p))
        x_i = F.relu(self.linear_i_out(item_embedding))

        x_cat = torch.cat((x_p,x_i),dim=1)

        output = self.predict(x_cat)

        return x_FC, output, attn_i, attn_p, x_p_out 
        # x_p is the persuasion representation of the text, batch_size * feature_size * hidden_size


