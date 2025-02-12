# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torchtext.vocab as Vocab
import collections
from sklearn.model_selection import train_test_split,KFold
import torch.utils.data as Data
from torch.autograd import Variable
import json
import math
import torch.nn.utils.rnn as rnn_utils
import jieba
import random
import copy

class DataLoader(object):
    def __init__(self,config,args):
        self.config = config
        self.args = args
    

    def token(self,sen,Voc,l):
        result=[]
        for i in sen:
            if i in Voc.stoi:
                result.append(i)
            else:
                result.append('<unk>')
        # print(result)
        # print(self.pad(result,l,True))
        return self.pad(result,l,True)

    def message_token(self,x):
            
        for index,sen in enumerate(x):
            x[index] = self.token(sen,self.Voc,self.config.sentence_len)
        return x

    def pad(self,x,max_l,flag,item='<pad>'):

        # sentence padding:item= '<pad>'
        # message padding:item=['<pad>']*len(x[0])

        if flag:
            x = x[:max_l] if len(x)>=max_l else x+[item]*(max_l-len(x))
        else:
            x = x[:max_l] if len(x)>=max_l else x
        return x

    def sen_len_cut(self,x):
        for index,i in enumerate(x):
            if i>self.config.sentence_len:
                x[index]=self.config.sentence_len
        return x

    def price_onehot(self,x,gate=[30,50,100,200]):
        result = np.zeros(len(gate)+1)
        for index,i in enumerate(gate):
            if index==0 and x<i:
                result[0] = 1
                return list(result)
            if index>0:
                if x>=result[index-1] and x<i:
                    result[index] = 1
                    return list(result)
        result[-1] = 1
        return list(result)

    def cate_onehot(self,x,cate=[100004,100010,100007,100012,100003,100008,100009,100011,100001]):
        result = np.zeros(len(cate))
        if x in cate:
            result[cate.index(x)] = 1
        return list(result)
    
    def age_onehot(self,x,age=['-40', '-60', '-18', '-3', '10-', '18-']):
        result = np.zeros(len(age))
        if x in age:
            result[age.index(x)] = 1
        return list(result)
    
    def sex_onehot(self,x,sex=['情侣', '女', '通用', '男']):
        result = np.zeros(len(sex))
        if x in sex:
            result[sex.index(x)] = 1
        return list(result)
    
    def heat_onehot(self,x,heat=[500,2000,10000,50000]):
        result = np.zeros(len(heat)+1)
        for index,i in enumerate(heat):
            if index==0 and x<i:
                result[0] = 1
                return list(result)
            if index>0:
                if x>=result[index-1] and x<i:
                    result[index] = 1
                    return list(result)
        result[-1] = 1
        return list(result)

    def brand_onehot(self,x,brand=['A类品牌','B类品牌','C类品牌','D类品牌']):
        result = np.zeros(len(brand))
        if x in brand:
            result[brand.index(x)] = 1
        return list(result)
    

    def Voc_match(self,x,Voc):
        result = []
        for sen in x:
            result.append([Voc.stoi[i] for i in sen])
        return result
    


    def Dataload(self):

        file = open(self.config.path,errors = 'ignore',encoding='utf-8-sig')
        data = pd.read_csv(file,index_col=0).reset_index()
        data = data[(data['sen_num']>0) & (data['sen_num']<=10)]
        data['post_id'] = data['post_id'].astype('int')
        data['content2'] = data['content'].apply(lambda x:eval(x))
        data['content_svm2'] = data['content'].apply(lambda x:eval(x))
        data['content_save'] = data['content'].apply(lambda x:''.join(eval(x)))
        data['content_svm1'] = data['content_save']
        data['cut_words'] = data['cut_words'].apply(lambda x:eval(x))
        data['sen_len'] = data['sen_len'].apply(lambda x:self.sen_len_cut(eval(x)))
        data['label'] = data['label'].apply(lambda x:eval(x))
        data['price'] = data['price'].apply(lambda x:self.price_onehot(x))
        data['bd_cate1_id'] = data['bd_cate1_id'].apply(lambda x:self.cate_onehot(x))
        data['props_sex'] = data['props_sex'].apply(lambda x:self.sex_onehot(x))
        data['props_age'] = data['props_age'].apply(lambda x:self.age_onehot(x))
        data['heat'] = data['heat'].apply(lambda x:self.heat_onehot(x))
        data['brand'] = data['brand_grade'].apply(lambda x:self.brand_onehot(x))
        data['item'] = data['price']+data['bd_cate1_id']+data['brand']+data['props_sex']+data['props_age']+data['heat']
        data['click'] = data['click_cnt']/data['read_cnt']
        data['content2'] = data['content2'].apply(lambda x:self.message_token(x))
        data['cut_words'] = data['cut_words'].apply(lambda x:self.message_token(x))
        data['content_save'] = data['content_save'].apply(lambda x:self.token(x,self.Voc,100))
        data['content2'] = data['content2'].apply(lambda x:self.pad(x,self.config.message_len,False))
        data['sen_len'] = data['sen_len'].apply(lambda x:self.pad(x,self.config.message_len,False))
        data['label'] = data['label'].apply(lambda x:self.pad(x,self.config.message_len,False))
    
        data['content_save'] = data['content_save'].apply(lambda x:[self.Voc.stoi[i] for i in x])
        data['sentence_voc'] = data['content2'].apply(lambda x:self.Voc_match(x,self.Voc))
     
        data['sentence_voc_word'] = data['cut_words'].apply(lambda x:self.Voc_match(x,self.Voc_word))
        data_ann = data[data['is_label']==1]
        data_un = data[data['is_label']==0]
        
        return data,data_ann,data_un


    def load_label(self): #flag=1:label
        data_result = self.Dataload()
        data_all = [[],[],[]]
        for index,data in enumerate(data_result):
            #post_id = data['post_id'].tolist()
            features = data['sentence_voc'].tolist()
        
            text_all = data['content_save'].tolist()
            label = data['label'].tolist()
            click = data['click'].tolist()
            post_id = data['post_id'].tolist()
            sen_len = data['sen_len'].tolist()
            sen_num = data['sen_num'].tolist()
            item = data['item'].tolist()
            features_all = []
            label_all = []
            sen_len_all = []
            for i in range(len(features)):
                sen_len_all.append(torch.tensor(sen_len[i]).unsqueeze(dim=1))
                features_all.append(torch.tensor(features[i]))
                label_all.append(torch.tensor(label[i]).unsqueeze(dim = 1))
            
            sen_len_all = rnn_utils.pad_sequence(sen_len_all,padding_value=1).permute(1,0,2).squeeze()
            features_all = rnn_utils.pad_sequence(features_all).permute(1,0,2)
            label_all = rnn_utils.pad_sequence(label_all,padding_value=self.config.feature_size+1).permute(1,0,2).squeeze()

            for i in range(features_all.shape[0]):
                t_all = torch.tensor(text_all[i])
                sen_n = torch.tensor(sen_num[i])
                iitem = torch.tensor(item[i])
                c = torch.tensor(float(click[i]))
                pid = torch.tensor(int(post_id[i]))
                data_all[index].append((features_all[i,:,:],c,label_all[i,:].int(),iitem,sen_n,sen_len_all[i,:],pid,t_all))

        data_all[0] = [i for i in data_all[0] if self.sampling(int(i[1].numpy().tolist()/0.01))]
        data_all[1] = [i for i in data_all[1] if self.sampling(int(i[1].numpy().tolist()/0.01))]
        data_all[2] = [i for i in data_all[2] if self.sampling(int(i[1].numpy().tolist()/0.01),p1=0.4,p2=0.5,p3=0.4,p4=0.3,p5=0.3,p6=0.8)]
        
        return data_all[0],data_all[1],data_all[2]

    def seg_sentence(self,sentence):
        sentence_seged = jieba.cut(sentence.strip())
        with open("/data/mas/yuanyuan/persuasion/word_vector/stopwords.json", 'r') as  f:
            stopwords = json.load(f)
        outstr = ''
        for word in sentence_seged:
            if word not in stopwords:
                if word != '\t':
                    outstr += word
                    outstr += " "
        return outstr

    def Voc_Count(self,df,column_name):
        sen_all = df[column_name].tolist()
        voc_count = collections.Counter([word for message in sen_all for sen in message for word in sen])
        return Vocab.Vocab(voc_count,min_freq=5,specials=['<unk>', '<pad>'])

    def generate_pretrained_vocab(self):
        file = open(self.config.path,errors = 'ignore',encoding='utf-8-sig')
        df = pd.read_csv(file,index_col=0)
        file.close()
        df['content'] = df['content'].apply(lambda x:eval(x))
        Voc = self.Voc_Count(df,'content')
        Voc_word = self.Voc_Count(df,'cut_words')

        with open(self.config.vector_path,'r') as json_file:
            Dict=json.load(json_file)

        def load_pretrained_embedding(words, pretrained_vocab):
            embed = torch.zeros(len(words), self.config.embedding_size)
            for i, word in enumerate(words):
                if word!='<pad>' and word!='<unk>' and word in pretrained_vocab:
                    embed[i, :] = torch.FloatTensor(pretrained_vocab[word])
            return embed
        
        return load_pretrained_embedding(Voc.itos, Dict),Voc,load_pretrained_embedding(Voc_word.itos, Dict),Voc_word

    def data_prepare(self):
        pretrained_vocab,Voc,pretrained_vocab_word,Voc_word = self.generate_pretrained_vocab()
        if self.args.vocab==0:
            self.pretrained_vocab = pretrained_vocab
        else:
            self.pretrained_vocab = pretrained_vocab_word
        self.Voc = Voc
        self.Voc_word = Voc_word
        _,data_ann,data_un = self.load_label()
        return data_ann,data_un

    def distribution(self,x):
        c = [i[1].numpy().tolist() for i in x]
        print(np.mean(c),np.std(c))
        c = [int(i[1].numpy().tolist()/0.01) for i in x]
        dis = [[key,value] for key,value in dict(pd.value_counts(c,normalize=True)).items()]
        dis.sort()
        for i in range(len(dis)-1):
            dis[i+1][1] = dis[i][1]+dis[i+1][1]
        return dis

    def sampling(self,x,p1=0.45,p2=0.45,p3=0.45,p4=0.4,p5=0.6,p6=0.8):
        if x==0 and np.random.uniform(0,1)<p1:
            return True
        elif x==1 and np.random.uniform(0,1)<p2:
            return True
        elif x==2 and np.random.uniform(0,1)<p3:
            return True
        elif x>2 and x<=7 and np.random.uniform(0,1)<p4:
            return True
        elif x>7 and x<=13 and np.random.uniform(0,1)<p5:
            return True
        elif x>13 and x<=20 and np.random.uniform(0,1)<p6:
            return True
        elif x>20:
            return True
        return False

    def train_prepare(self,flag=1,deep = True):
        data_ann,data_un = self.data_prepare()
        print('data prepare finished!')

        train_data_un,test_data_un = train_test_split(data_un, test_size=0.2, random_state=self.args.seed)

        train_data,test_data = train_test_split(data_ann, test_size=0.2, random_state=self.args.seed)

        test_all = test_data # labeled data
        val_all = test_data_un+test_data # labeled and unlabeled data

        # print(np.mean([i[1] for i in test_all]),np.std([i[1] for i in test_all]))
        # print(np.mean([i[1] for i in val_all]),np.std([i[1] for i in val_all]))

        train_data_un = random.sample(train_data_un,int(len(train_data_un)*self.args.unlabel_pro))

        if flag==1:
            
            data_ann_train = Data.DataLoader(train_data, batch_size = self.args.batch_size,shuffle=True)
            data_un_iter = Data.DataLoader(train_data_un, batch_size = self.args.batch_size,shuffle=True)
        
        else:

            data_ann_train = Data.DataLoader(train_data, batch_size = 10000)
            data_un_iter = Data.DataLoader(train_data_un, batch_size = 10000)
            
        data_val = Data.DataLoader(val_all, batch_size = 5000)
        data_test = Data.DataLoader(test_all, batch_size = 5000)

        if deep:

            return data_ann_train,data_un_iter,data_val,data_test,self.pretrained_vocab

        else:
            return train_data,test_all,val_all