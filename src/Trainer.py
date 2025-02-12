import torch
import pandas as pd
import json
import collections
import numpy as np
import torchtext.vocab as Vocab
import copy
import time
import torch.utils.data as Data
from config import Config
from data_processing import *
from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix,mean_squared_error,mean_absolute_error
from MessageLoss import MessageLoss
from sklearn.model_selection import KFold
from dataloader import DataLoader
from sklearn.model_selection import train_test_split
from model import *
import random
from metrics import *
import torch.nn.functional as F
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVR,SVC
import xlearn as xl


class Trainer(object):
    def __init__(self,args,logging):
        self.config = Config()
        self.args = args
        self.logging = logging

    def evaluate(self,test_iter,net,Myloss,f='Test'):
        net.eval()
        _,test_iter = self.getTrainBatches(test_iter)
        correct = 0
        count = 0
        y_true = []
        y_pred = []
        rmse = 0
        num = 0
        
        for d in test_iter:
            X,click,labels,item,sen_num,sen_len,post_id,t_all = self.cuda_pre(d)
            sen_out,y,_,_,p_embed = net(X,item,sen_num,sen_len,t_all)
            loss,labeled_doc_loss, unlabeled_doc_loss,labeled_sent_loss, covariance_loss, y_true,y_pred = Myloss(labeled_doc = y, target1 = click, labeled_sent = sen_out, target2 = labels, unlabeled_doc = None, target3 = None, persuasion_embed = p_embed,mode = 'test')

        loss,labeled_sent_loss,covariance_loss = loss.item(),labeled_sent_loss.item(),covariance_loss.item()

        mae = labeled_doc_loss.cpu().detach().numpy().tolist()

        rmse = np.sqrt(mean_squared_error(y.cpu().detach().numpy(), click.cpu().detach().numpy()))

        accuracy_group = acc(y_pred,y_true)

        true_rmse = np.std(click.cpu().detach().numpy())

        result_show_acc(self.logging,(loss,labeled_sent_loss,covariance_loss,rmse,mae),accuracy_group[:-1],flag = f)

        print('true_rmse:%.3f' % true_rmse)

        return loss,rmse,mae,accuracy_group[1],accuracy_group[0],accuracy_group[2],accuracy_group[3] # f1 score

    def cuda_pre(self,d):
        X = d[0].cuda()
        click = d[1].cuda()
        labels = d[2].long().cuda()
        item = d[3].cuda()
        sen_num = d[4].cuda()
        sen_len = d[5].cuda()
        post_id = d[6].cuda()
        t_all = d[7].cuda()
        return X,click,labels,item,sen_num,sen_len ,post_id,t_all


    def getTrainBatches(self,train_iter):
        data = []
        for X,click,labels,item,sen_num,sen_len,post_id,t_all in train_iter:
            data.append((X,click,labels,item,sen_num,sen_len,post_id,t_all))
        return len(data)-1,data

    def init_model(self):
        net = Disentangle_attn(self.config,self.pretrained_vocab,self.args).cuda()
        optimizer = torch.optim.Adam(params=net.parameters(),lr=self.args.lr,weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience = self.args.patience, min_lr=self.args.min_lr)
        Myloss = MessageLoss(self.config,self.args).cuda()
            
        def init_weights(m):
            for name, param in m.named_parameters():
                nn.init.uniform_(param.data, -0.1, 0.1)
                #nn.init.normal_(param.data, mean=0, std=0.01)
        net.apply(init_weights)
        train_params = list(filter(lambda p: p.requires_grad, net.parameters()))
        print('Trainable Parameters:', np.sum([p.numel() for p in train_params]))
        return (net,optimizer,scheduler,Myloss)

    def train(self,model,optimizer,scheduler,Myloss,train_iter_ann,train_iter_un,data_val,data_test,model_save):
        min_loss = 1e5
        early_stop_num,step = 0,0
        max_f1,max_acc = 0,0
        min_loss= 100000
        min_mae_test,min_rmse_test,min_mae_val,min_rmse_val = 100000,100000,100000,100000
        ann_num,un_num = -1,-1
        epoch = 0
        loss,labeled_doc_loss,unlabeled_doc_loss,labeled_sent_loss,unlabeled_sent_loss,covariance_loss = 0,0,0,0,0,0
        y_true, y_pred = [],[]

        while step<=self.args.step_num:
            if ann_num == -1:
                
                y_true, y_pred = [],[]
                loss,labeled_doc_loss,unlabeled_doc_loss,labeled_sent_loss,unlabeled_sent_loss,covariance_loss = 0,0,0,0,0,0

                epoch += 1
                
                ann_num,data_ann = self.getTrainBatches(train_iter_ann)
                
                loss_test,rmse_test,mae_test,f1_test, acc_test,_,_= self.evaluate(data_test,model,Myloss,'Test!')

                _,rmse_val,mae_val,_,_,_,_ = self.evaluate(data_val,model,Myloss,'Validate')

                scheduler.step(loss_test)

                min_rmse_test = min(min_rmse_test,rmse_test)
                min_mae_test = min(min_mae_test,mae_test)
                min_rmse_val = min(min_rmse_val,rmse_val)
                min_mae_val = min(min_mae_val,mae_val)

                if loss_test<min_loss or f1_test>max_f1 or acc_test>max_acc:
                    early_stop_num = 0
                    print('satisfied!',loss_test<min_loss,f1_test>max_f1,acc_test>max_acc)
                    if f1_test>max_f1 or acc_test>max_acc:
                        torch.save(model.state_dict(),model_save)
                        print("!!!!!!!!!! Model Saved !!!!!!!!!!")
                        self.logging.info("!!!!!!!!!! Model Saved !!!!!!!!!!")
                    
                    min_loss = min(loss_test,min_loss)
                    max_f1 = max(f1_test,max_f1)
                    max_acc = max(acc_test,max_acc)

                else:
                    if early_stop_num>=self.args.epoch_num:
                        print('early stop!!!')
                        self.logging.info('early stop!!!')
                        break
                    elif optimizer.param_groups[0]['lr']<=self.args.min_lr: 
                        early_stop_num += 1
                        print('early_stop_num:%d',early_stop_num)
                        self.logging.info('early_stop_num:%d',early_stop_num)
                    else:
                        print('dissatisfied!')

                

            if un_num == -1:
                un_num,data_un = self.getTrainBatches(train_iter_un)
                

            model.train()

            X_ann,click_ann,label_ann,item_ann,sen_num_ann,sen_len_ann,post_id_ann,t_all_ann = self.cuda_pre(data_ann[ann_num])
            
            ann_sentence_out, ann_message_out,_,_,ann_p_embed = model(X_ann,item_ann,sen_num_ann,sen_len_ann,t_all_ann)


            ann_num = ann_num - 1

            if self.args.without_un==0:
                X_un,click_un,label_un,item_un,sen_num_un,sen_len_un ,post_id_un,t_all_un= self.cuda_pre(data_un[un_num])
                _, un_message_out,_,_,un_p_embed = model(X_un,item_un,sen_num_un,sen_len_un,t_all_un)

                un_num = un_num-1
            
            if self.args.without_un==1:
                unlabel,t3 = None,None
            else:
                unlabel,t3 = un_message_out,click_un
            
            l, labeled_doc_l,unlabeled_doc_l,labeled_sent_l,covariance_l,true,pred = Myloss(labeled_doc = ann_message_out, target1 = click_ann,labeled_sent = ann_sentence_out, target2 = label_ann, unlabeled_doc = unlabel, target3 =t3,persuasion_embed = ann_p_embed,mode = 'train')  
            
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            loss += l.item()
            labeled_sent_loss += labeled_sent_l.item()
            labeled_doc_loss += labeled_doc_l.item()*click_ann.shape[0]
            covariance_loss += covariance_l.item()*click_ann.shape[0]
            y_true += true
            y_pred += pred

            if ann_num == -1:
                
                print('learning rate:',optimizer.param_groups[0]['lr'])  
                self.logging.info('learning rate:{}'.format(optimizer.param_groups[0]['lr']))
                
                step += 1
                accuracy_group = acc(y_pred,y_true)

                # print
                print('-------------------------------------------------------------')
                print('train step:%d' % epoch)
                result_show_acc(self.logging,(loss,labeled_sent_loss,covariance_loss,labeled_doc_loss,0),accuracy_group[:-1],flag = 'Train!')
                self.logging.info('-------------------------------------------------------------')
                self.logging.info('train step:%d' % epoch)
        
        return min_rmse_test,min_mae_test,min_rmse_val,min_mae_val

    def data_batch_cuda(self,d_iter,model,info,file_name=''):
        num,data = self.getTrainBatches(d_iter)
        for d in d_iter:
            X,click,label,item,sen_num,sen_len,post_id,t_all = self.cuda_pre(d)
            p_out,_,attn_i,attn_p,_ = model(X,item,sen_num,sen_len,t_all)

        p_out = p_out.cpu().detach()
        p_out = F.softmax(p_out, dim = 1)
        p_out = p_out.view([-1,self.config.message_len,self.config.feature_size])

        attn_i,attn_p,item,label,click,p_pred = attn_i.cpu().detach().numpy(),attn_p.cpu().detach().numpy(),item.cpu().detach().numpy(),label.cpu().detach().numpy(),click.cpu().detach().numpy(),p_out.numpy()
        result = []
        for i in range(attn_i.shape[0]):
            result.append([attn_i[i],attn_p[i],item[i],label[i],click[i],p_pred[i]])

        result = pd.DataFrame(result,columns=['attn_i','attn_p','item','label','click','p_pred'])
        result.to_pickle('/home/mas/yuanyuan/workplace/persuasion_disentangle_p/result/attn/{}_{}.pkl'.format(info,file_name))
        print('save!')

    def attn_get(self,model,info,data_ann_train,data_un_train,test,val):
        r_ann = self.data_batch_cuda(data_ann_train,model,info,'ann')
        r_un = self.data_batch_cuda(data_un_train,model,info,'un')
        r_test = self.data_batch_cuda(test,model,info,'test')
        r_val = self.data_batch_cuda(val,model,info,'val')
        

    def trainstep(self):

        mae_test,rmse_test,mae_val,rmse_val = 0,0,0,0
        
        model_save = '/home/mas/yuanyuan/workplace/persuasion_disentangle_p/model/model:{},unlabel:{},dropout:{},lr:{},weight_decay:{},batch_size:{},w2:{},w3:{},hidden_size:{},item_size:{},unlabel_pro:{}'.format(self.args.model,self.args.without_un,self.args.dropout,self.args.lr,self.args.weight_decay,self.args.batch_size,self.args.w2,self.args.w3,self.args.hidden_size,self.args.item_size,self.args.unlabel_pro)

        start_time = time.time()
        print('------------------------- Loading data -------------------------')

        nlp = DataLoader(self.config,self.args)
        data_ann_train,data_un_iter,data_val,data_test,pretrained_vocab= nlp.train_prepare()

        self.pretrained_vocab = pretrained_vocab
    
        print('\n------------------------- Initialize Model -------------------------')
        model,optimizer,scheduler,Myloss = self.init_model()

        if self.args.test == 0:
            print('\n------------------------- Training -------------------------')
            rmse_test,mae_test,rmse_val,mae_val = self.train(model,optimizer,scheduler,Myloss,data_ann_train,data_un_iter,data_val,data_test,model_save)
            
        model.load_state_dict(torch.load(model_save))

        
        info = 'model:{},unlabel:{},dropout:{},lr:{},weight_decay:{},batch_size:{},w2:{},w3:{},hidden_size:{},item_size:{},unlabel_pro:{}'.format(self.args.model,self.args.without_un,self.args.dropout,self.args.lr,self.args.weight_decay,self.args.batch_size,self.args.w2,self.args.w3,self.args.hidden_size,self.args.item_size,self.args.unlabel_pro)
        self.logging.info('--------------------info----------------------')
        self.logging.info(info)

        print('\n------------------------- Validate -------------------------')
        self.logging.info('\n------------------------- Validate -------------------------')

        loss,rmse,mae,f1,accuracy,precision,recall = self.evaluate(data_val,model,Myloss)

        temp1 = [rmse,mae,f1,accuracy,precision,recall]

        print('rmse:%.4f,mae:%.4f,rmse_min:%.4f,mae_min:%.4f,f1:%.4f,acc:%.4f,precision:%.4f,recall:%.4f' % (rmse,mae,rmse_val,mae_val,f1,accuracy,precision,recall))
        
        print('\n------------------------- Testing -------------------------')
        self.logging.info('\n------------------------- Testing -------------------------')
        loss,rmse,mae,f1,accuracy,precision,recall = self.evaluate(data_test,model,Myloss)

        temp2 = [rmse,mae,f1,accuracy,precision,recall]

        print('rmse:%.4f,mae:%.4f,rmse_min:%.4f,mae_min:%.4f,f1:%.4f,acc:%.4f,precision:%.4f,recall:%.4f' % (rmse,mae,rmse_test,mae_test,f1,accuracy,precision,recall))

        # print('\n------------------------- attn weight -------------------------')
        # att_ann_train,att_un_iter,att_val,att_test,_= nlp.train_prepare(flag=0)

        # self.attn_get(model,info,att_ann_train,att_un_iter,att_val,att_test)

        return [info]+temp1+temp2

    def data_ML(self):
        def index_str(data):
            x = [i[0] for i in data]
            sen_num = [i[4] for i in data]
            sen_len = [i[5] for i in data]
            label = [i[2] for i in data]
            click = [[i[1]] for i in data]
            product = np.array([i[3] for i in data])
            feature = []
            label_true = []
            click_true = []
            text = []
            for index in range(len(sen_num)):
                click_true.append(click[index])
                x[index] = x[index][:sen_num[index]]
                for i in range(sen_num[index]):
                    temp = x[index][i][:sen_len[index][i]]
                    temp = ''.join([str(j)+'0 ' for j in temp])
                    feature.append(temp) # 每个句子tf-idf特征
                    label_true.append(label[index][i])
                text.append(''.join(feature[-sen_num[index]:])) # 全文tf-idf特征
            
            return [feature,np.array(label_true),text,np.array(click_true),product]

        nlp = DataLoader(self.config,self.args)
        train,test,val= nlp.train_prepare(deep=False)
        train = [[i[0].numpy(),i[1].numpy().tolist(),i[2].numpy(),i[3].numpy(),i[4].numpy(),i[5].numpy()] for i in train]
        test = [[i[0].numpy(),i[1].numpy().tolist(),i[2].numpy(),i[3].numpy(),i[4].numpy(),i[5].numpy()] for i in test]
        val = [[i[0].numpy(),i[1].numpy().tolist(),i[2].numpy(),i[3].numpy(),i[4].numpy(),i[5].numpy()] for i in val]

        train = index_str(train)
        test = index_str(test)
        val = index_str(val)

        corpus1 = train[0]+test[0]+val[0]
        corpus2 = train[2]+test[2]+val[2]
        vectorizer1 = TfidfVectorizer(analyzer='word',  ngram_range=(0,1),max_features = 1000)
        vectorizer2 = TfidfVectorizer(analyzer='word',  ngram_range=(0,1),max_features = 1000)
        vectorizer1.fit(corpus1)
        vectorizer2.fit(corpus2)

        train[0] = vectorizer1.transform(train[0]).toarray()
        train[2] = np.concatenate((vectorizer2.transform(train[2]).toarray(),train[4]),axis=1)
        test[0] = vectorizer1.transform(test[0]).toarray()
        test[2] = np.concatenate((vectorizer2.transform(test[2]).toarray(),test[4]),axis=1)
        val[0] = vectorizer1.transform(val[0]).toarray()
        val[2] = np.concatenate((vectorizer2.transform(val[2]).toarray(),val[4]),axis=1)

        return train, test, val


    def SVM(self):
        train,test,val = self.data_ML()
        # regr = LinearSVR(random_state=100, tol=1e-8)
        # regr.fit(train[2],train[3])
        # y_pred_test = regr.predict(test[2])
        # y_pred_val = regr.predict(val[2])
        # rmse_test = np.sqrt(mean_squared_error(y_pred_test,test[3]))
        # mae_test = mean_absolute_error(y_pred_test,test[3])
        # rmse_val = np.sqrt(mean_squared_error(y_pred_val,val[3]))
        # mae_val = mean_absolute_error(y_pred_val,val[3])
        # print('rmse_test:%.4f,mae_test:%.4f,rmse_val:%.4f,mae_val:%.4f' % (rmse_test,mae_test,rmse_val,mae_val))

        clf = SVC(kernel='rbf',random_state=0,C=0.3)
        clf.fit(train[0], train[1])
        y_pred_test = clf.predict(test[0])
        acc_test = np.mean([int(y_pred_test[i]==test[1][i]) for i in range(len(y_pred_test))])
        f1_test = f1_score(test[1],y_pred_test,average='macro')
        precision_test = precision_score(test[1],y_pred_test,average='macro')
        recall_test = recall_score(test[1],y_pred_test,average='macro')

        print('acc_test:%.4f,f1_test:%.4f,precision_test:%.4f,recall_test:%.4f' % (acc_test,f1_test,precision_test,recall_test))
        

    def FM(self):
        train,test,val = self.data_ML()
        
        train = np.concatenate((train[3],train[2]),axis=1)
        test = np.concatenate((test[3],test[2]),axis=1)
        val = np.concatenate((val[3],val[2]),axis=1)

        np.savetxt("/home/mas/yuanyuan/workplace/persuasion_disentangle_p/data/train_item_text.txt", train)
        np.savetxt("/home/mas/yuanyuan/workplace/persuasion_disentangle_p/data/test_item_text.txt",test)
        np.savetxt("/home/mas/yuanyuan/workplace/persuasion_disentangle_p/data/val_item_text.txt",val)

        fm_model = xl.create_fm()
        fm_model.setTrain("/home/mas/yuanyuan/workplace/persuasion_disentangle_p/data/train_item_text.txt") 
        param = {'task':'reg', 'lr':0.1, 'metric': 'mae','epoch':100,'k':4}
        fm_model.fit(param, "/home/mas/yuanyuan/workplace/persuasion_disentangle_p/data/model.out")


        fm_model.setTest("/home/mas/yuanyuan/workplace/persuasion_disentangle_p/data/test_item_text.txt")
        fm_model.predict("/home/mas/yuanyuan/workplace/persuasion_disentangle_p/data/model.out", "/home/mas/yuanyuan/workplace/persuasion_disentangle_p/data/output_test.txt")

        fm_model.setTest("/home/mas/yuanyuan/workplace/persuasion_disentangle_p/data/val_item_text.txt")
        fm_model.predict("/home/mas/yuanyuan/workplace/persuasion_disentangle_p/data/model.out", "/home/mas/yuanyuan/workplace/persuasion_disentangle_p/data/output_val.txt")

        result=[]
        with open("/home/mas/yuanyuan/workplace/persuasion_disentangle_p/data/output_test.txt",'r') as f:
            for line in f:
                result.append(float(line))

        p = []
        with open("/home/mas/yuanyuan/workplace/persuasion_disentangle_p/data/test_item_text.txt",'r') as f:
            for line in f:
                p.append(float(line.split(' ')[0]))

        rmse_test = np.sqrt(mean_squared_error(result,p))
        mae_test = mean_absolute_error(result,p)

        result=[]
        with open("/home/mas/yuanyuan/workplace/persuasion_disentangle_p/data/output_val.txt",'r') as f:
            for line in f:
                result.append(float(line))

        p = []
        with open("/home/mas/yuanyuan/workplace/persuasion_disentangle_p/data/val_item_text.txt",'r') as f:
            for line in f:
                p.append(float(line.split(' ')[0]))

        rmse_val = np.sqrt(mean_squared_error(result,p))
        mae_val = mean_absolute_error(result,p)

        print('rmse_test:%.4f,mae_test:%.4f,rmse_val:%.4f,mae_val:%.4f' % (rmse_test,mae_test,rmse_val,mae_val))
        



        
