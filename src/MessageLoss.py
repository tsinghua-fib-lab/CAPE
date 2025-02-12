import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MessageLoss(nn.Module):
    def __init__(self, Config, args):
        super(MessageLoss, self).__init__()
        self.Config = Config
        self.args = args
        self.w1 = args.w1
        self.w2 = args.w2
        self.w3 = args.w3
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        
    def forward(self, labeled_doc = None, target1 = None, labeled_sent = None, target2 = None, unlabeled_doc = None, target3 = None, persuasion_embed = None, w1=None, mode = 'None'):

        # persuasion_embed: batch_size * feature_size * hidden_size
        
        if w1 is not None:
            self.w1 = w1

        if labeled_doc is not None:
            labeled_doc = labeled_doc.squeeze(1)
            labeled_doc_loss = self.mae(labeled_doc, target1)
        else:
            labeled_doc_loss = 0
            
        if unlabeled_doc is not None:
            unlabeled_doc = unlabeled_doc.squeeze(1)
            unlabeled_doc_loss = self.mae(unlabeled_doc, target3)
        else:
            unlabeled_doc_loss = 0
        
        labeled_sent_loss,count = 0,0

        y_true = []
        y_pred = []

        # covariance_loss
        p_avg = torch.mean(persuasion_embed,dim=2).unsqueeze(dim=2).repeat(1,1,persuasion_embed.shape[2])
        C = 1/self.args.hidden_size*(persuasion_embed-p_avg).matmul((persuasion_embed-p_avg).permute(0,2,1))

        C_f = torch.norm(C+1e-9, p='fro', dim=(1,2))

        diag = torch.eye(self.Config.feature_size).unsqueeze(dim=0).repeat(C_f.shape[0],1,1).cuda()
        diag_C = C.mul(diag)
        C_diag_f = torch.norm(diag_C+1e-9, p='fro', dim=(1,2))

        covariance_loss = torch.mean(1/2*(C_f**2-C_diag_f**2))

        #covariance_loss = labeled_doc_loss
        
        if target2 is not None:
            target2 = target2.view(target2.shape[0] * target2.shape[1])
            labeled_sent1 = F.log_softmax(labeled_sent, dim = 1)
            labeled_sent2 = torch.argmax(F.softmax(labeled_sent, dim = 1), dim = 1)
            
            for i in range(0, target2.shape[0]):
                if target2[i]>=self.Config.feature_size:
                    continue
                else:
                    y_true.append(target2[i].item())
                    y_pred.append(labeled_sent2[i].item())
                    count += 1
                    labeled_sent_loss += (-1 * labeled_sent1[i][target2[i].item()])
            
            if count != 0:
                labeled_sent_loss = labeled_sent_loss / count 

        if mode == 'train':

            loss = self.w1 * labeled_doc_loss + self.w1 * unlabeled_doc_loss + self.w2 * labeled_sent_loss + self.w3 * covariance_loss
            
        else:
            loss = self.w1 * labeled_doc_loss + self.w2 * labeled_sent_loss + self.w3 * covariance_loss
            
        return loss, labeled_doc_loss, unlabeled_doc_loss,labeled_sent_loss, covariance_loss, y_true,y_pred
