import os
import setproctitle
import argparse
from Trainer import *
from Tester import *
import torch
import random
import numpy as np
from config import Config
from absl import logging
import absl
from absl import app as app_google

def parse_args():
    parser = argparse.ArgumentParser(description="Run persuasion")
    parser.add_argument('--without_un', type=int,nargs='?', default=0,
                        help='without_un or not')
    parser.add_argument('--gpu', type=str, default='1',
                        help='GPU.')
    parser.add_argument('--step_num', type=int, default=500,
                        help='step num for train')
    parser.add_argument('--epoch_num', type=int, default=10,
                        help='epoch num for early stop')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='number of folds for cross validation')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size in dataloader')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout')
    parser.add_argument('--w1', type=float, default=0.5,
                        help='w1')
    parser.add_argument('--w2', type=float, default=1.0,
                        help='w2')
    parser.add_argument('--w3', type=float, default=0.0,
                        help='w3')
    parser.add_argument('--weight_decay', type=float, default=1e-7,
                        help='weight_decay')
    parser.add_argument('--seed', type=int, default=100,
                        help='random split seed')
    parser.add_argument('--folder_file', type=str, default='test',
                        help='model saved')
    parser.add_argument('--vocab', type=int, default=0,
                        help='jieba cut words or not')
    parser.add_argument('--patience', type=int, default=5,
                        help='patience to lr decay')
    parser.add_argument('--log_file', type=str, default='test',
                        help='the filename of logging')
    parser.add_argument('--hidden_size', type=int, default=32,
                        help='hidden_size for GRU')
    parser.add_argument('--item_size', type=int, default=8,
                        help='hidden_size for GRU')
    parser.add_argument('--hidden_k', type=int, default=8,
                        help='times')
    parser.add_argument('--unlabel_pro', type=float, default=0.25,
                        help='proportion of unlabeled data')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='the minimum lr')
    parser.add_argument('--test', type=int, default=1,
                        help='if the mode is testing')

    
    return parser.parse_args()

def seed_set():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.autograd.set_detect_anomaly(True)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True  # 每次训练得到相同结果
    torch.backends.cudnn.benchmark = False

args = parse_args()
args.epoch_num=5

# gpu selection
args.gpu = '0'

setproctitle.setproctitle("run")
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)


def set_log(exp_name):
    logging.flush()
    logging.get_absl_handler().use_absl_log_file(exp_name, '/home/mas/yuanyuan/workplace/persuasion_disentangle_p/log/')
    logging.get_absl_handler().setFormatter(None)


def args_init():
    args.weight_decay=1e-6
    args.hidden_k = 4
    args.item_size = 8*args.hidden_k
    args.hidden_size = 16*args.hidden_k
    args.lr=0.01
    args.w1 = 0.5
    args.without_un = 0
    args.w3=1e-3
    args.w2=1.0
    args.test=0
    args.unlabel_pro=1
    args.batch_size = 64
    args.dropout = 0.0

def main(argv):
    args_init()
    seed_set()
    log_file = 'model:{},dropout:{},lr:{},weight_decay:{},batch_size:{},w2:{},w3:{},k:{},unlabel_pro:{}'.format(args.model,args.dropout,args.lr,args.weight_decay,args.batch_size,args.w2,args.w3,args.hidden_k,args.unlabel_pro)
    set_log(log_file)
    app = Trainer(args,logging)
    app.trainstep()

if __name__ == '__main__':
    app_google.run(main)
