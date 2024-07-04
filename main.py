import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='mdfend') #模型名称
parser.add_argument('--epoch', type=int, default=50) # 训练轮数
# parser.add_argument('--max_len', type=int, default=170) #weibo21可处理新闻最大长度
parser.add_argument('--max_len', type=int, default=500) #课题新数据集可处理新闻最大长度
parser.add_argument('--num_workers', type=int, default=4)#线程数
parser.add_argument('--early_stop', type=int, default=3)#早停
# parser.add_argument('--bert_vocab_file', default='./pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt')  #weibo21词表文件
parser.add_argument('--bert_vocab_file', default='./pretrained_model/bert-base-cased/vocab.txt') #课堂新数据集词表文件
parser.add_argument('--root_path', default='./data/gossip/')  #课堂新数据集根目录
# parser.add_argument('--root_path', default='./data/weibo21/') #weibo21数据集
# parser.add_argument('--bert', default='./pretrained_model/chinese_roberta_wwm_base_ext_pytorch') #weibo21 bert模型地址
parser.add_argument('--bert', default='./pretrained_model/bert-base-cased') #课堂新数据集 bert模型地址
parser.add_argument('--batchsize', type=int, default=64) #批处理大小
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--gpu', default='0')
parser.add_argument('--bert_emb_dim', type=int, default=768) #bert embeeding的维度
parser.add_argument('--w2v_emb_dim', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.0007) #学习率
parser.add_argument('--emb_type', default='bert')
parser.add_argument('--w2v_vocab_file', default='./pretrained_model/w2v/Tencent_AILab_Chinese_w2v_model.kv')
parser.add_argument('--save_param_dir', default= './param_model/gossip') #保存参数地址

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from run import Run
import torch
import numpy as np
import random

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if args.emb_type == 'bert':
    emb_dim = args.bert_emb_dim
    vocab_file = args.bert_vocab_file
elif args.emb_type == 'w2v':
    emb_dim = args.w2v_emb_dim
    vocab_file = args.w2v_vocab_file

print('lr: {}; model name: {}; emb_type: {}; batchsize: {}; epoch: {}; gpu: {}; emb_dim: {}'.format(args.lr, args.model_name, args.emb_type,  args.batchsize, args.epoch, args.gpu, emb_dim))


config = {
        'use_cuda': True,
        'batchsize': args.batchsize,
        'max_len': args.max_len,
        'early_stop': args.early_stop,
        'num_workers': args.num_workers,
        'vocab_file': vocab_file,
        'emb_type': args.emb_type,
        'bert': args.bert,
        'root_path': args.root_path,
        'weight_decay': 5e-5,
        'model':
            {
            'mlp': {'dims': [384], 'dropout': 0.2}
            },
        'emb_dim': emb_dim,
        'lr': args.lr,
        'epoch': args.epoch,
        'model_name': args.model_name,
        'seed': args.seed,
        'save_param_dir': args.save_param_dir
        }



if __name__ == '__main__':
    Run(config = config
        ).main()
