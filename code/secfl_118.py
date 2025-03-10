# FedRA FOR 118BusSystem
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import sys
import csv
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from utils import ConfusionMatrix
from aggregation import ModelEncryptor, ModelAggre
import tenseal as ts
import time
import threading
from flpart import part
import torch.multiprocessing as mp

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
np.random.seed(0)

if __name__ == '__main__':
    N=3
    datafilename = [f'data_split/data118/data_user_{i}.csv' for i in range(1, N + 1)]
    parts = [part(n+1,datafilename[n],savefile='logs_FL_HE',dataset='118') for n in range(N)]                          
    for n in range(N):
        counter = parts[n].pathname()
                              
# FL_HE
    # 创建加密器对象  
    print('创建加密器对象ing')
    encryptor = ModelEncryptor(N)
    print('完成')    
    
    # 同时训练
    mp.set_start_method('spawn', force=True)
    for i in range(26):     
        for part in parts:
            part.set_commu(i)

        processes = []   
        for part in parts:
            p = mp.Process(target=part.train)
            p.start()
            processes.append(p)

        # 等待所有进程完成
        for p in processes:
            p.join()

        model_paths = [f'./logs_FL_HE/118/part{j}/118exp{counter}/final{i+1}.pth' for j in range(1, N+1)]
        print('进行-第%d轮' % (i+1))
        # 使用加密器聚合和平均模型参数
        print('加密、聚合和平均模型参数ing')
        averaged_params = encryptor.aggregate_encrypted_models(model_paths)
        print('完成')

        # 指定解密后模型的保存路径，并保存
        print('解密后模型并保存ing')
        save_path = f"./logs_FL_HE/118/pth/model_118exp{counter}_{i+1}.pth"
        encryptor.save_decrypted_model(averaged_params, save_path)
        print('完成-第%d轮' % (i+1))
