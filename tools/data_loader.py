from torch.utils.data import Dataset
import numpy as np
import torch
import random

def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Dataprocess(Dataset):
    def __init__(self,data,seq_len,pred_len,stride):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.data = data
        self.stride = stride

    def __len__(self):
        return (len(self.data)-self.seq_len-self.pred_len)//(self.stride)+1

    def __getitem__(self, idx):
        s_begin=idx * self.stride
        s_end=s_begin+self.seq_len
        p_begin=s_end
        p_end=s_end+self.pred_len
        if p_end > len(self.data):
            raise IndexError("Index out of range")
        back = self.data[s_begin:s_end].to_numpy()
        fore = self.data[p_begin:p_end].to_numpy()
        return back,fore

class FuturePredictor(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.data = data

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        s_begin = len(self.data) - self.seq_len
        s_end = len(self.data)
        back = self.data[s_begin:s_end].to_numpy()
        return back