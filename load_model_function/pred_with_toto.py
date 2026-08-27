import torch
from toto2 import Toto2Model
from tools.data_loader import set_seed,Dataprocess,FuturePredictor
from torch.utils.data import DataLoader
import warnings
import tqdm
import numpy as np
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")
warnings.filterwarnings("ignore",message=r"Using `json`-module for json-handling\..*",category=UserWarning,module=r"gluonts\.json")
warnings.filterwarnings('ignore', category=UserWarning, module='gluonts')

class TotoFunction:
    def __init__(self, data, context_len, pred_len, stride, model_path, seed, quantile=[0,8]):
        self.data = data
        self.context_len = context_len
        self.pred_len = pred_len
        self.stride = stride
        self.model_path = model_path
        self.seed = seed
        self.quantile = quantile

    def toto_valid(self):
        set_seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data0 = Dataprocess(self.data, self.context_len, self.pred_len,stride=self.stride)
        data1 = DataLoader(data0, batch_size=1, shuffle=False)
        model = Toto2Model.from_pretrained(self.model_path)
        model = model.to(device).eval()
        pred_list_median=[]
        pred_list_up=[]
        pred_list_down=[]
        true_list=[]
        for x,y in tqdm.tqdm(data1):
            x=x.permute(0,2,1).to(device).float() 
            n_vars=x.shape[1]
            y=y.permute(0,2,1).to(device)
            target_mask = torch.ones_like(x, dtype=torch.bool)
            series_ids = torch.zeros(1, n_vars, dtype=torch.long, device=device)
            quantiles = model.forecast(
                        {"target": x, "target_mask": target_mask, "series_ids": series_ids},
                        horizon=self.pred_len,
                        decode_block_size=768,
                         has_missing_values=False,
                        )
            pred_median=quantiles[4].squeeze(0).cpu().numpy()
            pred_up=quantiles[self.quantile[1]].squeeze(0).cpu().numpy()
            pred_down=quantiles[self.quantile[0]].squeeze(0).cpu().numpy()
            true_data1=y.squeeze(0).cpu().numpy()
            pred_list_median.append(pred_median)
            pred_list_up.append(pred_up)
            pred_list_down.append(pred_down)
            true_list.append(true_data1)
        pred_data_median=np.concatenate(pred_list_median, axis=1)
        pred_data_up=np.concatenate(pred_list_up, axis=1)
        pred_data_down=np.concatenate(pred_list_down, axis=1)
        true_data=np.concatenate(true_list, axis=1)
        return true_data,pred_data_median,pred_data_up,pred_data_down

    def toto_predict(self):
        set_seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data0 = FuturePredictor(self.data, self.context_len, self.pred_len)
        data1 = DataLoader(data0, batch_size=1, shuffle=False)
        model = Toto2Model.from_pretrained(self.model_path)
        model = model.to(device).eval()
        for x in tqdm.tqdm(data1):
            x=x.permute(0,2,1).to(device).float() 
            n_vars=x.shape[1]
            target_mask = torch.ones_like(x, dtype=torch.bool)
            series_ids = torch.zeros(1, n_vars, dtype=torch.long, device=device)
            quantiles = model.forecast(
                                {"target": x, "target_mask": target_mask, "series_ids": series_ids},
                                horizon=self.pred_len,
                                decode_block_size=768,
                                 has_missing_values=False,
                                )
            pred_median=quantiles[4].squeeze(0).cpu().numpy()
            pred_up=quantiles[self.quantile[1]].squeeze(0).cpu().numpy()
            pred_down=quantiles[self.quantile[0]].squeeze(0).cpu().numpy()
        return pred_median,pred_up,pred_down


