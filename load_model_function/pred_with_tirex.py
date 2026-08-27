import torch
from tirex2 import ForecastModel, TimeseriesType, load_model
from tools.data_loader import set_seed,Dataprocess,FuturePredictor
from torch.utils.data import DataLoader
import tqdm
import numpy as np

class TirexFunction:
    def __init__(self, data, context_len, pred_len, stride, model_path, seed, quantile=[0,8]):
        self.data=data
        self.context_len=context_len
        self.pred_len=pred_len
        self.stride=stride
        self.model_path=model_path
        self.seed=seed
        self.quantile=quantile

    def tirex_valid(self):
        set_seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data0 = Dataprocess(self.data, self.context_len, self.pred_len,self.stride)
        data1 = DataLoader(data0, batch_size=1, shuffle=False)
        pred_list_median=[]
        pred_list_up=[]
        pred_list_down=[]
        true_list=[]
        for x,y in tqdm.tqdm(data1):
            x=x.squeeze(0).permute(1,0).to(device).float()
            y=y.squeeze(0).permute(1,0).to(device)
            model: ForecastModel = load_model(self.model_path, device=str(device))
            ts_data=TimeseriesType(target=x.cpu(),
                                    past_covariates=None,
                                    future_covariates=None,)
            forecasts=model.forecast(timeseries=[ts_data],
                                     prediction_length=self.pred_len,
                                     )
            pred_list_median.append(forecasts[0][:,4,:].cpu().numpy())
            pred_list_up.append(forecasts[0][:,self.quantile[1],:].cpu().numpy())
            pred_list_down.append(forecasts[0][:,self.quantile[0],:].cpu().numpy())
            true_list.append(y.cpu().numpy())
        pred_median_data=np.concatenate(pred_list_median, axis=1)
        pred_up_data=np.concatenate(pred_list_up, axis=1)
        pred_down_data=np.concatenate(pred_list_down, axis=1)
        true_data=np.concatenate(true_list, axis=1)
        return true_data,pred_median_data,pred_up_data,pred_down_data

    def tirex_predict(self):
        set_seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data0 = FuturePredictor(self.data, self.context_len, self.pred_len)
        data1 = DataLoader(data0, batch_size=1, shuffle=False)
        for x in tqdm.tqdm(data1):
            x=x.squeeze(0).permute(1,0).to(device).float()
            model: ForecastModel = load_model(self.model_path, device=str(device))
            ts_data=TimeseriesType(target=x.cpu(),
                                    past_covariates=None,
                                    future_covariates=None,)
            forecasts=model.forecast(timeseries=[ts_data],
                                     prediction_length=self.pred_len,
                                     )
            pred_median_data=forecasts[0][:,4,:].cpu().numpy()
            pred_up_data=forecasts[0][:,self.quantile[1],:].cpu().numpy()
            pred_down_data=forecasts[0][:,self.quantile[0],:].cpu().numpy()
        return pred_median_data,pred_up_data,pred_down_data
