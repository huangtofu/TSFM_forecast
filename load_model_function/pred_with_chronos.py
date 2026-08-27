import torch
from tools.data_loader import set_seed,Dataprocess,FuturePredictor
from torch.utils.data import DataLoader
import warnings
import tqdm
import numpy as np
from chronos import Chronos2Pipeline
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")
warnings.filterwarnings("ignore",message=r"Using `json`-module for json-handling\..*",category=UserWarning,module=r"gluonts\.json")

class ChronosFunction:
    def __init__(self,data,context_len,pred_len,stride,model_path,seed,quantile=[0.1,0.5,0.9]):
        self.data=data
        self.context_len=context_len
        self.pred_len=pred_len
        self.stride=stride
        self.model_path=model_path
        self.seed=seed
        self.quantile=quantile

    def chronos_valid(self):
        set_seed(self.seed)
        data=Dataprocess(self.data, seq_len=self.context_len,pred_len=self.pred_len,stride=self.stride)
        data1 = DataLoader(data, batch_size=1, shuffle=False)
        pipeline = Chronos2Pipeline.from_pretrained(self.model_path)
        pred_median_list = []
        pred_up_list=[]
        pred_down_list=[]
        true_list = []
        for x, y in tqdm.tqdm(data1):
            x=x.permute(0,2,1)
            y=y.permute(0,2,1)
            quantiles, mean = pipeline.predict_quantiles(
                inputs=x,
                prediction_length=self.pred_len,
                quantile_levels=self.quantile)
            pred_median_data=quantiles[0][:,:,1].cpu().numpy()
            pred_up_data=quantiles[0][:,:,2].cpu().numpy()
            pred_down_data=quantiles[0][:,:,0].cpu().numpy()
            true_data=y.squeeze(0).cpu().numpy()
            pred_median_list.append(pred_median_data)
            pred_up_list.append(pred_up_data)
            pred_down_list.append(pred_down_data)
            true_list.append(true_data)
        pred_median_data1=np.concatenate(pred_median_list,axis=1)
        pred_up_data1=np.concatenate(pred_up_list,axis=1)
        pred_down_data1=np.concatenate(pred_down_list,axis=1)
        true_data1=np.concatenate(true_list,axis=1)
        return true_data1,pred_median_data1,pred_up_data1,pred_down_data1

    def chronos_predict(self):
        set_seed(self.seed)
        data=FuturePredictor(data=self.data, seq_len=self.context_len,pred_len=self.pred_len)
        data1 = DataLoader(data, batch_size=1, shuffle=False)
        pipeline = Chronos2Pipeline.from_pretrained(self.model_path)
        for x in tqdm.tqdm(data1):
            x=x.permute(0,2,1)
            quantiles, mean = pipeline.predict_quantiles(
                inputs=x,
                prediction_length=self.pred_len,
                quantile_levels=self.quantile)
            pred_median_data=quantiles[0][:,:,1].cpu().numpy()
            pred_up_data=quantiles[0][:,:,2].cpu().numpy()
            pred_down_data=quantiles[0][:,:,0].cpu().numpy()
        return pred_median_data,pred_up_data,pred_down_data


class ChronosFunctionWithCovariate:
        def __init__(self,data,co_data,context_len,pred_len,stride,model_path,seed,quantile=[0.1,0.5,0.9]):
            self.data=data
            self.co_data=co_data
            self.context_len=context_len
            self.pred_len=pred_len
            self.stride=stride
            self.model_path=model_path
            self.seed=seed
            self.quantile=quantile

        def chronoswithcovariate_valid(self):
            set_seed(self.seed)
            data=Dataprocess(self.data, seq_len=self.context_len,pred_len=self.pred_len,stride=self.stride)
            co_data=Dataprocess(self.co_data, seq_len=self.context_len,pred_len=self.pred_len,stride=self.stride)
            data1 = DataLoader(data, batch_size=1, shuffle=False)
            data_co = DataLoader(co_data,batch_size=1, shuffle=False)
            pipeline = Chronos2Pipeline.from_pretrained(self.model_path)
            pred_median_list = []
            pred_up_list=[]
            pred_down_list=[]
            true_list = []
            for x1, x2 in tqdm.tqdm(zip(data1,data_co)):
                x=x1[0].squeeze(0).permute(1,0)
                y=x1[1].permute(0,2,1)
                covariates = x2[0].squeeze(0).permute(1,0)
                inputs=[{
                    "target":x,
                    "past_covariates": {
                        f"var_{i + 1}": covariates[i, :].reshape(-1)
                        for i in range(covariates.shape[0])
                        }
                }]
                quantiles, mean = pipeline.predict_quantiles(
                                    inputs=inputs,
                                    prediction_length=self.pred_len,
                                    quantile_levels=self.quantile)
                pred_median_data=quantiles[0][:,:,1].cpu().numpy()
                pred_up_data=quantiles[0][:,:,2].cpu().numpy()
                pred_down_data=quantiles[0][:,:,0].cpu().numpy()
                true_data=y.squeeze(0).cpu().numpy()
                pred_median_list.append(pred_median_data)
                pred_up_list.append(pred_up_data)
                pred_down_list.append(pred_down_data)
                true_list.append(true_data)
            pred_median_data1=np.concatenate(pred_median_list,axis=1)
            pred_up_data1=np.concatenate(pred_up_list,axis=1)
            pred_down_data1=np.concatenate(pred_down_list,axis=1)
            true_data1=np.concatenate(true_list,axis=1)
            return true_data1,pred_median_data1,pred_up_data1,pred_down_data1

        def chronoswithcovariate_predict(self):
            data=FuturePredictor(self.data, seq_len=self.context_len,pred_len=self.pred_len)
            co_data=FuturePredictor(self.co_data, seq_len=self.context_len,pred_len=self.pred_len)
            data1 = DataLoader(data, batch_size=1, shuffle=False)
            data_co = DataLoader(co_data,batch_size=1, shuffle=False)
            pipeline = Chronos2Pipeline.from_pretrained(self.model_path)
            for x1, x2 in tqdm.tqdm(zip(data1,data_co)):
                x=x1.squeeze(0).permute(1,0)
                covariates = x2.squeeze(0).permute(1,0)
                inputs=[{
                    "target":x,
                    "past_covariates": {
                        f"var_{i + 1}": covariates[i, :].reshape(-1)
                        for i in range(covariates.shape[0])
                        }
                }]
                quantiles, mean = pipeline.predict_quantiles(
                                    inputs=inputs,
                                    prediction_length=self.pred_len,
                                    quantile_levels=self.quantile)
                pred_median_data=quantiles[0][:,:,1].cpu().numpy()
                pred_up_data=quantiles[0][:,:,2].cpu().numpy()
                pred_down_data=quantiles[0][:,:,0].cpu().numpy()
            return pred_median_data,pred_up_data,pred_down_data