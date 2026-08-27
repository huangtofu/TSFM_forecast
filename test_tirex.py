import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
import pandas as pd
from load_model_function.pred_with_tirex import TirexFunction
from tools.result_save import save_forecast_to_csv

#原始数据地址
data=pd.read_excel(r"test_data\股票指数日度行情数据.xlsx")
#原始数据中选择进行预测的变量
#[开盘价	收盘价	最高价	最低价]
variate_list=["开盘价","收盘价"]
#历史步长（即使用历史的多少数据对未来进行预测）
context_len=64
#预测步长（即预测未来多少长度的数据）
pred_len=12
#选择模型的地址
model_path=r"models\Tirex-2"
#种子
seed=2026
#测试集长度（默认使用整个传入的数据集作为测试集，不过是排除掉context_len的剩下长度）
n=len(data)

data=data.loc[:,variate_list]
model=TirexFunction(data=data,
                    context_len=context_len,
                    pred_len=pred_len,
                    stride=pred_len,
                    model_path=model_path,
                    seed=seed)

true_data1,pred_median_data1,pred_up_data1,pred_down_data1=model.tirex_valid()
print(pred_median_data1.shape,pred_up_data1.shape,pred_down_data1.shape)

result_df = save_forecast_to_csv(
    true_data=true_data1,
    pred_median_data=pred_median_data1,
    pred_up_data=pred_up_data1,
    pred_down_data=pred_down_data1,
    variate_list=variate_list,
    save_path="results/forecast_result.csv"
)
#pred_median_data1,pred_up_data1,pred_down_data1=model.tirex_predict()
#print(pred_median_data1.shape,pred_up_data1.shape,pred_down_data1.shape)