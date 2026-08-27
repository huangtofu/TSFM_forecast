import numpy as np
import pandas as pd


def save_forecast_to_csv(
    true_data,
    pred_median_data,
    pred_up_data,
    pred_down_data,
    variate_list,
    save_path="results/forecast_result.csv"
):
    true_data = np.asarray(true_data)
    pred_median_data = np.asarray(pred_median_data)
    pred_up_data = np.asarray(pred_up_data)
    pred_down_data = np.asarray(pred_down_data)
    if true_data.ndim != 2:
        raise ValueError(
            f"true_data 应该是二维数组 (变量数, 长度)，"
            f"当前形状为 {true_data.shape}"
        )
    if pred_median_data.shape != true_data.shape:
        raise ValueError(
            f"pred_median_data 形状 {pred_median_data.shape} "
            f"与 true_data {true_data.shape} 不一致"
        )
    if pred_up_data.shape != true_data.shape:
        raise ValueError(
            f"pred_up_data 形状 {pred_up_data.shape} "
            f"与 true_data {true_data.shape} 不一致"
        )
    if pred_down_data.shape != true_data.shape:
        raise ValueError(
            f"pred_down_data 形状 {pred_down_data.shape} "
            f"与 true_data {true_data.shape} 不一致"
        )
    n_variables, seq_len = true_data.shape
    if len(variate_list) != n_variables:
        raise ValueError(
            f"variate_list 有 {len(variate_list)} 个变量，"
            f"但数据中有 {n_variables} 个变量"
        )
    result_list = []
    for i, variable_name in enumerate(variate_list):
        temp_df = pd.DataFrame({
            "变量": variable_name,
            "时间点": np.arange(seq_len),
            "真实值": true_data[i],
            "预测中位数": pred_median_data[i],
            "预测上界": pred_up_data[i],
            "预测下界": pred_down_data[i]
        })
        result_list.append(temp_df)
    result_df = pd.concat(result_list, ignore_index=True)
    result_df.to_csv(
        save_path,
        index=False,
        encoding="utf-8-sig"
    )
    print(f"预测结果已保存到：{save_path}")
    return result_df
