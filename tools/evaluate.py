import numpy  as np

def calculate_mae_rmse(list1, list2):
    list1 = np.array(list1)
    list2 = np.array(list2)
    mae = np.mean(np.abs(list1 - list2))
    rmse = np.sqrt(np.mean((list1 - list2) ** 2))
    return mae, rmse