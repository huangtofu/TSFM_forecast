# TSFM Forecasting

## 一、项目简介

本项目面向**时间序列基础模型（Time Series Foundation Models, TSFM）**的预测任务，主要集成并对比以下三类时间序列基础模型：

- **Chronos-2**
- **Toto-2.0**
- **TiRex-2**

项目的目标是将不同的时间序列基础模型封装为统一、易于调用的预测接口，使用户可以基于自己的时间序列数据快速完成：

1. 模型加载
2. 历史数据构造
3. 多变量时间序列预测
4. 基于协变量的预测
5. 点预测与预测区间获取
6. 验证集预测
7. 未来数据预测
8. 预测结果保存与后续分析

整体项目结构强调**统一调用方式、模型可替换以及预测结果标准化**，方便后续进行不同时间序列基础模型之间的效果比较。

---

## 二、项目支持的模型

### 1. Chronos-2

Chronos-2 是本项目当前 Notebook 中已经提供完整调用示例的时间序列基础模型。

模型下载地址：

[Chronos-2](https://huggingface.co/amazon/chronos-2)

项目中支持两种使用方式：

- 使用协变量（Covariates）
- 不使用协变量

其中，使用协变量时，可以同时输入目标时间序列和辅助变量，为预测提供额外的信息。

---

### 2. Toto-2.0

Toto-2.0 是本项目集成的另一类时间序列基础模型。

模型下载地址：

[Toto-2.0-22m](https://huggingface.co/Datadog/Toto-2.0-22m)

Toto-2.0 提供不同参数规模的模型版本，可以根据实际硬件资源和预测任务需求选择合适的模型。

---

### 3. TiRex-2

TiRex-2 是本项目集成的第三类时间序列基础模型。

模型下载地址：

[TiRex-2](https://huggingface.co/NX-AI/TiRex-2)

---

## 三、项目目录结构

下载模型后，建议统一放置在项目的 `models` 目录下。

```text
TSFM_forecasting/
│
├── models/
│   ├── Chronos-2/
│   ├── Toto-2.0-22m/
│   └── TiRex-2/
│
├── load_model_function/
│   ├── pred_with_chronos.py
│   ├── pred_with_toto.py
│   └── ...
│
├── tools/
│   ├── result_save.py
│   └── ...
│
├── test_data/
│   ├── 股票指数日度行情数据.xlsx
│   └── 宏观经济指标日度数据.xlsx
│
├── requirements.txt
└── tutorial.ipynb
```

不同模型的具体封装文件以项目实际目录为准。

---

## 四、环境配置

首先安装项目所需依赖：

```bash
pip install -r requirements.txt
```

建议根据实际 GPU、CUDA、PyTorch 以及各模型官方要求配置运行环境。

模型文件下载完成后，将对应模型放入：

```text
models/
```

目录中，并在代码中通过 `model_path` 指定模型路径。

---

## 五、预测任务基本流程

项目整体预测流程可以概括为：

```text
原始时间序列数据
        │
        ├───────────────┐
        │               │
        ▼               ▼
  目标时间序列       协变量数据
        │               │
        └───────┬───────┘
                ▼
          数据选择与整理
                │
                ▼
        设置历史窗口长度
          context_len
                │
                ▼
        设置预测窗口长度
           pred_len
                │
                ▼
          加载 TSFM 模型
                │
                ▼
        ┌───────┴────────┐
        │                │
        ▼                ▼
      验证预测          未来预测
        │                │
        ▼                ▼
  真实值/预测值      未来预测结果
  /预测区间
        │                │
        └───────┬────────┘
                ▼
          结果保存与分析
```

---

## 六、Chronos-2 使用示例

### 6.1 使用协变量进行预测

当前 Notebook 中提供了 Chronos-2 使用协变量的完整示例。

首先读取目标时间序列和协变量数据：

```python
import pandas as pd

from load_model_function.pred_with_chronos import ChronosFunctionWithCovariate
from tools.result_save import save_forecast_to_csv

# 原始数据
data = pd.read_excel("test_data\\股票指数日度行情数据.xlsx")

# 协变量数据
co_data = pd.read_excel("test_data\\宏观经济指标日度数据.xlsx")
```

然后选择需要预测的变量：

```python
variate_list = ["开盘价", "收盘价"]
```

选择用于辅助预测的协变量：

```python
co_variate_list = [
    "GDP同比(%)",
    "CPI同比(%)",
    "M2同比(%)",
    "失业率(%)"
]
```

其中：

- `variate_list`：需要进行预测的目标变量
- `co_variate_list`：辅助预测的协变量

---

### 6.2 设置预测参数

项目通过 `context_len` 和 `pred_len` 控制预测窗口：

```python
context_len = 64
pred_len = 12
```

含义分别为：

| 参数 | 含义 |
|---|---|
| `context_len` | 用于预测的历史数据长度 |
| `pred_len` | 每次预测未来的数据长度 |
| `stride` | 滑动窗口的步长 |
| `seed` | 随机种子 |
| `model_path` | 模型文件路径 |

当前 Notebook 中使用：

```python
model_path = r"models\Chronos-2"
seed = 2026
```

并将：

```python
stride = pred_len
```

设置为预测窗口长度。

---

### 6.3 创建模型对象

```python
model = ChronosFunctionWithCovariate(
    data=data,
    co_data=co_data,
    context_len=context_len,
    pred_len=pred_len,
    stride=pred_len,
    model_path=model_path,
    seed=seed
)
```

模型对象创建后，可以分别执行验证和未来预测。

---

### 6.4 验证集预测

```python
true_data1, pred_median_data1, pred_up_data1, pred_down_data1 = \
    model.chronoswithcovariate_valid()
```

返回结果包括：

- `true_data1`：真实值
- `pred_median_data1`：预测中位数
- `pred_up_data1`：预测区间上界
- `pred_down_data1`：预测区间下界

Notebook 中输出的结果形状示例为：

```text
(2, 36)
(2, 36)
(2, 36)
(2, 36)
```

说明当前示例包含 2 个预测变量。

---

### 6.5 未来数据预测

完成验证后，可以直接进行未来预测：

```python
pred_median_data, pred_up_data, pred_down_data = \
    model.chronoswithcovariate_predict()
```

返回：

- `pred_median_data`：未来预测中位数
- `pred_up_data`：未来预测区间上界
- `pred_down_data`：未来预测区间下界

当前 Notebook 中的示例输出形状为：

```text
(2, 12)
(2, 12)
(2, 12)
```

其中 `12` 对应设置的 `pred_len=12`。

---

## 七、Chronos-2 不使用协变量

除了协变量预测之外，项目还支持仅使用目标时间序列进行预测。

调用方式为：

```python
from load_model_function.pred_with_chronos import ChronosFunction
```

读取目标数据：

```python
data = pd.read_excel("test_data\\股票指数日度行情数据.xlsx")
```

选择预测变量：

```python
variate_list = ["开盘价", "收盘价"]
```

设置预测参数：

```python
context_len = 64
pred_len = 12
model_path = r"models\Chronos-2"
seed = 2026
```

创建模型：

```python
model = ChronosFunction(
    data=data,
    context_len=context_len,
    pred_len=pred_len,
    stride=pred_len,
    model_path=model_path,
    seed=seed
)
```

---

## 八、验证与未来预测

不使用协变量时，验证过程为：

```python
true_data1, pred_median_data1, pred_up_data1, pred_down_data1 = \
    model.chronos_valid()
```

未来预测过程为：

```python
pred_median_data, pred_up_data, pred_down_data = \
    model.chronos_predict()
```

因此，无论是否使用协变量，项目都尽量保持类似的调用逻辑：

```text
创建模型
   │
   ├── valid()
   │      └── 验证集预测
   │
   └── predict()
          └── 未来预测
```

这种封装方式可以降低不同模型之间的调用差异，方便后续进行模型替换和效果比较。

---

## 九、预测结果

项目预测结果主要包括两类信息：

### 1. 点预测

模型输出预测结果的中心估计值，例如：

```python
pred_median_data
```

可以用于：

- 后续业务预测
- 趋势分析
- 预测值与真实值比较
- 预测误差计算

### 2. 预测区间

项目同时保留预测区间信息：

```python
pred_up_data
pred_down_data
```

因此可以进一步构建：

```text
          预测上界
             │
             │
历史数据 ────┼──── 预测中位数
             │
             │
          预测下界
```

相比只输出单一预测值，这种结果形式能够提供更多的不确定性信息。

---

## 十、统一模型对比

本项目的一个核心目标，是在相同或尽可能一致的数据处理与预测设置下，对比不同时间序列基础模型的预测能力。

建议统一以下实验条件：

| 实验条件 | 设置 |
|---|---|
| 数据集 | 相同 |
| 目标变量 | 相同 |
| 历史窗口 | 相同 |
| 预测窗口 | 相同 |
| 数据预处理 | 尽可能保持一致 |
| 测试集 | 相同 |
| 评价指标 | 相同 |
| 随机种子 | 尽可能保持一致 |

最终可以形成：

```text
                    时间序列数据
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
      Chronos-2       Toto-2.0       TiRex-2
          │              │              │
          ▼              ▼              ▼
       预测结果       预测结果        预测结果
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                    统一评价指标
                         │
                         ▼
                    模型效果对比
```

这样可以从预测精度、稳定性以及预测区间表现等多个角度分析不同 TSFM 的适用性。

---

## 十一、数据输入要求

根据当前 Notebook 的使用方式，输入数据至少需要满足以下条件：

### 目标时间序列

目标数据通过 `pandas` 读取，例如：

```python
data = pd.read_excel("test_data\\股票指数日度行情数据.xlsx")
```

然后从中选择需要预测的变量：

```python
data = data.loc[:, variate_list]
```

### 协变量

如果使用协变量，则额外提供：

```python
co_data = pd.read_excel("test_data\\宏观经济指标日度数据.xlsx")
```

并选择：

```python
co_data = co_data.loc[:, co_variate_list]
```

使用协变量时，应确保目标数据与协变量数据在时间维度上能够正确对应。

---

## 十二、项目使用建议

实际使用时，可以按照以下顺序进行：

### Step 1：安装依赖

```bash
pip install -r requirements.txt
```

### Step 2：下载模型

分别下载：

- Chronos-2
- Toto-2.0
- TiRex-2

### Step 3：放置模型

统一放入：

```text
models/
```

### Step 4：准备数据

准备目标时间序列数据。

如果需要使用协变量，则同时准备协变量数据。

### Step 5：设置预测参数

例如：

```python
context_len = 64
pred_len = 12
```

### Step 6：选择模型

指定对应的模型路径。

### Step 7：执行验证

首先在已有数据上进行验证，获得：

```text
真实值
预测值
预测区间
```

### Step 8：执行未来预测

使用模型预测未来 `pred_len` 个时间点。

### Step 9：保存预测结果

项目提供结果保存工具：

```python
from tools.result_save import save_forecast_to_csv
```

可进一步将预测结果保存为 CSV，用于后续分析和可视化。

---

## 十三、项目特点

### 统一接口

通过模型封装函数隐藏不同基础模型底层调用细节，使上层代码更加简洁。

### 支持多变量预测

当前 Notebook 示例同时预测：

```python
["开盘价", "收盘价"]
```

因此可以处理多个目标变量。

### 支持协变量

Chronos-2 示例支持额外输入宏观经济指标等协变量：

```text
GDP
CPI
M2
失业率
```

用于辅助时间序列预测。

### 支持预测区间

除了预测中位数，还可以获得预测上下界，为后续不确定性分析提供基础。

### 支持验证与未来预测

将模型调用划分为：

```text
验证预测
未来预测
```

便于分别进行模型评估和实际应用。

### 便于模型横向比较

Chronos、Toto 和 TiRex 可以作为统一 TSFM forecasting 项目中的不同模型后端，方便后续进行系统性实验。

---

## 十四、后续扩展方向

在当前项目基础上，可以进一步扩展：

1. **Chronos / Toto / TiRex 统一 API**
2. **统一预测结果格式**
3. **自动计算 RMSE、MAE、MAPE、R² 等指标**
4. **预测结果可视化**
5. **预测区间可视化**
6. **不同模型自动批量测试**
7. **不同 `context_len` 与 `pred_len` 的实验**
8. **多数据集批量评估**
9. **模型推理速度与显存占用对比**
10. **最终形成 TSFM Benchmark**

最终可以形成一个完整的时间序列基础模型实验框架：

```text
                    TSFM Forecasting
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
       Chronos-2        Toto-2.0         TiRex-2
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                    统一预测接口
                           │
                  ┌────────┴────────┐
                  ▼                 ▼
               验证预测            未来预测
                  │                 │
                  └────────┬────────┘
                           ▼
                     结果标准化
                           │
                           ▼
                     指标计算/可视化
                           │
                           ▼
                      模型效果对比
```

---

## 十五、说明

当前 `tutorial.ipynb` 主要展示了 **Chronos-2** 的两种调用方式：

1. 使用协变量进行预测
2. 不使用协变量进行预测

Notebook 中展示了模型下载、依赖安装、数据读取、变量选择、预测参数配置、模型初始化、验证预测以及未来预测等完整流程。

Toto-2.0 和 TiRex-2 属于本项目整体支持的模型，具体调用方式应以项目中对应的模型封装代码为准。

