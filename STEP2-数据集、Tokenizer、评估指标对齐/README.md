# STEP2-数据集、Tokenizer、评估指标对齐
## 文件结构
&emsp;&emsp;文件结构如下：
```
D:.
    accuracy.py  # Torch的评估指标（Accuracy）
    compare_dataset.py  # 数据集对齐
    compare_metric.py  # 评估指标对齐
    data_diff.log  # 数据集误差
    generate_metric.py  # 生成评估指标数据
    metric_diff.log  # 评估指标误差
    metric_paddle.npy
    metric_torch.npy
    modeling.py  # Paddle实现的FNet
    README.md
    tokenizer.py  # Paddle实现的FNetTokenizer
```

## 数据集对齐
&emsp;&emsp;现欲测试torch与paddle的Dataloader是否能迭代得到相同的数据，数据承载自CoLA数据集。
``` python
cd STEP2-数据集、Tokenizer、评估指标对齐
# 对比生成log
python compare_dataset.py
```
&emsp;&emsp;我们对比了前五个Batch的input ids、token type id与label，相关误差在data_diff.log中查看：
```
[2021/11/30 22:22:23] root INFO: length: 
[2021/11/30 22:22:23] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/11/30 22:22:23] root INFO: dataloader_0_0: 
[2021/11/30 22:22:23] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/11/30 22:22:23] root INFO: dataloader_0_1: 
[2021/11/30 22:22:23] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/11/30 22:22:23] root INFO: dataloader_0_2: 
[2021/11/30 22:22:23] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/11/30 22:22:23] root INFO: dataloader_1_0: 
[2021/11/30 22:22:23] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/11/30 22:22:23] root INFO: dataloader_1_1: 
[2021/11/30 22:22:23] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/11/30 22:22:23] root INFO: dataloader_1_2: 
[2021/11/30 22:22:23] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/11/30 22:22:23] root INFO: dataloader_2_0: 
[2021/11/30 22:22:23] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/11/30 22:22:23] root INFO: dataloader_2_1: 
[2021/11/30 22:22:23] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/11/30 22:22:23] root INFO: dataloader_2_2: 
[2021/11/30 22:22:23] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/11/30 22:22:23] root INFO: dataloader_3_0: 
[2021/11/30 22:22:23] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/11/30 22:22:23] root INFO: dataloader_3_1: 
[2021/11/30 22:22:23] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/11/30 22:22:23] root INFO: dataloader_3_2: 
[2021/11/30 22:22:23] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/11/30 22:22:23] root INFO: dataloader_4_0: 
[2021/11/30 22:22:23] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/11/30 22:22:23] root INFO: dataloader_4_1: 
[2021/11/30 22:22:23] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/11/30 22:22:23] root INFO: dataloader_4_2: 
[2021/11/30 22:22:23] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/11/30 22:22:23] root INFO: diff check passed
```
&emsp;&emsp;可见两个Dataloader得到的数据完全一致。

## 评估指标对齐
``` python
cd STEP2-数据集、Tokenizer、评估指标对齐
# 生成评估指标数据
python generate_metric.py
# 生成误差log
Python compare_metric.py
```
&emsp;&emsp;可见两个ACC算子得到的结果完全一致。
``` python
[2021/11/30 22:28:04] root INFO: accuracy: 
[2021/11/30 22:28:04] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/11/30 22:28:04] root INFO: diff check passed
```