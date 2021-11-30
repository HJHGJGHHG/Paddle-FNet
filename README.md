# Paddle FNet 复现文档
## 目标
&emsp;&emsp;基于PaddlePaddle复现FNet，在CoLA数据集上得到接近原论文ACC=78%的结果。
## 论文要点

## 源代码
&emsp;&emsp;参考：

## 复现过程
##### STEP1. 前向对齐
&emsp;&emsp;期望对齐模型结构，详见 /STEP1。

##### STEP2. 数据集、Tokenizer、评估指标对齐
&emsp;&emsp;实验数据承载自CoLA，期望对齐Tokenizer、Dataset、DataLoader、Metircs（ACC），详见 /STEP2。

##### STEP3. 损失函数对齐
&emsp;&emsp;期望对齐损失函数CrossEntropyLoss，详见 /STEP3。