# Paddle FNet 复现文档
## 目标
&emsp;&emsp;基于PaddlePaddle复现FNet，在CoLA数据集上得到接近原论文ACC=78%的结果。
## 论文要点

## 复现过程
##### STEP0. 权重转换
&emsp;&emsp;Huggingface上的：[FNet-base](https://huggingface.co/google/fnet-base) 、[FNet-large ](https://huggingface.co/google/fnet-large)   
&emsp;&emsp;转换脚本：https://github.com/HJHGJGHHG/Paddle-FNet/blob/main/torch2paddle.py  
&emsp;&emsp;也可以用我传到HF上的模型：[Paddle-FNet-base](https://huggingface.co/google/fnet-base) 、[Paddle-FNet-large](https://huggingface.co/google/fnet-large)  

##### STEP1. 前向对齐
&emsp;&emsp;期望对齐模型结构，详见 /STEP1。  

##### STEP2. 数据集、Tokenizer、评估指标对齐
&emsp;&emsp;实验数据承载自CoLA，期望对齐Tokenizer、Dataset、DataLoader、Metircs（ACC），详见 /STEP2。  

##### STEP3. 损失函数对齐
&emsp;&emsp;期望对齐损失函数CrossEntropyLoss，详见 /STEP3。  

##### STEP4.反向对齐
&emsp;&emsp;在固定随机量（如Dropout）的情况下期望对齐Loss反向传播过程，详见 /STEP4。

##### STEP5.训练对齐
&emsp;&emsp;完成STEP0~STEP4后，在CoLA上完成训练，并对比最终结果。  

#### 复现结果与出现的**一些问题**见 /STEP5/README！！
