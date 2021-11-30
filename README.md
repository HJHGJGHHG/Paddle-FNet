# Paddle FNet 复现文档
## 目标
&emsp;&emsp;基于PaddlePaddle复现FNet，在CoLA数据集上得到接近原论文ACC=78%的结果。
## 论文要点

## 原代码
&emsp;&emsp;参考：[HuggingFace](https://github.com/huggingface/transformers/tree/master/src/transformers/models/fnet)
&emsp;&emsp;跑通原代码：
```
cd FNet  # 跑通原代码相关文件在 /FNet中
python Preprocessor.py  # 或直接在test.ipynb中查看结果
```
&emsp;&emsp;结果：FNet-large在CoLA上FineTune：ACC= ***0.69*** ！  
&emsp;&emsp;**为啥自己跑的原代码与论文结果差别这么大！！！**  
&emsp;&emsp;原论文只给出了 ***JAX*** 版本的checkpoint，转换脚本：https://github.com/erksch/fnet-pytorch/blob/master/convert_jax_checkpoint.py   
&emsp;&emsp;至于为什么差异这么大，个人猜想：

1. 原论文预训练在 TPU 上完成，相关算子或有差异；  
2. 超参未知。在Huggingface官方Pytorch模型的文档（https://huggingface.co/google/fnet-base/tree/main/ ）中有如下论述：
> Note that the training hyperparameters of the reproduced models were not the same as the official model, so the performance may differ significantly for some tasks (for example: CoLA). 

<center><img src="https://github.com/HJHGJGHHG/Paddle-FNet/blob/main/img/1.png"  style="zoom:30%;" width="70%"/></center>

&emsp;&emsp;不过个人调过许多次参，也试着用过Kaggle上的TPU，ACC均为超过0.7……原因还是未知  
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