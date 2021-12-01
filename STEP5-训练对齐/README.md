# STEP5-训练对齐
&emsp;&emsp;在本节中，我们期望对齐训练过程并最终完成复现。  
## 训练数据对齐
&emsp;&emsp;该部分见STEP2。

## 训练对齐
&emsp;&emsp;完成上述所有步骤后，我们期望对齐训练过程与结果，并最终完成复现。在训练对齐过程中，受到较多随机量的影响，精度有少量diff是正常的，diff在0.15%以内可以认为是正常的。  
&emsp;&emsp;本部分数据承载自 CoLA（GLUE）。

&emsp;&emsp;相关参数：（单机单卡单精）
* LR = 2e-5
* batch_size = 16
* seed = 1234
* optimizer = AdamW
* LR-sheduler = Linear （Warmup Steps: 0）
* epochs = 2

&emsp;&emsp;结果对比  
&emsp;&emsp;预训练模型取fnet-large。在CoLA validation数据集上有：

|  评价指标 | 原论文 | Transformers实现 | Paddle复现 |
| ----- | ----- | ----- | ----- |
| ACC | 78% |0.6912751678 | 0.6912751678 |
| Time Cost | - |0:02:54 | 0:03:42 |

&emsp;&emsp;相关结果见train_align_diff.log：
```
[2021/12/01 15:34:46] root INFO: acc: 
[2021/12/01 15:34:46] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/01 15:34:46] root INFO: diff check passed
```
## 问题来了!为啥结果一模一样！！！而且ACC远低于原论文的0.78？？？
##### 1.结果完全一致：  
* 没随机？  
&emsp;&emsp;将PyTorch中SequentialSampler改为RandomSampler，Paddle中BatchSampler参数Shuffle=True，结果：
|  评价指标 | Transformers实现 | Paddle复现 |
| ----- | ----- | ----- |
| ACC |0.6912751678 | 0.6912751678 |

&emsp;&emsp;……所以原因是啥我也不知道

##### 2.指标不如原论文
&emsp;&emsp;真是头疼……你说paddle复现的效果不行也就算了，为啥用HuggingFace上的也不行？？？？
&emsp;&emsp;至于为什么差异这么大，个人猜想：

1. 原论文预训练在 TPU 上完成，相关算子或有差异；  
2. 超参未知。在Huggingface官方Pytorch模型的文档（https://huggingface.co/google/fnet-base/tree/main/ ）中有如下论述：
> Note that the training hyperparameters of the reproduced models were not the same as the official model, so the performance may differ significantly for some tasks (for example: CoLA). 

<center><img src="https://github.com/HJHGJGHHG/Paddle-FNet/blob/main/img/1.png"  style="zoom:30%;" width="70%"/></center>

&emsp;&emsp;不过个人调过许多次参，也试着用过Kaggle上的TPU，ACC均未超过0.7……
