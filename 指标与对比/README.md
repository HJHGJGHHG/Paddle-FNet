# 指标与对比
&emsp;&emsp;在本节中，我们期望复现出论文的结果。  
## 目标
&emsp;&emsp;我们将对比FNet与BERT在SST2、QQP上相关指标、训练时间等方面的表现。
&emsp;&emsp;***原论文***的结果：（Table1）

|  数据集 | FNet-base| FNet-Large  | BERT-Large |
|  :--:  |  :--:   |  :--:   | :--: |
| SST2 (ACC) | 95% | 94% | 95% |
| QQP (ACC+F1)/2 | 83% | 85% | 88% |

&emsp;&emsp;***官方pytorch复现版本***相关指标：

|  数据集 | FNet-base | FNET-large | BERT-base |
|  :--:  |  :--:   | :--: | :--: |
| SST2 | [89.45%](https://huggingface.co/gchhablani/fnet-base-finetuned-sst2) | [90.48%](https://huggingface.co/gchhablani/fnet-large-finetuned-sst2) | [92.32%](https://huggingface.co/gchhablani/bert-base-cased-finetuned-sst2) |
| QQP | [86.57%](https://huggingface.co/gchhablani/fnet-base-finetuned-qqp) | [87.5%](https://huggingface.co/gchhablani/fnet-large-finetuned-qqp) | [89.26%](https://huggingface.co/gchhablani/bert-base-cased-finetuned-qqp) |

&emsp;&emsp;Fine-Tune耗时：
| Task/Model | FNet-base (PyTorch) |Bert-base (PyTorch)|
|:----:|:-----------:|:----:|
| QQP  | [06:21:16](https://huggingface.co/gchhablani/fnet-base-finetuned-qqp) | [09:25:01](https://huggingface.co/gchhablani/bert-base-cased-finetuned-qqp) |
| SST-2 | [01:09:27](https://huggingface.co/gchhablani/fnet-base-finetuned-sst2) | [01:42:17](https://huggingface.co/gchhablani/bert-base-cased-finetuned-sst2)|
| SUM | 07:30:43 | 11:07:18 |

&emsp;&emsp;Time：$FNet/BERT=0.675$

## paddle复现
### 1.最优结果
|  数据集 | FNet-base | fnet-Large |
|  :--:  |  :--:   | :--: |
| SST2 (ACC) | 89.56% | 90.71% |
| QQP (ACC+F1)/2 | 85% | 87.19% |

&emsp;&emsp;单机单卡单精，Tesla V100
* SST2超参组与训练策略：（seed均为1234）
  * batchsize = 16
  * lr = 2e-5
  * optimizer = AdamW（epsilon=1e-8）
  * LR-sheduler = Linear （Warmup Steps: 1000）
  * epochs = 3
  * eval_freq = 300 steps（每隔300步测试一次）
  * early_stop = 1500 steps
* QQP超参组与训练策略：（seed均为1234）
  * batchsize = 16
  * lr = 3e-5
  * optimizer = AdamW（epsilon=1e-6）
  * LR-sheduler = Linear （Warmup Steps: 3000）
  * epochs = 2
  * eval_freq = 2000 steps（每隔300步测试一次）
  * early_stop = 10000 steps
