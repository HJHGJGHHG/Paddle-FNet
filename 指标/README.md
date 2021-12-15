# 指标
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

&emsp;&emsp;Time：$FNet/BERT=0.675$

## paddle复现
|  数据集 | FNet-base | fnet-Large |
|  :--:  |  :--:   | :--: |
| SST2 (ACC) | 90.13% | 91.06% |
| QQP (ACC+F1)/2 | 85.71% | 87.18% |

&emsp;&emsp;单机单卡单精，Tesla V100
