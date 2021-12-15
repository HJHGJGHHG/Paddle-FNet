# Paddle FNet 复现文档
## 目标
&emsp;&emsp;基于PaddlePaddle复现FNet。
## 摘要
&emsp;&emsp;我们证明，通过用简单的线性转换可以取代自注意力层，transformer编码器架构可以通过“混合”输入标记的线性转换来大幅提高速度，精度成本有限。这些线性变换，以及前馈层中简单的非线性层，足以在几个文本分类任务中建模语义关系。也许最令人惊讶的是，我们发现用标准的、非参数化的傅里叶变换替换transformer编码器中的自注意层在GLUE基准上实现了92%的精度，但在GPU上运行速度快7倍，在TPU上快两倍。所得到的模型，我们命名为FNet，非常有效地扩展到长输入，与 the Long Range Arena benchmark上最精确的“高效的”transformer的精度相匹配，但在GPU上的**所有序列长度**和TPU上相对较短的序列长度上训练和运行得更快。最后，FNet具有较轻的内存占用，在较小的模型尺寸下特别有效：对于固定的速度和精度预算，小的FNet模型优于transformer对应的模型。

## 复现过程
##### STEP0. 权重转换
&emsp;&emsp;转换预训练权重。详见 /STEP0。  

##### STEP1. 前向对齐
&emsp;&emsp;期望对齐模型结构，详见 /STEP1。  

##### STEP2. 数据集、Tokenizer、评估指标对齐
&emsp;&emsp;实验数据承载自SST2，期望对齐Tokenizer、Dataset、DataLoader、Metircs（ACC），详见 /STEP2。  

##### STEP3. 损失函数对齐
&emsp;&emsp;期望对齐损失函数CrossEntropyLoss，详见 /STEP3。  

##### STEP4.反向对齐
&emsp;&emsp;在固定随机量（如Dropout）的情况下期望对齐Loss反向传播过程，详见 /STEP4。

##### STEP5.训练对齐
&emsp;&emsp;完成STEP0~STEP4后，在SST2、QQP上完成训练，并对比训练精度。  

##### STEP6.复现论文指标
&emsp;&emsp;完成复现过程后，在SST2&QQP上复现论文结果。  

## 参考
* https://github.com/JunnYu/paddle_reformer
* https://github.com/PaddlePaddle/models/blob/tipc/docs/lwfx/ArticleReproduction_NLP.md
* https://github.com/JunnYu/BERT-SST2-Prod
