# STEP1-前向对齐
``` python
# 生成classifier权重
cd STEP1/ && python generate_classifier_weights.py
# 生成paddle的前向数据
python pd_forward_bert.py
# 生成torch的前向数据
python pt_forward_bert.py
# 对比生成log
python check_step1.py

```

&emsp;&emsp;文件结构如下：
```
D:.
│  check_step1.py  # 校验程序
│  forward_diff.log  # 对比结果
│  forward_paddle.npy  # paddle模型前向数据
│  forward_torch.npy  # pytorch模型前向数据
│  modeling.py  #paddle模型
│  pd_forward_fnet.py  # paddle模型前向数据生成程序
│  pt_forward_fnet.py  # pytorch模型前向数据生成程序
│  README.md
│
├─classifier_weights  # 测试用权重
│      generate_classifier_weights.py
│      paddle_classifier_weights.bin
│      torch_classifier_weights.bin
│
└─fake_data  # 测试用数据
        fake_data.npy
        fake_label.npy
        gen_fake_data.py
```
&emsp;&emsp;模型前向数据误差在forward_diff.log中查看：
```
[2021/11/30 17:17:31] root INFO: logits: 
[2021/11/30 17:17:31] root INFO: 	mean diff: check passed: True, value: 1.1222437024116516e-06
[2021/11/30 17:17:31] root INFO: diff check passed
```
&emsp;&emsp;阈值设定为2e-6，测试通过。