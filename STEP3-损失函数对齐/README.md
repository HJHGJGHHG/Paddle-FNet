# STEP3-损失函数对齐
&emsp;&emsp;在本节中，我们期望对齐损失函数：CrossEntropyLoss()
&emsp;&emsp;文件目录：

```
D:.
│  check_step3.py  # 损失函数对齐检测
│  loss_diff.log  # 损失函数误差
│  loss_paddle.npy  # paddle数据
│  loss_torch.npy  # pytorch数据
│  modeling.py
│  paddle_loss.py  
│  README.md
│  tokenizer.py
│  torch_loss.py
│
├─classifier_weights
│      generate_classifier_weights.py
│      paddle_classifier_weights.bin
│      torch_classifier_weights.bin
│
└─fake_data
        fake_data.npy
        fake_label.npy
        gen_fake_data.py
```
&emsp;&emsp;二者误差见loss_diff.log：
```
[2021/11/30 23:19:29] root INFO: loss: 
[2021/11/30 23:19:29] root INFO: 	mean diff: check passed: True, value: 4.172325134277344e-07
[2021/11/30 23:19:29] root INFO: diff check passed
```