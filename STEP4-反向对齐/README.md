# STEP4-反向对齐
&emsp;&emsp;在本节中，我们期望对齐Scheduler、Optimizer、正则化策略以及最终实现两个模型反向传播的对齐。  
&emsp;&emsp;文件目录：

```
D:.
│  bp_align_diff.log
│  bp_align_paddle.npy
│  bp_align_torch.npy
│  check_step4.py
│  compare_scheduler.py
│  modeling.py
│  padddle.ipynb
│  README.md
│  scheduler_diff.log
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

## LR Scheduler对齐
&emsp;&emsp;NOTE:本部分Copy自：https://github.com/JunnYu/BERT-SST2-Prod/blob/main/pipeline/Step4/test_lr_scheduler.py
```
cd STEP4-反向对齐
python compare_schedular.py
```
&emsp;&emsp;相关误差见：scheduler_diff.log，线性与多项式策略均无误差，余弦学习率由于算子的原因有少许误差。
```
[2021/12/01 10:17:19] root INFO: step_100_linear_lr: 
[2021/12/01 10:17:19] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/01 10:17:19] root INFO: step_300_linear_lr: 
[2021/12/01 10:17:19] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/01 10:17:19] root INFO: step_500_linear_lr: 
[2021/12/01 10:17:19] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/01 10:17:19] root INFO: step_700_linear_lr: 
[2021/12/01 10:17:19] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/01 10:17:19] root INFO: step_900_linear_lr: 
[2021/12/01 10:17:19] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/01 10:17:19] root INFO: step_100_cosine_lr: 
[2021/12/01 10:17:19] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/01 10:17:19] root INFO: step_300_cosine_lr: 
[2021/12/01 10:17:19] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/01 10:17:19] root INFO: step_500_cosine_lr: 
[2021/12/01 10:17:19] root INFO: 	mean diff: check passed: False, value: 9.35605818719964e-06
[2021/12/01 10:17:19] root INFO: step_700_cosine_lr: 
[2021/12/01 10:17:19] root INFO: 	mean diff: check passed: False, value: 1.3681476625617212e-05
[2021/12/01 10:17:19] root INFO: step_900_cosine_lr: 
[2021/12/01 10:17:19] root INFO: 	mean diff: check passed: False, value: 1.8924391285779562e-05
[2021/12/01 10:17:19] root INFO: step_100_polynomial_lr: 
[2021/12/01 10:17:19] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/01 10:17:19] root INFO: step_300_polynomial_lr: 
[2021/12/01 10:17:19] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/01 10:17:19] root INFO: step_500_polynomial_lr: 
[2021/12/01 10:17:19] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/01 10:17:19] root INFO: step_700_polynomial_lr: 
[2021/12/01 10:17:19] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/01 10:17:19] root INFO: step_900_polynomial_lr: 
[2021/12/01 10:17:19] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/01 10:17:19] root INFO: diff check failed
```

## 反向对齐
&emsp;&emsp;在check_step4.py中我们分别测试了pytorch与paddle模型在EVAL状态（即关闭随机因子如Dropout）下10次反向传播下Loss的误差。见bp_align_diff.log
```
python check_step4.py
```
&emsp;&emsp;可见是完全一致的。
```
[2021/12/01 14:02:01] root INFO: loss_0: 
[2021/12/01 14:02:01] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/01 14:02:01] root INFO: loss_1: 
[2021/12/01 14:02:01] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/01 14:02:01] root INFO: loss_2: 
[2021/12/01 14:02:01] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/01 14:02:01] root INFO: loss_3: 
[2021/12/01 14:02:02] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/01 14:02:02] root INFO: loss_4: 
[2021/12/01 14:02:02] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/01 14:02:02] root INFO: loss_5: 
[2021/12/01 14:02:02] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/01 14:02:02] root INFO: loss_6: 
[2021/12/01 14:02:02] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/01 14:02:02] root INFO: loss_7: 
[2021/12/01 14:02:02] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/01 14:02:02] root INFO: loss_8: 
[2021/12/01 14:02:02] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/01 14:02:02] root INFO: loss_9: 
[2021/12/01 14:02:02] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/01 14:02:02] root INFO: diff check passed

```
