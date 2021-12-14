# STEP0-权重转换
&emsp;&emsp;使用权重转换脚本  
```
git clone https://huggingface.co/google/fnet-base
git clone https://huggingface.co/google/fnet-large

python torch2paddle.py --torch_file="fnet-base/pytorch_model.bin" --paddle_file="fnet-base-paddle/model_state.pdparams"

python torch2paddle.py --torch_file "fnet-large/pytorch_model.bin" --paddle_file "fnet-large-paddle/model_state.pdparams"
```

&emsp;&emsp;也可以用我传到HF上的模型：[Paddle-FNet-base](https://huggingface.co/HJHGJGHHG/paddle-fnet-base) 、[Paddle-FNet-large](https://huggingface.co/HJHGJGHHG/paddle-fnet-large)  