import argparse


def args_initialization():
    parser = argparse.ArgumentParser(description="FNet-CoLA")
    
    parser.add_argument("--seed", type=int, default=1234, help='随机种子')
    parser.add_argument("--model-dir", type=str, default='google/fnet-large', help='模型位置')
    parser.add_argument("--batch-size", type=int, default=4, help='Batch Size')
    parser.add_argument("--lr", type=float, default=1e-5, help='Learning Rate')
    parser.add_argument("--warmup", type=int, default=0, help='Warmup Steps')
    parser.add_argument("--num-epochs", type=int, default=3, help='Epoch 数')

    
    args = parser.parse_args(args=[])
    return args
