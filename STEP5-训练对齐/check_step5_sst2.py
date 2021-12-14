import numpy as np
from reprod_log import ReprodDiffHelper
from tensorboardX import SummaryWriter

if __name__ == "__main__":
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("torch_train/sst2_train_align_benchmark.npy")
    paddle_info = diff_helper.load_info("paddle_train/sst2_train_align_paddle.npy")

    diff_helper.compare_info(torch_info, paddle_info)

    diff_helper.report(path="sst2_train_align_diff.log", diff_threshold=0.0015)
    
    paddle_sst2_losses = np.load("./paddle_train/paddle_sst2_losses.npy")
    with SummaryWriter('/root/tf-logs/paddle_sst2') as writer:
        for i in range(len(paddle_sst2_losses)):
            writer.add_scalar('Loss', paddle_sst2_losses[i], i)
            
    torch_sst2_losses = np.load("./torch_train/torch_sst2_losses.npy")
    with SummaryWriter('/root/tf-logs/torch_sst2') as writer:
        for i in range(len(torch_sst2_losses)):
            writer.add_scalar('Loss', torch_sst2_losses[i], i)