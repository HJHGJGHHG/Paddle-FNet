import numpy as np
import paddle
import torch
from datasets import load_metric
from reprod_log import ReprodLogger
from paddle_metric import F1_score
from paddle.metric.metrics import Accuracy


def main():
    pd_f1 = F1_score()
    pd_acc = Accuracy()
    hf_metric = load_metric("torch_metric.py")
    for i in range(4):
        logits = np.random.normal(0, 1, size=(64, 2)).astype("float32")
        labels = np.random.randint(0, 2, size=(64,)).astype("int64")
        # paddle metric
        corrects = pd_f1.compute(
            paddle.to_tensor(logits), paddle.to_tensor(labels))
        pd_acc.update(corrects)
        pd_f1.update(corrects=corrects, labels=paddle.to_tensor(labels))
        # hf metric
        hf_metric.add_batch(
            predictions=torch.from_numpy(logits).argmax(dim=-1),
            references=torch.from_numpy(labels), )
    
    hf_result = hf_metric.compute()
    # ACC
    pd_accuracy = pd_acc.accumulate()
    hf_accuracy = hf_result["accuracy"]
    reprod_logger = ReprodLogger()
    reprod_logger.add("accuracy", np.array([pd_accuracy]))
    reprod_logger.save("acc_paddle.npy")
    reprod_logger = ReprodLogger()
    reprod_logger.add("accuracy", np.array([hf_accuracy]))
    reprod_logger.save("acc_torch.npy")
    
    # F1
    pd_f1 = pd_f1.accumulate()
    hf_f1 = hf_result["f1"]
    reprod_logger = ReprodLogger()
    reprod_logger.add("f1", np.array([pd_f1]))
    reprod_logger.save("f1_paddle.npy")
    reprod_logger = ReprodLogger()
    reprod_logger.add("f1", np.array([hf_f1]))
    reprod_logger.save("f1_torch.npy")


if __name__ == "__main__":
    main()
