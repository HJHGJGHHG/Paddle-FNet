import numpy as np
import paddle
import torch
from datasets import load_metric
from paddle.metric import Accuracy
from reprod_log import ReprodLogger


def main():
    pd_metric = Accuracy()
    pd_metric.reset()
    hf_metric = load_metric("accuracy.py")
    for i in range(4):
        logits = np.random.normal(0, 1, size=(64, 2)).astype("float32")
        labels = np.random.randint(0, 2, size=(64, )).astype("int64")
        # paddle metric
        correct = pd_metric.compute(
            paddle.to_tensor(logits), paddle.to_tensor(labels))
        pd_metric.update(correct)
        # hf metric
        hf_metric.add_batch(
            predictions=torch.from_numpy(logits).argmax(dim=-1),
            references=torch.from_numpy(labels), )
    pd_accuracy = pd_metric.accumulate()
    hf_accuracy = hf_metric.compute()["accuracy"]
    reprod_logger = ReprodLogger()
    reprod_logger.add("accuracy", np.array([pd_accuracy]))
    reprod_logger.save("metric_paddle.npy")
    reprod_logger = ReprodLogger()
    reprod_logger.add("accuracy", np.array([hf_accuracy]))
    reprod_logger.save("metric_torch.npy")


if __name__ == "__main__":
    main()