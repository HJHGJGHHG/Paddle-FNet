import numpy as np
import paddle
import torch


def generate(seed):
    np.random.seed(seed)
    weight = np.random.normal(0, 0.02, (768, 2)).astype("float32")  # base
    #weight = np.random.normal(0, 0.02, (1024, 2)).astype("float32")  # large
    bias = np.zeros((2, )).astype("float32")
    paddle_weights = {
        "classifier.weight": weight,
        "classifier.bias": bias,
    }
    torch_weights = {
        "classifier.weight": torch.from_numpy(weight).t(),
        "classifier.bias": torch.from_numpy(bias),
    }
    torch.save(torch_weights, "torch_classifier_weights.bin")
    paddle.save(paddle_weights, "paddle_classifier_weights.bin")


if __name__ == "__main__":
    generate(seed=42)