import numpy as np
import paddle
import torch


def generate(seed):
    np.random.seed(seed)
    weight_base = np.random.normal(0, 0.02, (768, 2)).astype("float32")  # base
    weight_large = np.random.normal(0, 0.02, (1024, 2)).astype("float32")  # large
    bias = np.zeros((2, )).astype("float32")
    paddle_weights_large = {
        "classifier.weight": weight_large,
        "classifier.bias": bias,
    }
    paddle_weights_base = {
        "classifier.weight": weight_base,
        "classifier.bias": bias,
    }
    paddle.save(paddle_weights_large, "paddle_classifier_weights_large.bin")
    paddle.save(paddle_weights_base, "paddle_classifier_weights_base.bin")


if __name__ == "__main__":
    generate(seed=1234)