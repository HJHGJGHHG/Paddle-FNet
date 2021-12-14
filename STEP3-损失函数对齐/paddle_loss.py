import numpy as np
import paddle
import paddle.nn as nn
from modeling import FNetForSequenceClassification
from reprod_log import ReprodLogger

if __name__ == "__main__":
    paddle.set_device("cpu")

    # def logger
    reprod_logger = ReprodLogger()

    model = FNetForSequenceClassification.from_pretrained(
        "/root/autodl-tmp/PaddleFNet/model/paddle/fnet-base/")
    classifier_weights = paddle.load(
        "classifier_weights/paddle_classifier_weights.bin")
    model.load_dict(classifier_weights)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    # read or gen fake data
    fake_data = np.load("fake_data/fake_data.npy")
    fake_data = paddle.to_tensor(fake_data)

    fake_label = np.load("fake_data/fake_label.npy")
    fake_label = paddle.to_tensor(fake_label)

    # forward
    out = model(fake_data)

    loss = criterion(out, fake_label)
    
    reprod_logger.add("loss", loss.cpu().detach().numpy())
    reprod_logger.save("loss_paddle.npy")