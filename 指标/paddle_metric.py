import paddle
import numpy as np
from paddle.metric.metrics import Metric


def _is_numpy_(var):
    return isinstance(var, (np.ndarray, np.generic))


class F1_score(Metric):
    def __init__(self, topk=(1,), name='f1', *args, **kwargs):
        super(F1_score, self).__init__(*args, **kwargs)
        self.tp = 0  # true positive
        self.fp = 0  # false positive
        self.fn = 0  # false negative
        self._name = name
        self.topk = topk
        self.maxk = max(topk)
        self.reset()
    
    def compute(self, pred, label, *args):
        """
        Compute the top-k (maximum value in `topk`) indices.

        Args:
            pred (Tensor): The predicted value is a Tensor with dtype
                float32 or float64. Shape is [batch_size, d0, ..., dN].
            label (Tensor): The ground truth value is Tensor with dtype
                int64. Shape is [batch_size, d0, ..., 1], or
                [batch_size, d0, ..., num_classes] in one hot representation.

        Return:
            Tensor: Correct mask, a tensor with shape [batch_size, d0, ..., topk].
        """
        pred = paddle.argsort(pred, descending=True)
        pred = paddle.slice(
            pred, axes=[len(pred.shape) - 1], starts=[0], ends=[self.maxk])
        if (len(label.shape) == 1) or \
                (len(label.shape) == 2 and label.shape[-1] == 1):
            # In static mode, the real label data shape may be different
            # from shape defined by paddle.static.InputSpec in model
            # building, reshape to the right shape.
            label = paddle.reshape(label, (-1, 1))
        elif label.shape[-1] != 1:
            # one-hot label
            label = paddle.argmax(label, axis=-1, keepdim=True)
        correct = pred == label
        return paddle.cast(correct, dtype='float32')
    
    def update(self, corrects, labels, *args):
        sample_num = corrects.shape[0]
        if isinstance(labels, paddle.Tensor):
            labels = labels.numpy()
        elif not _is_numpy_(labels):
            raise ValueError("The 'labels' must be a numpy ndarray or Tensor.")
        
        for i in range(sample_num):
            correct = corrects[i]
            label = labels[i]
            if label == 1:
                if correct == 1:
                    self.tp += 1
                else:
                    self.fn += 1
            if label == 0:
                if correct == 0:
                    self.fp += 1
    
    def reset(self):
        """
        Resets all of the metric state.
        """
        self.tp = 0
        self.fp = 0
        self.fn = 0
    
    def accumulate(self):
        """
        Calculate the final precision.

        Returns:
            A scaler float: results of the calculated precision.
        """
        n = 2 * self.tp + self.fp + self.fn
        return float(2 * self.tp) / n if n != 0 else .0
    
    def name(self):
        """
        Returns metric name
        """
        return self._name


if __name__ == '__main__':
    logits = np.random.normal(0, 1, size=(64, 2)).astype("float32")
    labels = np.random.randint(0, 2, size=(64,)).astype("int64")
    pd_metric = F1_score()
    pd_metric.reset()
    pd_metric.update(preds=paddle.to_tensor(logits),
                     labels=paddle.to_tensor(labels))
    pd_metric.accumulate()
