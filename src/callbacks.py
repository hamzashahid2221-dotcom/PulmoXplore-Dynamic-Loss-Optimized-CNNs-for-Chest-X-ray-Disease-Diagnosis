import numpy as np
import tensorflow as tf
from sklearn.metrics import recall_score


class AdaptiveAlphaGammaCallback(tf.keras.callbacks.Callback):
    def __init__(self, loss_fn, val_data, class_names):
        super().__init__()
        self.loss_fn = loss_fn
        self.val_data = val_data
        self.class_names = class_names

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_pred = [], []

        for x_batch, y_batch in self.val_data:
            preds = self.model.predict(x_batch, verbose=0)
            y_pred.extend(np.argmax(preds, axis=1))
            y_true.extend(np.argmax(y_batch.numpy(), axis=1))

        recalls = recall_score(
            y_true,
            y_pred,
            labels=list(range(len(self.class_names))),
            average=None
        )

        new_gamma = 2.0 + (1.0 - recalls) * 2.0
        new_alpha = 1.0 + (1.0 - recalls) * 2.0

        self.loss_fn.gamma.assign(
            tf.convert_to_tensor(new_gamma, dtype=tf.float32)
        )

        self.loss_fn.alpha.assign(
            tf.convert_to_tensor(new_alpha, dtype=tf.float32)
        )

        print(f"[AdaptiveAlphaGamma] Updated gamma: {new_gamma}")
        print(f"[AdaptiveAlphaGamma] Updated alpha: {new_alpha}")
