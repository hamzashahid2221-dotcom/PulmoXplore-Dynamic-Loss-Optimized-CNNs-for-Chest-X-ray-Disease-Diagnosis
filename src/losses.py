import tensorflow as tf
from tensorflow.keras import backend as K


def categorical_focal_loss(alpha, gamma):
    alpha = tf.constant(alpha, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1. - K.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

    return loss


class AdaptiveCategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha, gamma, from_logits=False,
                 reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE):
        super().__init__(reduction=reduction)

        self.alpha = tf.Variable(
            initial_value=tf.convert_to_tensor(alpha, dtype=tf.float32),
            trainable=False
        )

        self.gamma = tf.Variable(
            initial_value=tf.convert_to_tensor(gamma, dtype=tf.float32),
            trainable=False
        )

        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred)

        cross_entropy = -y_true * tf.math.log(
            tf.clip_by_value(y_pred, 1e-8, 1.0)
        )

        weight = self.alpha * tf.pow(1 - y_pred, self.gamma)
        loss = tf.reduce_sum(weight * cross_entropy, axis=1)

        return loss
