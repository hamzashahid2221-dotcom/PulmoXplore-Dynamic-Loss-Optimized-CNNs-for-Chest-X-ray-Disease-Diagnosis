import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import get_datasets


def evaluate():

    _, _, test_ds, class_names = get_datasets()

    model = tf.keras.models.load_model(
        'fine_tune_best_model.keras',
        compile=False
    )

    y_pred = []
    y_true = []

    for images, labels in test_ds:
        pred_prob = model.predict(images)
        y_pred.extend(np.argmax(pred_prob, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    cr = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )

    print("Classification Report:\n", cr)

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)


if __name__ == "__main__":
    evaluate()
