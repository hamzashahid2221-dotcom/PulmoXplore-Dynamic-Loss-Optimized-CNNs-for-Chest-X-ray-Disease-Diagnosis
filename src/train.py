import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks

from data_loader import get_datasets
from model import build_model
from losses import categorical_focal_loss, AdaptiveCategoricalFocalLoss
from callbacks import AdaptiveAlphaGammaCallback


def main():

    batch_size = 64
    initial_lr = 0.001
    fine_tune_lr = 0.001

    train_ds, val_ds, test_ds, class_names = get_datasets(batch_size)

    model = build_model(len(class_names))

    alpha = [0.4646, 1.3379, 9.9773]
    gamma = [2, 2, 2]

    # Phase 1 Training
    optimizer = Adam(learning_rate=initial_lr)

    model.compile(
        optimizer=optimizer,
        loss=categorical_focal_loss(alpha=alpha, gamma=gamma),
        metrics=['accuracy']
    )

    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True
    )

    checkpoint = callbacks.ModelCheckpoint(
        monitor='val_loss',
        filepath='best_model.keras',
        save_best_only=True
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[early_stopping, checkpoint]
    )

    # Fine-Tuning
    model.layers[0].trainable = True

    loss_fn = AdaptiveCategoricalFocalLoss(alpha=alpha, gamma=gamma)

    adaptive_cb = AdaptiveAlphaGammaCallback(
        loss_fn, val_ds, class_names
    )

    optimizer = Adam(learning_rate=fine_tune_lr)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=[adaptive_cb]
    )


if __name__ == "__main__":
    main()
