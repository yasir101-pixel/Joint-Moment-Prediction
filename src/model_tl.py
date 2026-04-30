import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import VGG16

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
IMAGE_SIZE = 150
EPOCHS = 200
BATCH_SIZE = 16
LEARNING_RATE = 0.001


def build_transfer_model(n_outputs, image_size=IMAGE_SIZE):
    """
    VGG16-based transfer learning model following Liew 2021.
    Frozen VGG16 backbone + custom regression head.
    Input: (150, 150, 3) RGB image
    Output: (n_outputs,) joint moments
    """
    # Load VGG16 without top layers, pretrained on ImageNet
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(image_size, image_size, 3)
    )

    # Freeze all VGG16 layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom regression head
    inputs = base_model.input
    x = base_model.output
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_outputs, activation='linear')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.RMSprop(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )
    return model


def train_transfer_model(X_train_img, y_train, X_val_img, y_val, n_outputs,
                         epochs=EPOCHS, batch_size=BATCH_SIZE):
    """
    Train VGG16 transfer learning model.
    X_train_img: (n, 150, 150, 3)
    y_train: (n, n_outputs)
    """
    model = build_transfer_model(n_outputs)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6
        )
    ]

    history = model.fit(
        X_train_img, y_train,
        validation_data=(X_val_img, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return model, history


def fine_tune_transfer_model(model, X_train_img, y_train, X_val_img, y_val,
                              unfreeze_last_n=4, epochs=50, batch_size=BATCH_SIZE):
    """
    Optional second stage: unfreeze last N VGG16 layers and fine-tune.
    """
    # Unfreeze last N layers of VGG16
    for layer in model.layers[-unfreeze_last_n:]:
        layer.trainable = True

    # Recompile with lower learning rate
    model.compile(
        optimizer=optimizers.RMSprop(learning_rate=LEARNING_RATE / 10),
        loss='mse',
        metrics=['mae']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train_img, y_train,
        validation_data=(X_val_img, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return model, history


if __name__ == '__main__':
    print("Building VGG16 Transfer Learning model...")
    model = build_transfer_model(n_outputs=10)
    model.summary()
    print("model_tl.py OK!")