import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
IMAGE_SIZE = 150
EPOCHS = 200
BATCH_SIZE = 16
LEARNING_RATE = 0.001

def build_dnn(n_outputs, image_size=IMAGE_SIZE):
    """
    Custom 3-block CNN following Liew 2021 architecture.
    Input: (150, 150, 3) RGB image
    Output: (n_outputs,) joint moments
    """
    inputs = layers.Input(shape=(image_size, image_size, 3))

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((3, 3))(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Head
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


def train_dnn(X_train_img, y_train, X_val_img, y_val, n_outputs,
              epochs=EPOCHS, batch_size=BATCH_SIZE):
    """
    Train the custom DNN.
    X_train_img: (n, 150, 150, 3)
    y_train: (n, n_outputs)
    """
    model = build_dnn(n_outputs)

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


if __name__ == '__main__':
    print("Building DNN model...")
    model = build_dnn(n_outputs=10)
    model.summary()
    print("model_dnn.py OK!")