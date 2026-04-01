import tensorflow as tf
from tensorflow.keras import layers, Model


def build_full_cnn_model(input_shape):
    """Fully-convolutional model: Input (H, W, C) -> Output (H, W, 1).

    Uses 'same' padding throughout so spatial dimensions are preserved.
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    outputs = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(x)

    model = Model(inputs, outputs, name='FullCNN')
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def build_full_mlp_model(input_shape):
    """Dense (MLP) model that flattens, processes, and reshapes back to (H, W, 1)."""
    h, w = input_shape[0], input_shape[1]

    inputs = layers.Input(shape=input_shape)
    x = layers.Flatten()(inputs)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(h * w, activation='sigmoid')(x)
    outputs = layers.Reshape((h, w, 1))(x)

    model = Model(inputs, outputs, name='FullMLP')
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def build_unet_model(input_shape):
    """Lightweight U-Net for 10x10 grids: Input (H, W, C) -> Output (H, W, 1).

    Uses padding='same' everywhere to keep shapes aligned at each skip connection.
    Two down-sampling / up-sampling stages are used (sufficient for 10x10 boards).
    """
    inputs = layers.Input(shape=input_shape)

    # --- Encoder ---
    c1 = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    c1 = layers.Conv2D(64, 3, padding='same', activation='relu')(c1)
    p1 = layers.MaxPooling2D(2, padding='same')(c1)  # -> (5, 5, 64)

    c2 = layers.Conv2D(128, 3, padding='same', activation='relu')(p1)
    c2 = layers.Conv2D(128, 3, padding='same', activation='relu')(c2)
    p2 = layers.MaxPooling2D(2, padding='same')(c2)  # -> (3, 3, 128)

    # --- Bottleneck ---
    bn = layers.Conv2D(256, 3, padding='same', activation='relu')(p2)
    bn = layers.Conv2D(256, 3, padding='same', activation='relu')(bn)

    # --- Decoder ---
    u2 = layers.UpSampling2D(2)(bn)                  # -> (6, 6, 256)
    # Resize to match c2 spatial dims before concatenation
    u2 = layers.Resizing(c2.shape[1], c2.shape[2])(u2)
    u2 = layers.Concatenate()([u2, c2])
    c3 = layers.Conv2D(128, 3, padding='same', activation='relu')(u2)
    c3 = layers.Conv2D(128, 3, padding='same', activation='relu')(c3)

    u1 = layers.UpSampling2D(2)(c3)                  # -> (10, 10, 128)
    u1 = layers.Resizing(c1.shape[1], c1.shape[2])(u1)
    u1 = layers.Concatenate()([u1, c1])
    c4 = layers.Conv2D(64, 3, padding='same', activation='relu')(u1)
    c4 = layers.Conv2D(64, 3, padding='same', activation='relu')(c4)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c4)

    model = Model(inputs, outputs, name='UNet')
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
