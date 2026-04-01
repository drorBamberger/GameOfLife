"""
Experiment loop: iteratively test architectures to reach >0.9 val accuracy
for reverse Game of Life prediction (SIZE=10, gen=2, pixel-level binary).

Key insight: gen=2 means we have 1 input board → predict 1 pixel of the
previous board.  This is a SPATIAL problem, not temporal.
RNNs add unnecessary complexity for 1 timestep.
"""

import os
import time
import numpy as np
import tensorflow as tf
from functions import load_reverse_df, prepare_reverse_dataset, to_numpy_4d

# ── Config ────────────────────────────────────────────────────────────────────
SIZE = 10
AMOUNT_BOARDS = 100_000
GEN = 2
os.makedirs("models", exist_ok=True)

# ── Load data once ────────────────────────────────────────────────────────────
print("Loading data...")
reverse_df = load_reverse_df(SIZE, AMOUNT_BOARDS, GEN)
X_train, X_val, X_test, y_train, y_val, y_test = prepare_reverse_dataset(
    reverse_df, SIZE, GEN, target_pixel_index=0,
    test_size=0.1, val_size=0.1, random_state=365,
)
X_train_a, X_val_a, X_test_a, y_train_a, y_val_a, y_test_a = to_numpy_4d(
    X_train, X_val, X_test, y_train, y_val, y_test, SIZE, GEN,
)
print(f"Train: {X_train_a.shape}, Val: {X_val_a.shape}, Test: {X_test_a.shape}")
print(f"y_train mean (class balance): {y_train_a.mean():.3f}")


# ── Helper ────────────────────────────────────────────────────────────────────
def run_experiment(name, model, X_tr, y_tr, X_v, y_v, X_te, y_te,
                   epochs=30, batch_size=128):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*60}")
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=8,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3,
            min_lr=1e-6, verbose=1),
    ]

    t0 = time.time()
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_v, y_v),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=2,
    )
    elapsed = time.time() - t0

    test_loss, test_acc = model.evaluate(X_te, y_te, verbose=0)
    best_val = max(history.history['val_accuracy'])
    best_epoch = int(np.argmax(history.history['val_accuracy'])) + 1

    print(f"\n--- {name} RESULTS ---")
    print(f"Best val_accuracy : {best_val:.4f}  (epoch {best_epoch})")
    print(f"Test accuracy     : {test_acc:.4f}")
    print(f"Time              : {elapsed:.1f}s")
    print(f"{'='*60}\n")

    return best_val, test_acc, history


# ══════════════════════════════════════════════════════════════════════════════
# ITERATION 1 — Deep ResNet-style CNN
#
# Rationale: gen=2 gives 1 timestep → purely spatial.
# GoL rules are local (3x3 neighbourhood) so Conv2D is ideal.
# Use residual connections to preserve gradient flow in deeper nets.
# Use all 100 input pixels, heavy feature extraction to predict 1 pixel.
# ══════════════════════════════════════════════════════════════════════════════
def residual_block(x, filters, kernel_size=3):
    """Conv-BN-ReLU-Conv-BN + skip connection."""
    shortcut = x
    y = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', use_bias=False)(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation('relu')(y)
    y = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', use_bias=False)(y)
    y = tf.keras.layers.BatchNormalization()(y)
    # project shortcut if channel mismatch
    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, padding='same', use_bias=False)(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    y = tf.keras.layers.Add()([shortcut, y])
    y = tf.keras.layers.Activation('relu')(y)
    return y


def build_resnet_cnn(input_shape):
    inp = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(64, 3, padding='same', use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128)
    x = residual_block(x, 128)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model


print("\n\n" + "#"*60)
print("# ITERATION 1: ResNet CNN")
print("#"*60)
tf.keras.backend.clear_session()
m1 = build_resnet_cnn(X_train_a.shape[1:])
val1, test1, h1 = run_experiment(
    "ResNet CNN", m1, X_train_a, y_train_a,
    X_val_a, y_val_a, X_test_a, y_test_a,
    epochs=30, batch_size=128,
)


# ══════════════════════════════════════════════════════════════════════════════
# ITERATION 2 — CNN + Coordinate Encoding + Self-Attention
#
# GoL is position-sensitive (edge vs centre pixels behave differently).
# Add coordinate channels so the model knows WHERE each pixel is,
# plus a self-attention mechanism to model long-range spatial dependencies.
# ══════════════════════════════════════════════════════════════════════════════
class CoordConv2D(tf.keras.layers.Layer):
    """Adds normalized (row, col) coordinate channels to the input."""
    def call(self, x):
        batch = tf.shape(x)[0]
        h, w = x.shape[1], x.shape[2]
        row = tf.linspace(-1.0, 1.0, h)
        col = tf.linspace(-1.0, 1.0, w)
        row_grid, col_grid = tf.meshgrid(row, col, indexing='ij')
        coords = tf.stack([row_grid, col_grid], axis=-1)           # (h, w, 2)
        coords = tf.expand_dims(coords, 0)                         # (1, h, w, 2)
        coords = tf.repeat(coords, batch, axis=0)                  # (B, h, w, 2)
        return tf.concat([x, coords], axis=-1)


def channel_attention(x, ratio=8):
    """Squeeze-and-Excitation (SE) block."""
    filters = x.shape[-1]
    se = tf.keras.layers.GlobalAveragePooling2D()(x)
    se = tf.keras.layers.Dense(filters // ratio, activation='relu')(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid')(se)
    se = tf.keras.layers.Reshape((1, 1, filters))(se)
    return tf.keras.layers.Multiply()([x, se])


def build_coord_attention_cnn(input_shape):
    inp = tf.keras.layers.Input(shape=input_shape)

    # Add coordinate channels
    x = CoordConv2D()(inp)

    # Initial conv
    x = tf.keras.layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Residual blocks with SE attention
    for filters in [64, 64, 128, 128]:
        x = residual_block(x, filters)
        x = channel_attention(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model


print("\n\n" + "#"*60)
print("# ITERATION 2: CoordConv + SE-Attention + ResNet")
print("#"*60)
tf.keras.backend.clear_session()
m2 = build_coord_attention_cnn(X_train_a.shape[1:])
val2, test2, h2 = run_experiment(
    "CoordConv+SE+ResNet", m2, X_train_a, y_train_a,
    X_val_a, y_val_a, X_test_a, y_test_a,
    epochs=40, batch_size=128,
)


# ══════════════════════════════════════════════════════════════════════════════
# ITERATION 3 — Multi-pixel prediction (predict ALL 100 pixels at once)
#
# Key paradigm shift: instead of predicting just 1 pixel, predict the
# ENTIRE previous board (100 outputs).  This gives the model a much richer
# learning signal and forces it to learn the full GoL inverse mapping.
# We then extract the pixel-0 prediction and measure accuracy on that.
# ══════════════════════════════════════════════════════════════════════════════
def prepare_full_board_targets(reverse_df, size, gen, test_size=0.1, val_size=0.1, random_state=365):
    """Prepare data where y = full previous board (size*size pixels)."""
    cols = [f'Col_{i}' for i in range(gen * size * size)]
    df = reverse_df.copy()
    df[cols] = df[cols].astype(int)

    amount_features = (gen - 1) * size * size
    features = df.iloc[:, :amount_features]
    target = df.iloc[:, amount_features:]  # ALL pixels of the target board

    X_tv, X_te, y_tv, y_te = train_test_split(
        features, target, test_size=test_size, random_state=random_state)
    X_tr, X_v, y_tr, y_v = train_test_split(
        X_tv, y_tv, test_size=val_size, random_state=random_state)

    def to4d(X):
        return X.to_numpy().reshape((-1, size, size, gen - 1)).astype('float32')

    return (to4d(X_tr), to4d(X_v), to4d(X_te),
            y_tr.to_numpy().astype('float32'),
            y_v.to_numpy().astype('float32'),
            y_te.to_numpy().astype('float32'))


def build_full_board_predictor(input_shape, output_size):
    """Predict entire previous board, then extract pixel 0."""
    inp = tf.keras.layers.Input(shape=input_shape)
    x = CoordConv2D()(inp)

    x = tf.keras.layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    for filters in [64, 128, 128, 256]:
        x = residual_block(x, filters)
        x = channel_attention(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(output_size, activation='sigmoid')(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy'],  # per-pixel accuracy across all 100 outputs
    )
    return model


from sklearn.model_selection import train_test_split

print("\n\n" + "#"*60)
print("# ITERATION 3: Full-board prediction (100 outputs)")
print("#"*60)
tf.keras.backend.clear_session()

X_tr3, X_v3, X_te3, y_tr3, y_v3, y_te3 = prepare_full_board_targets(
    reverse_df, SIZE, GEN)

m3 = build_full_board_predictor(X_tr3.shape[1:], SIZE * SIZE)
val3, test3, h3 = run_experiment(
    "FullBoard CoordConv+SE+ResNet", m3,
    X_tr3, y_tr3, X_v3, y_v3, X_te3, y_te3,
    epochs=50, batch_size=128,
)

# Extract pixel-0 accuracy specifically
y_pred3 = m3.predict(X_te3, verbose=0)
pixel0_pred = (y_pred3[:, 0] > 0.5).astype(int)
pixel0_true = y_te3[:, 0].astype(int)
pixel0_acc = np.mean(pixel0_pred == pixel0_true)
print(f"Pixel-0 specific accuracy: {pixel0_acc:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# ITERATION 4 — U-Net style encoder-decoder for full board prediction
#
# Spatial→Spatial: input board → output board.
# U-Net preserves fine spatial details via skip connections.
# This architecture is ideal for dense pixel-wise prediction.
# ══════════════════════════════════════════════════════════════════════════════
def build_unet(input_shape, output_size):
    inp = tf.keras.layers.Input(shape=input_shape)
    x = CoordConv2D()(inp)

    # Encoder
    e1 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    e1 = tf.keras.layers.BatchNormalization()(e1)
    e1 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(e1)
    e1 = tf.keras.layers.BatchNormalization()(e1)
    p1 = tf.keras.layers.MaxPooling2D(2)(e1)  # 5x5

    e2 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(p1)
    e2 = tf.keras.layers.BatchNormalization()(e2)
    e2 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(e2)
    e2 = tf.keras.layers.BatchNormalization()(e2)

    # Bottleneck
    b = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(e2)
    b = tf.keras.layers.BatchNormalization()(b)
    b = channel_attention(b)

    # Decoder
    d2 = tf.keras.layers.UpSampling2D(2)(b)  # back to 10x10
    d2 = tf.keras.layers.Concatenate()([d2, e1])
    d2 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(d2)
    d2 = tf.keras.layers.BatchNormalization()(d2)
    d2 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(d2)
    d2 = tf.keras.layers.BatchNormalization()(d2)

    # Output: per-pixel sigmoid
    out_map = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(d2)  # (B, 10, 10, 1)
    out_flat = tf.keras.layers.Flatten()(out_map)  # (B, 100)

    model = tf.keras.Model(inp, out_flat)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model


print("\n\n" + "#"*60)
print("# ITERATION 4: U-Net encoder-decoder (full board)")
print("#"*60)
tf.keras.backend.clear_session()
m4 = build_unet(X_tr3.shape[1:], SIZE * SIZE)
val4, test4, h4 = run_experiment(
    "U-Net Full Board", m4,
    X_tr3, y_tr3, X_v3, y_v3, X_te3, y_te3,
    epochs=50, batch_size=128,
)

y_pred4 = m4.predict(X_te3, verbose=0)
pixel0_pred4 = (y_pred4[:, 0] > 0.5).astype(int)
pixel0_acc4 = np.mean(pixel0_pred4 == y_te3[:, 0].astype(int))
print(f"Pixel-0 specific accuracy: {pixel0_acc4:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "="*60)
print("EXPERIMENT SUMMARY")
print("="*60)
print(f"{'Model':<35} {'Val Acc':>10} {'Test Acc':>10}")
print("-"*55)
print(f"{'1. ResNet CNN (1-pixel)':<35} {val1:>10.4f} {test1:>10.4f}")
print(f"{'2. CoordConv+SE+ResNet (1-pixel)':<35} {val2:>10.4f} {test2:>10.4f}")
print(f"{'3. Full-board CoordConv+SE+Res':<35} {val3:>10.4f} {test3:>10.4f}")
print(f"   (pixel-0 acc){'':>19} {pixel0_acc:>10.4f}")
print(f"{'4. U-Net full-board':<35} {val4:>10.4f} {test4:>10.4f}")
print(f"   (pixel-0 acc){'':>19} {pixel0_acc4:>10.4f}")
print("="*60)

# Save best model
best_val = max(val1, val2, val3, val4)
if best_val >= 0.90:
    print(f"\n*** TARGET 0.90 REACHED! Best val_accuracy = {best_val:.4f}")
else:
    print(f"\nBest val_accuracy so far: {best_val:.4f} — will need further iterations.")
