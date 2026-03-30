"""
Advanced Image-to-Image Model for Reverse Game of Life (10×10 grids).

Architecture: Attention Residual U-Net with wrap padding for toroidal topology.
Custom metric: Forward Consistency — applies GoL rules to predicted T-1 board and
               checks how well it reproduces the original T input.

Usage:
    # Standalone
    python advanced_discovery_model.py

    # From notebook / another script
    from advanced_discovery_model import (
        load_data_full_board, build_attention_resunet,
        forward_consistency_np, train_and_evaluate
    )
    results = train_and_evaluate(size=10, amount_boards=10000, gen=2, epochs=80)
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ─── Import existing data-loading utilities ──────────────────────────────
from functions import load_reverse_df, PATH_DF


# ═══════════════════════════════════════════════════════════════════════════
#  1. DATA LOADING  (reuses functions.py logic, outputs full boards)
# ═══════════════════════════════════════════════════════════════════════════

def load_data_full_board(size=10, amount_boards=10000, gen=2,
                         test_size=0.1, val_size=0.1, random_state=365):
    """Load reverse GoL data and return (input=T, target=T-1) as 4-D arrays.

    Returns
    -------
    dict with keys: X_train, X_val, X_test  — shape (n, size, size, 1)  [state T]
                    y_train, y_val, y_test  — shape (n, size, size, 1)  [state T-1]
    """
    reverse_df = load_reverse_df(size, amount_boards, gen)
    n_pixels = size * size

    # Columns 0..(gen-1)*n_pixels-1 = state T (input)
    # Columns (gen-1)*n_pixels..gen*n_pixels-1 = state T-1 (target)
    feat_cols = [f'Col_{i}' for i in range(n_pixels)]
    tgt_cols = [f'Col_{n_pixels + i}' for i in range(n_pixels)]

    X_all = reverse_df[feat_cols].to_numpy().reshape(-1, size, size, 1).astype('float32')
    y_all = reverse_df[tgt_cols].to_numpy().reshape(-1, size, size, 1).astype('float32')

    # Train / val / test split
    X_tv, X_test, y_tv, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_size, random_state=random_state)

    print(f"Data loaded — train: {X_train.shape[0]}, val: {X_val.shape[0]}, "
          f"test: {X_test.shape[0]}")
    return dict(X_train=X_train, X_val=X_val, X_test=X_test,
                y_train=y_train, y_val=y_val, y_test=y_test)


# ═══════════════════════════════════════════════════════════════════════════
#  2. FORWARD CONSISTENCY — vectorised GoL step (numpy & TF versions)
# ═══════════════════════════════════════════════════════════════════════════

def _gol_step_np(boards):
    """Apply one Game-of-Life step to a batch of boards (numpy).

    Parameters
    ----------
    boards : np.ndarray, shape (n, h, w) or (n, h, w, 1), binary {0,1}

    Returns
    -------
    np.ndarray same shape as input, binary {0,1}
    """
    squeeze = False
    if boards.ndim == 4:
        boards = boards[..., 0]
        squeeze = True

    b = boards.astype(np.float32)
    # Toroidal neighbor count via wrap padding + 2-D convolution kernel
    padded = np.pad(b, ((0, 0), (1, 1), (1, 1)), mode='wrap')
    n, h, w = b.shape
    neigh = np.zeros_like(b)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            neigh += padded[:, 1 + di:h + 1 + di, 1 + dj:w + 1 + dj]

    # GoL rules
    alive = b.astype(bool)
    survive = alive & ((neigh == 2) | (neigh == 3))
    birth = (~alive) & (neigh == 3)
    result = (survive | birth).astype(np.float32)
    if squeeze:
        result = result[..., np.newaxis]
    return result


def forward_consistency_np(y_pred_t_minus_1, x_input_t):
    """Compute forward consistency: step y_pred forward and compare to x_input.

    Parameters
    ----------
    y_pred_t_minus_1 : np.ndarray (n, h, w, 1), model output (predicted T-1)
    x_input_t        : np.ndarray (n, h, w, 1), original input   (state T)

    Returns
    -------
    float — fraction of pixels that match (0.0 – 1.0)
    """
    pred_binary = (y_pred_t_minus_1 > 0.5).astype(np.float32)
    stepped = _gol_step_np(pred_binary)
    matches = (stepped == (x_input_t > 0.5).astype(np.float32))
    return float(np.mean(matches))


def _gol_step_tf(boards):
    """Differentiable-friendly GoL forward step in TensorFlow.

    Uses a depthwise conv2d with a fixed [1,1,1;1,0,1;1,1,1] kernel on
    wrap-padded boards to count neighbours, then applies GoL rules.
    """
    # boards: (batch, h, w, 1)  float32, values in [0,1]
    b = boards
    h, w = tf.shape(b)[1], tf.shape(b)[2]

    # Wrap pad (1 pixel each side)
    top = b[:, -1:, :, :]
    bot = b[:, :1, :, :]
    b_p = tf.concat([top, b, bot], axis=1)
    left = b_p[:, :, -1:, :]
    right = b_p[:, :, :1, :]
    b_p = tf.concat([left, b_p, right], axis=2)

    # Neighbour count via depthwise conv (kernel sums the 8 neighbours)
    kernel = tf.constant([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=tf.float32)
    kernel = tf.reshape(kernel, [3, 3, 1, 1])
    neigh = tf.nn.conv2d(b_p, kernel, strides=[1, 1, 1, 1], padding='VALID')

    # Hard GoL rules (non-differentiable but used only in metrics)
    alive = tf.cast(boards > 0.5, tf.float32)
    survive = alive * tf.cast((tf.abs(neigh - 2) < 0.5) | (tf.abs(neigh - 3) < 0.5), tf.float32)
    birth = (1.0 - alive) * tf.cast(tf.abs(neigh - 3) < 0.5, tf.float32)
    return survive + birth


class ForwardConsistency(tf.keras.metrics.Metric):
    """Keras metric: applies GoL forward to prediction and compares to input T."""

    def __init__(self, name='forward_consistency', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_match = self.add_weight('total_match', initializer='zeros')
        self.total_pixels = self.add_weight('total_pixels', initializer='zeros')
        # Will be set externally before each evaluation batch
        self._current_input_t = None

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true is the T-1 ground truth, but we need the original T input.
        # Since Keras doesn't pass the input to metrics, we use the stored
        # reference or fall back to comparing stepped prediction vs y_true
        # (which is still meaningful: pred→step→compare vs what T should be).
        # In the training script we compute FC separately for exact results.
        pred_binary = tf.cast(y_pred > 0.5, tf.float32)
        stepped = _gol_step_tf(pred_binary)
        # We compare stepped(pred) vs the input T.  If _current_input_t is
        # not available, skip (this metric is mainly used in eval).
        if self._current_input_t is not None:
            target = tf.cast(self._current_input_t > 0.5, tf.float32)
        else:
            # Fallback: can't compute without input T
            return
        matches = tf.cast(tf.equal(stepped, target), tf.float32)
        self.total_match.assign_add(tf.reduce_sum(matches))
        self.total_pixels.assign_add(tf.cast(tf.size(matches), tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.total_match, self.total_pixels)

    def reset_state(self):
        self.total_match.assign(0.0)
        self.total_pixels.assign(0.0)


# ═══════════════════════════════════════════════════════════════════════════
#  3. MODEL — Attention Residual U-Net with Wrap Padding
# ═══════════════════════════════════════════════════════════════════════════

def _wrap_pad(x, pad=1):
    """Toroidal wrap-padding as a Lambda layer."""
    top = x[:, -pad:, :, :]
    bot = x[:, :pad, :, :]
    x_p = tf.concat([top, x, bot], axis=1)
    left = x_p[:, :, -pad:, :]
    right = x_p[:, :, :pad, :]
    return tf.concat([left, x_p, right], axis=2)


def _res_conv_block(x, filters, dropout_rate=0.15):
    """Two conv layers with BatchNorm, ReLU, residual shortcut."""
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, padding='same')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Conv2D(filters, 3, padding='same',
                                kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.SpatialDropout2D(dropout_rate)(x)

    x = tf.keras.layers.Conv2D(filters, 3, padding='same',
                                kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x


def _attention_gate(skip, gating, filters):
    """Additive attention gate: learns to focus on relevant skip features."""
    W_skip = tf.keras.layers.Conv2D(filters, 1, padding='same', use_bias=False)(skip)
    W_gate = tf.keras.layers.Conv2D(filters, 1, padding='same', use_bias=False)(gating)
    psi = tf.keras.layers.Activation('relu')(W_skip + W_gate)
    psi = tf.keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid')(psi)
    return tf.keras.layers.Multiply()([skip, psi])


def build_attention_resunet(size=10, pad=2, base_filters=48, depth=3,
                            dropout_rate=0.2):
    """Build an Attention Residual U-Net for image-to-image GoL reversal.

    The encoder operates on wrap-padded inputs to handle toroidal topology.
    The decoder uses attention-gated skip connections and crops back to (size, size).

    Parameters
    ----------
    size : int — board side length (10)
    pad  : int — wrap padding amount
    base_filters : int — filters in first encoder level (doubles each level)
    depth : int — number of encoder/decoder levels
    dropout_rate : float
    """
    padded = size + 2 * pad  # 14 for pad=2

    inp_raw = tf.keras.Input(shape=(size, size, 1), name='input_board_T')

    # Wrap-pad the input
    x = tf.keras.layers.Lambda(_wrap_pad, arguments={'pad': pad},
                                name='wrap_pad')(inp_raw)  # (14,14,1)

    # ─── Encoder ──────────────────────────────────────────────────────
    skips = []
    filters = base_filters
    for i in range(depth):
        x = _res_conv_block(x, filters, dropout_rate)
        skips.append(x)
        x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)
        filters *= 2

    # ─── Bottleneck ───────────────────────────────────────────────────
    x = _res_conv_block(x, filters, dropout_rate)

    # ─── Decoder ──────────────────────────────────────────────────────
    for i in reversed(range(depth)):
        filters //= 2
        x = tf.keras.layers.UpSampling2D(2)(x)

        # Crop x to match skip shape (may differ due to pooling rounding)
        skip = skips[i]
        target_h, target_w = skip.shape[1], skip.shape[2]
        x = tf.keras.layers.Lambda(
            lambda t, th=target_h, tw=target_w: t[:, :th, :tw, :]
        )(x)

        # Attention gate on skip connection
        skip_att = _attention_gate(skip, x, filters)
        x = tf.keras.layers.Concatenate()([x, skip_att])
        x = _res_conv_block(x, filters, dropout_rate)

    # ─── Output: 1×1 conv → sigmoid → crop to original size ──────────
    x = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', name='pred_raw')(x)
    x = tf.keras.layers.Cropping2D(cropping=((pad, pad), (pad, pad)),
                                    name='output_T_minus_1')(x)

    model = tf.keras.Model(inp_raw, x, name='AttentionResUNet')
    return model


# ═══════════════════════════════════════════════════════════════════════════
#  4. D4 AUGMENTATION  (GoL rules are rotationally symmetric)
# ═══════════════════════════════════════════════════════════════════════════

def augment_d4(X, y):
    """8× augmentation via the dihedral group D4 (rotations + flips)."""
    Xs, ys = [X], [y]
    for k in range(1, 4):
        Xs.append(np.rot90(X, k, axes=(1, 2)))
        ys.append(np.rot90(y, k, axes=(1, 2)))
    X_flip = X[:, :, ::-1, :]
    y_flip = y[:, :, ::-1, :]
    Xs.append(X_flip)
    ys.append(y_flip)
    for k in range(1, 4):
        Xs.append(np.rot90(X_flip, k, axes=(1, 2)))
        ys.append(np.rot90(y_flip, k, axes=(1, 2)))
    return np.concatenate(Xs), np.concatenate(ys)


# ═══════════════════════════════════════════════════════════════════════════
#  5. TRAINING + EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def train_and_evaluate(size=10, amount_boards=10000, gen=2,
                       base_filters=48, depth=3, pad=2,
                       dropout_rate=0.2, learning_rate=1e-3, weight_decay=5e-4,
                       batch_size=256, epochs=80, use_augmentation=True):
    """Full pipeline: load data → build model → train → evaluate.

    Returns dict with model, history, and all metrics.
    """
    # ── Load data ─────────────────────────────────────────────────────
    data = load_data_full_board(size, amount_boards, gen)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']

    # ── D4 augmentation ──────────────────────────────────────────────
    if use_augmentation:
        X_train, y_train = augment_d4(X_train, y_train)
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(X_train))
        X_train, y_train = X_train[idx], y_train[idx]
        print(f"After D4 augmentation: {X_train.shape[0]} training samples")

    # ── Build model ──────────────────────────────────────────────────
    tf.keras.backend.clear_session()
    model = build_attention_resunet(size, pad, base_filters, depth, dropout_rate)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.03),
        metrics=['accuracy'],
    )
    model.summary()
    print(f"Total parameters: {model.count_params():,}")

    # ── Callbacks ─────────────────────────────────────────────────────
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=12,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5,
            min_lr=1e-6, verbose=1),
    ]

    # ── Train ─────────────────────────────────────────────────────────
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    # ── Evaluate ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION ON TEST SET")
    print("=" * 60)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Standard pixel accuracy : {test_acc:.4f}")

    y_pred = model.predict(X_test, verbose=0)
    fc_score = forward_consistency_np(y_pred, X_test)
    print(f"Forward Consistency     : {fc_score:.4f}")

    # Per-class breakdown
    pred_bin = (y_pred > 0.5).astype(int)
    true_bin = y_test.astype(int)
    alive_mask = true_bin.flatten() == 1
    dead_mask = true_bin.flatten() == 0
    alive_acc = np.mean(pred_bin.flatten()[alive_mask] == 1) if alive_mask.any() else 0
    dead_acc = np.mean(pred_bin.flatten()[dead_mask] == 0) if dead_mask.any() else 0
    print(f"Accuracy on ALIVE cells : {alive_acc:.4f}")
    print(f"Accuracy on DEAD cells  : {dead_acc:.4f}")

    # Validation best
    best_val_acc = max(history.history['val_accuracy'])
    best_train_acc = max(history.history['accuracy'])
    print(f"\nBest train accuracy     : {best_train_acc:.4f}")
    print(f"Best val accuracy       : {best_val_acc:.4f}")
    print(f"Overfitting gap         : {best_train_acc - best_val_acc:.4f}")

    # Forward consistency on validation too
    y_pred_val = model.predict(X_val, verbose=0)
    fc_val = forward_consistency_np(y_pred_val, X_val)
    print(f"Val Forward Consistency : {fc_val:.4f}")

    # ── Save if good ──────────────────────────────────────────────────
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir,
                             f"AttResUNet_{size}x{size}_acc{test_acc:.4f}_fc{fc_score:.4f}.keras")
    model.save(save_path)
    print(f"\nModel saved → {save_path}")

    print("=" * 60)

    return {
        'model': model,
        'history': history,
        'test_accuracy': test_acc,
        'forward_consistency_test': fc_score,
        'forward_consistency_val': fc_val,
        'best_val_accuracy': best_val_acc,
        'alive_accuracy': alive_acc,
        'dead_accuracy': dead_acc,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  6. STANDALONE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = train_and_evaluate(
        size=10,
        amount_boards=10000,
        gen=2,
        base_filters=48,
        depth=3,
        pad=2,
        dropout_rate=0.2,
        learning_rate=1e-3,
        weight_decay=5e-4,
        batch_size=256,
        epochs=80,
        use_augmentation=True,
    )
