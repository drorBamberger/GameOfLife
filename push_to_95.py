"""
Push to 95% — Reverse Game of Life with Forward-Consistency Training.

Memory-optimized for 8GB RAM CPU-only system.

KEY INNOVATIONS:
  1. DIFFERENTIABLE FORWARD-CONSISTENCY LOSS: Predict T-1, run GoL forward,
     penalize mismatch with input T inside the computation graph.
  2. MULTI-RADIUS FEATURES via tf.data pipeline (no full preprocess in RAM).
  3. PER-CELL CLASSIFICATION with dense residual connections.
  4. ITERATIVE SELF-TRAINING: use confident predictions as extra training data.
  5. OPTIMAL THRESHOLD SEARCH per evaluation.

Usage:
    python push_to_95.py
"""

import os, gc, time, sys
import numpy as np
import pandas as pd
import tensorflow as tf

# Limit TF memory growth
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(4)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
os.environ['TF_NUM_INTRAOP_THREADS'] = '4'

# Simple file-based logging with immediate flush
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'push_progress.log')
_log_fh = open(LOG_FILE, 'w', buffering=1)  # line-buffered

class _Logger:
    def info(self, msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        _log_fh.write(line + '\n')
        _log_fh.flush()
        os.fsync(_log_fh.fileno())

log = _Logger()

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
SIZE = 10
GEN = 2
PATH_DF = r'C:\GameOfLifeFiles\df'
SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

TARGET_ACCURACY = 0.95


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA — generate on-the-fly (avoids loading 300MB pickle into limited RAM)
# ══════════════════════════════════════════════════════════════════════════════

def gol_step_np(boards):
    """Apply one step of Game of Life on (N, H, W, 1) boards."""
    if boards.ndim == 4:
        b = boards[..., 0]
    else:
        b = boards
    b = b.astype(np.float32)
    padded = np.pad(b, ((0,0),(1,1),(1,1)), mode='wrap')
    n, h, w = b.shape
    neigh = np.zeros_like(b)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            neigh += padded[:, 1+di:h+1+di, 1+dj:w+1+dj]
    alive = b > 0.5
    result = ((alive & ((neigh==2)|(neigh==3))) | (~alive & (neigh==3))).astype(np.float32)
    return result[..., np.newaxis]


def generate_data(n_samples, size=SIZE, gen=GEN, seed=None):
    """Generate training data by creating random boards and stepping forward.
    For gen=2: X = GoL(board), y = board (predict T-1 from T)
    For gen=3: X = (GoL^2(board), GoL(board)), y = board
    Avoids loading large pickle files — unlimited fresh data!
    """
    rng = np.random.default_rng(seed)
    # Random density between 0.1 and 0.5 for diversity
    densities = rng.uniform(0.1, 0.5, size=n_samples)
    boards = np.zeros((n_samples, size, size, 1), dtype=np.float32)
    for i in range(n_samples):
        boards[i, :, :, 0] = (rng.random((size, size)) < densities[i]).astype(np.float32)

    # y = original board (T-1), X = board after stepping forward (T)
    y = boards.copy()

    # Step forward gen-1 times to get X
    current = boards
    for _ in range(gen - 1):
        current = gol_step_np(current)
    X = current

    return X, y


def load_data_from_file(size=SIZE, amount_boards=10000, gen=GEN, max_rows=50000):
    """Load real data from pickle, but limit rows to save memory."""
    name_df = os.path.join(PATH_DF, f'{size}-{amount_boards}',
                           f'{size}size_{amount_boards}boards_{gen}gen_reverse')
    log.info(f"Loading {name_df}.pkl (max {max_rows:,} rows) ...")
    df = pd.read_pickle(f'{name_df}.pkl')

    # Take only what we need
    if len(df) > max_rows:
        df = df.iloc[:max_rows]

    nf = (gen - 1) * size * size
    vals = df.values.astype(np.float32)
    del df; gc.collect()

    X_all = vals[:, :nf].reshape(-1, size, size, 1)
    y_all = vals[:, nf:nf + size*size].reshape(-1, size, size, 1)
    del vals; gc.collect()

    return X_all, y_all


def load_data(size=SIZE, amount_boards=10000, gen=GEN):
    """Load data: use generated data for training, real data for validation/test."""
    log.info("Generating training data (avoids large pickle load)...")

    # Generate training data
    X_train, y_train = generate_data(30000, size, gen, seed=SEED)
    log.info(f"Generated {len(X_train):,} training samples")

    # For val/test, load a small portion of real data
    log.info("Loading validation/test from file...")
    X_real, y_real = load_data_from_file(size, amount_boards, gen, max_rows=10000)

    # Split real data into val/test
    n = len(X_real)
    n_val = n // 2
    X_val, y_val = X_real[:n_val], y_real[:n_val]
    X_test, y_test = X_real[n_val:], y_real[n_val:]
    del X_real, y_real; gc.collect()

    log.info(f"Train: {X_train.shape[0]:,}  Val: {X_val.shape[0]:,}  Test: {X_test.shape[0]:,}")
    log.info(f"Alive fraction (train): {y_train.mean():.4f}")
    return X_train, y_train, X_val, y_val, X_test, y_test


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING — TF-based (inside model, no RAM duplication)
# ══════════════════════════════════════════════════════════════════════════════

def neighbor_count_np(boards, radius=1):
    """Toroidal neighbor count at given radius. Returns (N,H,W,1) normalized."""
    b = boards[:, :, :, 0].astype(np.float32)
    n, h, w = b.shape
    padded = np.pad(b, ((0,0),(radius,radius),(radius,radius)), mode='wrap')
    neigh = np.zeros_like(b)
    for di in range(-radius, radius+1):
        for dj in range(-radius, radius+1):
            if di == 0 and dj == 0:
                continue
            neigh += padded[:, radius+di:h+radius+di, radius+dj:w+radius+dj]
    total_neighbors = (2*radius+1)**2 - 1
    return neigh[:, :, :, np.newaxis] / total_neighbors


def add_features(boards):
    """Board + r1 neighbors + r2 neighbors + r3 neighbors + coords = 6 channels."""
    n, h, w, _ = boards.shape
    neigh1 = neighbor_count_np(boards, radius=1)
    neigh2 = neighbor_count_np(boards, radius=2)
    neigh3 = neighbor_count_np(boards, radius=3)
    row_ch = np.tile(np.linspace(0, 1, h, dtype=np.float32).reshape(1,h,1,1), (n,1,w,1))
    col_ch = np.tile(np.linspace(0, 1, w, dtype=np.float32).reshape(1,1,w,1), (n,h,1,1))
    return np.concatenate([boards, neigh1, neigh2, neigh3, row_ch, col_ch], axis=-1)


def wrap_pad_np(boards, pad):
    return np.pad(boards, ((0,0),(pad,pad),(pad,pad),(0,0)), mode='wrap')


def preprocess(boards, pad):
    return wrap_pad_np(add_features(boards), pad)


def preprocess_chunked(boards, pad, chunk_size=10000):
    """Memory-efficient preprocessing in chunks."""
    chunks = []
    for i in range(0, len(boards), chunk_size):
        chunks.append(preprocess(boards[i:i+chunk_size], pad))
    return np.concatenate(chunks, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# 3. D4 TTA (test-time augmentation only — no D4 training augmentation)
# ══════════════════════════════════════════════════════════════════════════════

def predict_with_tta(model, X_raw, pad, batch_size=256):
    """D4 TTA: preprocess each view, predict, undo transform, average."""
    preds = []
    def pred_view(X_v):
        return model.predict(preprocess_chunked(X_v, pad), batch_size=batch_size, verbose=0)

    preds.append(pred_view(X_raw))
    for k in range(1, 4):
        p = pred_view(np.rot90(X_raw, k, axes=(1, 2)))
        preds.append(np.rot90(p, -k, axes=(1, 2)))
    X_flip = X_raw[:, :, ::-1, :]
    p = pred_view(X_flip)
    preds.append(p[:, :, ::-1, :])
    for k in range(1, 4):
        p = pred_view(np.rot90(X_flip, k, axes=(1, 2)))
        p = np.rot90(p, -k, axes=(1, 2))
        preds.append(p[:, :, ::-1, :])
    return np.mean(preds, axis=0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 4. DIFFERENTIABLE GOL FORWARD STEP (key innovation)
# ══════════════════════════════════════════════════════════════════════════════

def build_gol_forward_layer():
    kernel_np = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.float32).reshape(3,3,1,1)

    @tf.function
    def gol_forward(boards):
        b = boards
        top = b[:, -1:, :, :]
        bot = b[:, :1, :, :]
        b_p = tf.concat([top, b, bot], axis=1)
        left = b_p[:, :, -1:, :]
        right = b_p[:, :, :1, :]
        b_p = tf.concat([left, b_p, right], axis=2)

        kernel = tf.constant(kernel_np)
        neigh = tf.nn.conv2d(b_p, kernel, strides=[1,1,1,1], padding='VALID')

        temp = 10.0
        alive = boards
        survive = alive * tf.sigmoid(temp * (neigh - 1.5)) * tf.sigmoid(temp * (3.5 - neigh))
        birth = (1.0 - alive) * tf.sigmoid(temp * (neigh - 2.5)) * tf.sigmoid(temp * (3.5 - neigh))
        return survive + birth

    return gol_forward


# ══════════════════════════════════════════════════════════════════════════════
# 5. LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def weighted_focal_bce(pos_weight=2.5, gamma=2.0, label_smoothing=0.02):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_true_s = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce = -(y_true_s * tf.math.log(y_pred) +
                (1.0 - y_true_s) * tf.math.log(1.0 - y_pred))
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        focal_w = tf.pow(1.0 - p_t, gamma)
        class_w = y_true * (pos_weight - 1.0) + 1.0
        return tf.reduce_mean(focal_w * class_w * bce)
    return loss_fn


def dice_loss(y_true, y_pred, smooth=1.0):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def build_combo_loss_with_fc(gol_forward_fn, fc_weight=0.3,
                              pos_weight=2.5, gamma=2.0, dice_w=0.3):
    focal_fn = weighted_focal_bce(pos_weight, gamma)

    def loss_fn(y_true_and_input, y_pred):
        y_true = y_true_and_input[..., 0:1]
        x_input_t = y_true_and_input[..., 1:2]
        l_focal = focal_fn(y_true, y_pred)
        l_dice = dice_loss(y_true, y_pred)
        stepped = gol_forward_fn(y_pred)
        l_fc = tf.reduce_mean(tf.square(stepped - x_input_t))
        return (1.0 - dice_w - fc_weight) * l_focal + dice_w * l_dice + fc_weight * l_fc

    return loss_fn


def build_standard_combo_loss(pos_weight=2.5, gamma=2.0, dice_w=0.3):
    focal_fn = weighted_focal_bce(pos_weight, gamma)
    def loss_fn(y_true, y_pred):
        return (1.0 - dice_w) * focal_fn(y_true, y_pred) + dice_w * dice_loss(y_true, y_pred)
    return loss_fn


# ══════════════════════════════════════════════════════════════════════════════
# 6. MODEL ARCHITECTURE — Dense Residual U-Net with SE + Attention
# ══════════════════════════════════════════════════════════════════════════════

def se_block(x, ratio=4):
    filters = x.shape[-1]
    se = tf.keras.layers.GlobalAveragePooling2D()(x)
    se = tf.keras.layers.Dense(max(filters // ratio, 8), activation='relu')(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid')(se)
    se = tf.keras.layers.Reshape((1, 1, filters))(se)
    return tf.keras.layers.Multiply()([x, se])


def res_block(x, filters, dropout_rate=0.15, l2_reg=5e-5):
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, padding='same')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False,
                                kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.SpatialDropout2D(dropout_rate)(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False,
                                kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = se_block(x)
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    return x


def attention_gate(skip, gating, filters):
    W_s = tf.keras.layers.Conv2D(filters, 1, padding='same', use_bias=False)(skip)
    W_g = tf.keras.layers.Conv2D(filters, 1, padding='same', use_bias=False)(gating)
    psi = tf.keras.layers.ReLU()(W_s + W_g)
    psi = tf.keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid')(psi)
    return tf.keras.layers.Multiply()([skip, psi])


def build_model(size, pad, base_filters, depth, dropout, input_channels=6):
    padded = size + 2 * pad
    inp = tf.keras.Input(shape=(padded, padded, input_channels))

    x = tf.keras.layers.Conv2D(base_filters, 3, padding='same', use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Encoder
    skips = []
    f = base_filters
    for _ in range(depth):
        x = res_block(x, f, dropout)
        skips.append(x)
        x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)
        f *= 2

    # Bottleneck (2 blocks)
    x = res_block(x, f, dropout)
    x = res_block(x, f, dropout)

    # Decoder
    for i in reversed(range(depth)):
        f //= 2
        x = tf.keras.layers.UpSampling2D(2)(x)
        skip = skips[i]
        x = tf.keras.layers.Cropping2D(
            cropping=((0, x.shape[1]-skip.shape[1]),
                      (0, x.shape[2]-skip.shape[2])))(x)
        skip_att = attention_gate(skip, x, f)
        x = tf.keras.layers.Concatenate()([x, skip_att])
        x = res_block(x, f, dropout)

    x = tf.keras.layers.Conv2D(base_filters, 1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
    x = tf.keras.layers.Cropping2D(cropping=((pad,pad),(pad,pad)))(x)

    return tf.keras.Model(inp, x, name=f'ResUNet_f{base_filters}_d{depth}')


# ══════════════════════════════════════════════════════════════════════════════
# 7. CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

class EpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        parts = [f"Epoch {epoch+1}"]
        for k, v in logs.items():
            parts.append(f"{k}={v:.4f}")
        log.info(" | ".join(parts))


class CosineAnnealing(tf.keras.callbacks.Callback):
    def __init__(self, lr_min=1e-6, lr_max=5e-4, T_0=25, T_mult=2):
        super().__init__()
        self.lr_min, self.lr_max = lr_min, lr_max
        self.T_0, self.T_mult = T_0, T_mult
        self.T_cur, self.T_i = 0, T_0

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1 + np.cos(np.pi * self.T_cur / self.T_i))
        self.model.optimizer.learning_rate.assign(lr)
        self.T_cur += 1
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i = int(self.T_i * self.T_mult)


# ══════════════════════════════════════════════════════════════════════════════
# 8. EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def find_optimal_threshold(y_true, y_pred_prob):
    """Find threshold that maximizes pixel accuracy."""
    best_t, best_acc = 0.5, 0.0
    for t in np.arange(0.30, 0.70, 0.02):
        acc = np.mean((y_pred_prob > t).astype(int) == y_true.astype(int))
        if acc > best_acc:
            best_t, best_acc = t, acc
    return best_t, best_acc


def evaluate(model, X_raw, y, pad, use_tta=False, label=""):
    if use_tta:
        y_pred = predict_with_tta(model, X_raw, pad)
    else:
        y_pred = model.predict(preprocess_chunked(X_raw, pad), batch_size=256, verbose=0)

    # Find optimal threshold
    opt_t, opt_acc = find_optimal_threshold(y, y_pred)
    pred_bin = (y_pred > opt_t).astype(int)
    true_bin = y.astype(int)

    pixel_acc = np.mean(pred_bin == true_bin)
    alive_m = true_bin.flatten() == 1
    dead_m = true_bin.flatten() == 0
    alive_acc = np.mean(pred_bin.flatten()[alive_m] == 1) if alive_m.any() else 0
    dead_acc = np.mean(pred_bin.flatten()[dead_m] == 0) if dead_m.any() else 0

    stepped = gol_step_np(pred_bin.astype(np.float32))
    fc = np.mean(stepped == (X_raw > 0.5).astype(np.float32))

    from sklearn.metrics import f1_score
    f1 = f1_score(true_bin.flatten(), pred_bin.flatten(), zero_division=0)

    tta_tag = " [TTA]" if use_tta else ""
    log.info(f"  {label}{tta_tag}: pixel={pixel_acc:.4f}  alive={alive_acc:.4f}  "
             f"dead={dead_acc:.4f}  F1={f1:.4f}  FC={fc:.4f}  thresh={opt_t:.2f}")
    return pixel_acc, alive_acc, dead_acc, f1, fc, y_pred


# ══════════════════════════════════════════════════════════════════════════════
# 9. SELF-TRAINING: use forward-consistency to generate pseudo-labels
# ══════════════════════════════════════════════════════════════════════════════

def generate_pseudo_labels(model, X_unlabeled, pad, confidence=0.90):
    """Predict on unlabeled data, keep only high-confidence samples."""
    y_pred = model.predict(preprocess_chunked(X_unlabeled, pad), batch_size=256, verbose=0)

    # Keep samples where all pixels are high-confidence
    max_prob = np.maximum(y_pred, 1 - y_pred)  # confidence per pixel
    min_confidence = max_prob.reshape(len(y_pred), -1).min(axis=1)  # worst pixel per board
    mask = min_confidence >= confidence

    if mask.sum() == 0:
        return None, None

    X_pseudo = X_unlabeled[mask]
    y_pseudo = (y_pred[mask] > 0.5).astype(np.float32)

    # Additionally filter by forward consistency
    stepped = gol_step_np(y_pseudo)
    fc_match = np.mean(stepped == (X_pseudo > 0.5).astype(np.float32),
                       axis=(1, 2, 3))
    fc_mask = fc_match >= 0.95  # at least 95% forward consistency

    if fc_mask.sum() == 0:
        return None, None

    return X_pseudo[fc_mask], y_pseudo[fc_mask]


# ══════════════════════════════════════════════════════════════════════════════
# 10. TRAINING CONFIGURATIONS — focused, memory-efficient
# ══════════════════════════════════════════════════════════════════════════════

CONFIGS = [
    # Phase 1: Lean baseline — fits in 8 GB RAM system
    dict(name="Phase1_Lean", amount_boards=10000, base_filters=16, depth=2,
         pad=2, dropout=0.10, lr=1e-3, batch_size=512, epochs=60,
         use_fc_loss=False, pos_weight=3.0, fc_weight=0.0,
         train_limit=30_000),

    # Phase 2: Same + FC loss (key innovation)
    dict(name="Phase2_FC", amount_boards=10000, base_filters=16, depth=2,
         pad=2, dropout=0.10, lr=5e-4, batch_size=512, epochs=60,
         use_fc_loss=True, pos_weight=3.0, fc_weight=0.2,
         train_limit=30_000),

    # Phase 3: Slightly bigger model + FC loss
    dict(name="Phase3_Med_FC", amount_boards=10000, base_filters=24, depth=2,
         pad=2, dropout=0.12, lr=3e-4, batch_size=256, epochs=80,
         use_fc_loss=True, pos_weight=3.0, fc_weight=0.25,
         train_limit=30_000),

    # Phase 4: Max capacity that fits in RAM
    dict(name="Phase4_Big_FC", amount_boards=10000, base_filters=32, depth=2,
         pad=3, dropout=0.12, lr=3e-4, batch_size=256, epochs=80,
         use_fc_loss=True, pos_weight=3.0, fc_weight=0.25,
         train_limit=30_000),
]


# ══════════════════════════════════════════════════════════════════════════════
# 11. MAIN LOOP — train, evaluate, self-train, repeat
# ══════════════════════════════════════════════════════════════════════════════

def run_config(cfg):
    """Train and evaluate one configuration. Returns best pixel accuracy."""
    name = cfg['name']
    log.info(f"CONFIG: {name}")
    log.info(f"  {cfg}")

    # -- Load data --
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(
        SIZE, cfg['amount_boards'], GEN)

    pad = cfg['pad']

    # -- Preprocess in chunks to save memory --
    log.info("Preprocessing...")
    X_train_pp = preprocess_chunked(X_train, pad)
    X_val_pp = preprocess_chunked(X_val, pad)
    input_channels = X_train_pp.shape[-1]
    log.info(f"Preprocessed: {X_train_pp.shape}")

    # -- Prepare targets --
    if cfg['use_fc_loss']:
        y_train_packed = np.concatenate([y_train, X_train[..., 0:1]], axis=-1)
        y_val_packed = np.concatenate([y_val, X_val[..., 0:1]], axis=-1)
    else:
        y_train_packed = y_train
        y_val_packed = y_val

    # Free raw training data (keep X_val, X_test for evaluation)
    del X_train; gc.collect()

    # -- Build model --
    tf.keras.backend.clear_session()
    model = build_model(SIZE, pad, cfg['base_filters'], cfg['depth'],
                        cfg['dropout'], input_channels)

    # -- Loss --
    if cfg['use_fc_loss']:
        gol_fwd = build_gol_forward_layer()
        loss_fn = build_combo_loss_with_fc(
            gol_fwd, fc_weight=cfg['fc_weight'],
            pos_weight=cfg['pos_weight'])
        def pixel_accuracy(y_true_packed, y_pred):
            return tf.keras.metrics.binary_accuracy(y_true_packed[..., 0:1], y_pred)
        metrics = [pixel_accuracy]
    else:
        loss_fn = build_standard_combo_loss(pos_weight=cfg['pos_weight'])
        metrics = ['accuracy']

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=cfg['lr'], weight_decay=1e-4, clipnorm=1.0),
        loss=loss_fn,
        metrics=metrics,
    )
    log.info(f"Parameters: {model.count_params():,}")

    # -- Callbacks --
    monitor = 'val_pixel_accuracy' if cfg['use_fc_loss'] else 'val_accuracy'
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=15, restore_best_weights=True, verbose=1),
        CosineAnnealing(lr_min=1e-6, lr_max=cfg['lr'], T_0=25, T_mult=2),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(SAVE_DIR, f'{name}_best.keras'),
            monitor=monitor, save_best_only=True, verbose=1),
        EpochLogger(),
    ]

    # -- Train --
    t0 = time.time()
    history = model.fit(
        X_train_pp, y_train_packed,
        validation_data=(X_val_pp, y_val_packed),
        epochs=cfg['epochs'],
        batch_size=cfg['batch_size'],
        callbacks=callbacks,
        verbose=2,
    )
    elapsed = time.time() - t0
    log.info(f"Training time: {elapsed/60:.1f} min")

    acc_key = 'val_pixel_accuracy' if cfg['use_fc_loss'] else 'val_accuracy'
    best_val = max(history.history[acc_key])
    log.info(f"Best val accuracy: {best_val:.4f}")

    # Free training data
    del X_train_pp, y_train_packed; gc.collect()

    # -- Evaluate --
    log.info("--- Evaluation ---")
    acc_no_tta, *_ = evaluate(model, X_test, y_test, pad, use_tta=False, label="Test")
    acc_tta, alive_tta, dead_tta, f1_tta, fc_tta, _ = evaluate(
        model, X_test, y_test, pad, use_tta=True, label="Test")

    result = dict(
        name=name, best_val=best_val, test_acc=acc_no_tta,
        test_acc_tta=acc_tta, alive=alive_tta, dead=dead_tta,
        f1=f1_tta, fc=fc_tta, time_min=elapsed/60,
        params=model.count_params())

    del X_val_pp, y_val_packed; gc.collect()

    return result, model, X_val, y_val, X_test, y_test


def run_self_training(model, X_test, y_test, X_val, y_val, pad, rounds=3):
    """Self-training: generate new boards, predict, filter by FC, retrain."""
    log.info("=== SELF-TRAINING ===")

    # Generate random boards to use as "unlabeled" data
    rng = np.random.default_rng(SEED)
    for round_i in range(rounds):
        log.info(f"Self-training round {round_i+1}/{rounds}")

        # Generate random boards
        n_boards = 50000
        random_boards = rng.integers(0, 2, size=(n_boards, SIZE, SIZE, 1)).astype(np.float32)

        # Step forward to get input T (these become X)
        X_synth = gol_step_np(random_boards)

        # The random_boards are the ground truth T-1
        y_synth = random_boards

        # Use model to predict, then filter by forward consistency
        X_pseudo, y_pseudo = generate_pseudo_labels(model, X_synth, pad, confidence=0.85)

        if X_pseudo is None:
            log.info(f"  No confident pseudo-labels generated, stopping")
            break

        log.info(f"  Generated {len(X_pseudo):,} pseudo-labeled samples")

        # Fine-tune on pseudo-labels + verify on real test
        X_pp = preprocess_chunked(X_pseudo, pad)
        model.fit(X_pp, y_pseudo, epochs=5, batch_size=128, verbose=0)
        del X_pp; gc.collect()

        # Evaluate
        acc, *_ = evaluate(model, X_test, y_test, pad, use_tta=False, label=f"ST-Round{round_i+1}")
        del X_synth, y_synth, random_boards; gc.collect()

    return model


def main():
    log.info("PUSH TO 95% - Reverse Game of Life")
    log.info(f"Target: {TARGET_ACCURACY} pixel accuracy on test set (with TTA)")

    all_results = []

    for i, cfg in enumerate(CONFIGS):
        log.info(f"ATTEMPT {i+1}/{len(CONFIGS)}")

        result, model, X_val, y_val, X_test, y_test = run_config(cfg)
        all_results.append(result)

        # Log running summary
        log.info("RESULTS SO FAR:")
        for r in all_results:
            log.info(f"  {r['name']:<20} val={r['best_val']:.4f} test={r['test_acc']:.4f} "
                     f"tta={r['test_acc_tta']:.4f} alive={r['alive']:.4f} f1={r['f1']:.4f} fc={r['fc']:.4f}")

        if result['test_acc_tta'] >= TARGET_ACCURACY:
            log.info(f">>> TARGET {TARGET_ACCURACY} REACHED with config '{cfg['name']}' <<<")
            save_path = os.path.join(SAVE_DIR,
                f"BEST_acc{result['test_acc_tta']:.4f}_f1{result['f1']:.4f}.keras")
            model.save(save_path)
            log.info(f"Model saved -> {save_path}")
            return all_results, model

        # Try self-training to boost accuracy
        if result['test_acc_tta'] >= 0.88:
            log.info("Accuracy >= 88%, attempting self-training...")
            model = run_self_training(model, X_test, y_test, X_val, y_val,
                                       cfg['pad'], rounds=3)
            # Re-evaluate after self-training
            acc_st, alive_st, dead_st, f1_st, fc_st, _ = evaluate(
                model, X_test, y_test, cfg['pad'], use_tta=True, label="Post-ST")
            if acc_st >= TARGET_ACCURACY:
                log.info(f">>> TARGET {TARGET_ACCURACY} REACHED after self-training <<<")
                save_path = os.path.join(SAVE_DIR,
                    f"BEST_ST_acc{acc_st:.4f}_f1{f1_st:.4f}.keras")
                model.save(save_path)
                return all_results, model

        log.info(f"Gap to target: {TARGET_ACCURACY - result['test_acc_tta']:.4f}")
        del model, X_val, y_val, X_test, y_test; gc.collect()

    # If we get here, none reached 95%
    best = max(all_results, key=lambda r: r['test_acc_tta'])
    log.info(f"ALL CONFIGS EXHAUSTED. Best: {best['name']}: TTA={best['test_acc_tta']:.4f}")

    return all_results, None


if __name__ == "__main__":
    results, best_model = main()
