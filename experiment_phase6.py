"""
experiment_phase6.py — Break the 95% Barrier: Alive-Cell Recall Attack

Two novel architectures + new loss functions to push past 90.6% overall accuracy
and dramatically improve alive-cell recall (currently bottlenecked at ~72.8%).

Architecture 1 — GoL Transformer (GoLT):
    CNN tokenizer extracts local neighborhood features per cell, then a
    Transformer encoder captures global dependencies via full self-attention.
    Toroidal sinusoidal positional encoding respects the wrap-around topology.

Architecture 2 — Physics-Informed CellAttentionNet (PI-CAN):
    CBAM (Channel + Spatial) attention residual blocks with toroidal padding.
    A custom train_step injects a "soft GoL" physics loss: the differentiable
    forward GoL step applied to predictions should reproduce the input T.

New loss functions:
    • Tversky Loss (alpha=0.3, beta=0.7) — heavily penalizes FN (missed alive)
    • Combo Loss = 0.5 * Tversky + 0.5 * Focal — balances recall and precision

Post-processing:
    • Test-Time Augmentation (D4 ensemble) — free ~0.5% boost
    • Dynamic threshold search optimised for alive F1

Data shapes (both models use 2-D output):
    X: (n, 10, 10, 1) — board at time T
    y: (n, 10, 10, 1) — board at time T-1 (target)
"""

import os, sys, gc, time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# ── path so we can import functions.py ───────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SIZE   = 10
GEN    = 2
AMOUNT = 100_000
PATH_DF = 'C:\\GameOfLifeFiles\\df\\'
os.makedirs("models", exist_ok=True)
np.random.seed(42)
tf.random.set_seed(42)


# ════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING  (y kept as 10×10×1 for image-to-image models)
# ════════════════════════════════════════════════════════════════════════════

def load_data():
    pkl = (f"{PATH_DF}{SIZE}-{AMOUNT}\\"
           f"{SIZE}size_{AMOUNT}boards_{GEN}gen_reverse.pkl")
    print(f"Loading {pkl} …", flush=True)
    df = pd.read_pickle(pkl)
    n_feat = (GEN - 1) * SIZE * SIZE   # 100 — state T columns

    X_all = df.iloc[:, :n_feat].to_numpy(dtype=np.float32
                ).reshape(-1, SIZE, SIZE, 1)
    y_all = df.iloc[:, n_feat:n_feat + SIZE * SIZE].to_numpy(dtype=np.float32
                ).reshape(-1, SIZE, SIZE, 1)
    del df; gc.collect()

    idx_tv, idx_te = train_test_split(np.arange(len(X_all)),
                                       test_size=0.10, random_state=365)
    idx_tr, idx_va = train_test_split(idx_tv,
                                       test_size=0.10, random_state=365)
    split = dict(
        X_train=X_all[idx_tr], y_train=y_all[idx_tr],
        X_val  =X_all[idx_va], y_val  =y_all[idx_va],
        X_test =X_all[idx_te], y_test =y_all[idx_te],
    )
    alive_ratio = float(y_all[idx_tr].mean())
    print(f"  train={len(idx_tr):,}  val={len(idx_va):,}  test={len(idx_te):,}", flush=True)
    print(f"  alive ratio in train = {alive_ratio:.3f}", flush=True)
    return split, alive_ratio


# ════════════════════════════════════════════════════════════════════════════
# 2.  CUSTOM LOSS FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def focal_loss(gamma=2.0, alpha=0.6):
    """Standard focal loss: alpha weights alive class, gamma down-weights easy pixels."""
    def _loss(y_true, y_pred):
        y_pred  = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        p_t     = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        bce     = -(y_true * tf.math.log(y_pred)
                    + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        return tf.reduce_mean(alpha_t * tf.pow(1.0 - p_t, gamma) * bce)
    return _loss


def tversky_loss(alpha=0.3, beta=0.7):
    """
    Tversky Loss — generalisation of Dice Loss.
    alpha < beta → penalises FN (missed alive cells) more than FP.
    Recommended: alpha=0.3, beta=0.7 for high-recall scenarios.
    """
    def _loss(y_true, y_pred):
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        TP = tf.reduce_sum(y_true_f * y_pred_f)
        FP = tf.reduce_sum((1.0 - y_true_f) * y_pred_f)
        FN = tf.reduce_sum(y_true_f * (1.0 - y_pred_f))
        return 1.0 - (TP + 1e-6) / (TP + alpha * FP + beta * FN + 1e-6)
    return _loss


def combo_loss(tv_alpha=0.3, tv_beta=0.7, f_gamma=2.0, f_alpha=0.6, ratio=0.5):
    """
    Combo Loss = ratio * Tversky + (1-ratio) * Focal.
    Tversky drives recall; Focal prevents precision collapse.
    """
    _tv = tversky_loss(tv_alpha, tv_beta)
    _fl = focal_loss(f_gamma, f_alpha)
    def _loss(y_true, y_pred):
        return ratio * _tv(y_true, y_pred) + (1.0 - ratio) * _fl(y_true, y_pred)
    return _loss


# ════════════════════════════════════════════════════════════════════════════
# 3.  CUSTOM METRICS  — alive-class precision / recall / F1
# ════════════════════════════════════════════════════════════════════════════

class _BinaryClassMetric(tf.keras.metrics.Metric):
    """Base for precision / recall / F1 on the positive (alive) class."""
    def __init__(self, threshold=0.5, name='metric', **kw):
        super().__init__(name=name, **kw)
        self.threshold = threshold
        self.tp = self.add_weight('tp', initializer='zeros')
        self.pp = self.add_weight('pp', initializer='zeros')
        self.ap = self.add_weight('ap', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        yp = tf.cast(y_pred > self.threshold, tf.float32)
        yt = tf.cast(y_true > 0.5, tf.float32)
        self.tp.assign_add(tf.reduce_sum(yt * yp))
        self.pp.assign_add(tf.reduce_sum(yp))
        self.ap.assign_add(tf.reduce_sum(yt))

    def reset_state(self):
        for v in (self.tp, self.pp, self.ap):
            v.assign(0.0)


class AlivePrecision(_BinaryClassMetric):
    def __init__(self, threshold=0.5, **kw):
        super().__init__(threshold, name='alive_precision', **kw)
    def result(self):
        return self.tp / (self.pp + 1e-7)


class AliveRecall(_BinaryClassMetric):
    def __init__(self, threshold=0.5, **kw):
        super().__init__(threshold, name='alive_recall', **kw)
    def result(self):
        return self.tp / (self.ap + 1e-7)


class AliveF1(_BinaryClassMetric):
    def __init__(self, threshold=0.5, **kw):
        super().__init__(threshold, name='alive_f1', **kw)
    def result(self):
        p = self.tp / (self.pp + 1e-7)
        r = self.tp / (self.ap + 1e-7)
        return 2.0 * p * r / (p + r + 1e-7)


def alive_metrics(threshold=0.5):
    return [
        tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=threshold),
        AlivePrecision(threshold),
        AliveRecall(threshold),
        AliveF1(threshold),
    ]


# ════════════════════════════════════════════════════════════════════════════
# 4.  SOFT (DIFFERENTIABLE) GoL FORWARD STEP
#     Used as physics regulariser in PI-CAN's custom train_step.
# ════════════════════════════════════════════════════════════════════════════

def soft_gol_step(p, temperature=10.0):
    """
    Differentiable approximation of Conway's Game of Life.

    Instead of hard argmax, sigmoid-smoothed gates implement:
      survival: alive cell with n ∈ {2, 3} neighbours
      birth:    dead  cell with n == 3   neighbours

    With temperature=10 the approximation is sharp enough to be accurate
    yet the gradient is non-zero, enabling back-prop through GoL rules.

    p : (B, 10, 10, 1) float32 probabilities in [0, 1]
    Returns next-state probabilities, same shape.
    """
    # Toroidal padding: 1 pixel on each side → (B, 12, 12, 1)
    p_top = p[:, -1:, :, :]
    p_bot = p[:, :1,  :, :]
    pp    = tf.concat([p_top, p, p_bot], axis=1)
    p_l   = pp[:, :, -1:, :]
    p_r   = pp[:, :, :1,  :]
    pp    = tf.concat([p_l, pp, p_r], axis=2)   # (B, 12, 12, 1)

    # Sum all 9 cells in 3×3 window, then subtract centre → neighbour count
    k9       = tf.ones([3, 3, 1, 1], dtype=p.dtype)
    sum9     = tf.nn.conv2d(pp, k9, strides=[1, 1, 1, 1], padding='VALID')  # (B,10,10,1)
    neigh    = sum9 - p

    T = temperature
    # Survival: alive cell, n ∈ [1.5, 3.5]  (covers exactly 2 and 3)
    survival = p * (tf.sigmoid(T * (neigh - 1.5)) * tf.sigmoid(T * (3.5 - neigh)))
    # Birth:   dead  cell, n ∈ [2.5, 3.5]  (covers exactly 3)
    birth    = (1.0 - p) * (tf.sigmoid(T * (neigh - 2.5)) * tf.sigmoid(T * (3.5 - neigh)))
    return survival + birth


# ════════════════════════════════════════════════════════════════════════════
# 5.  ARCHITECTURE 1 — GoL Transformer  (CNN tokeniser + Transformer encoder)
#
#   Input  : (B, 10, 10, 1)
#   Tokens : each of the 100 cells is first enriched by a small CNN that sees
#            its 3×3 toroidal neighbourhood, giving it an embed_dim feature vec.
#   Encoder: N Transformer layers with full self-attention over 100 tokens.
#   Output : (B, 10, 10, 1)   (per-cell prediction reshaped back)
# ════════════════════════════════════════════════════════════════════════════

def _toroidal_pos_encoding(board_size=10, embed_dim=64):
    """
    Initialise position embedding for board_size² tokens using circular harmonics.
    Encodes both row and col with sin/cos at multiple frequencies so that
    distance on the torus is reflected in the embedding distance.
    Returns (board_size², embed_dim) float32 array.
    """
    n     = board_size * board_size
    half  = embed_dim // 2
    n_freq = half // 2          # number of frequency bands per spatial dim
    enc   = np.zeros((n, embed_dim), dtype=np.float32)

    for idx in range(n):
        r, c = divmod(idx, board_size)
        for k in range(n_freq):
            freq = (k + 1) * 2.0 * np.pi / board_size
            enc[idx, 4 * k    ] = np.sin(r * freq)
            enc[idx, 4 * k + 1] = np.cos(r * freq)
            enc[idx, 4 * k + 2] = np.sin(c * freq)
            enc[idx, 4 * k + 3] = np.cos(c * freq)
    return enc   # (100, embed_dim)


def _transformer_block(x, num_heads, embed_dim, mlp_ratio=4, dropout=0.1):
    """Pre-norm Transformer block (more stable than post-norm for small models)."""
    # Multi-head self-attention
    x_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    attn   = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim // num_heads,
        dropout=dropout)(x_norm, x_norm)
    x = x + attn

    # Feed-forward network
    x_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    ffn    = tf.keras.layers.Dense(embed_dim * mlp_ratio,
                                    activation='gelu')(x_norm)
    ffn    = tf.keras.layers.Dropout(dropout)(ffn)
    ffn    = tf.keras.layers.Dense(embed_dim)(ffn)
    ffn    = tf.keras.layers.Dropout(dropout)(ffn)
    x = x + ffn
    return x


def build_gol_transformer(board_size=10, embed_dim=64, num_heads=4,
                           num_layers=4, mlp_ratio=4, dropout=0.1, pad=1):
    """
    GoL Transformer  (Architecture 1)

    Design rationale
    ----------------
    • CNN tokeniser (toroidal pad → 2 conv layers) replaces a raw linear
      projection: each token already encodes its local neighbourhood, giving
      the Transformer the inductive bias that GoL rules are local.
    • Full self-attention over 100 tokens then adds global context — cells
      far away may be part of coordinated patterns.
    • Toroidal positional encoding (learnable, initialised from circular
      harmonics) tells the Transformer about the wrap-around topology.
    """
    inp = tf.keras.Input(shape=(board_size, board_size, 1), name='board_T')

    # — CNN tokeniser: local neighbourhood features —
    # Toroidal pad by `pad` pixels before each conv so edges see their neighbours
    x = tf.keras.layers.Lambda(
        lambda t: tf.concat([
            tf.concat([t[:, -pad:, :, :], t, t[:, :pad, :, :]], axis=1)
        ], axis=0),
        name='row_pad'
    )(inp)
    x = tf.keras.layers.Lambda(
        lambda t: tf.concat([t[:, :, -pad:, :], t, t[:, :, :pad, :]], axis=2),
        name='col_pad'
    )(x)
    # After pad: (B, 10+2p, 10+2p, 1)
    x = tf.keras.layers.Conv2D(embed_dim // 2, 3, padding='valid',
                                use_bias=False, name='cnn_tok1')(x)
    x = tf.keras.layers.BatchNormalization(name='bn_tok1')(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Shape is now (B, board_size, board_size, embed_dim//2) for pad=1, kernel=3

    # Second local conv (same padding — stays at board_size × board_size)
    x = tf.keras.layers.Conv2D(embed_dim, 3, padding='same',
                                use_bias=False, name='cnn_tok2')(x)
    x = tf.keras.layers.BatchNormalization(name='bn_tok2')(x)
    x = tf.keras.layers.Activation('relu')(x)
    # x: (B, 10, 10, embed_dim)

    # — Flatten spatial dims → token sequence —
    x = tf.keras.layers.Reshape((board_size * board_size, embed_dim),
                                  name='to_tokens')(x)  # (B, 100, embed_dim)

    # — Learnable toroidal positional embedding (initialised from circular harmonics) —
    pos_init = _toroidal_pos_encoding(board_size, embed_dim)    # (100, embed_dim)
    pos_embed = tf.keras.layers.Embedding(
        input_dim=board_size * board_size, output_dim=embed_dim,
        embeddings_initializer=tf.keras.initializers.Constant(pos_init),
        name='pos_embed'
    )
    positions = tf.range(board_size * board_size)               # (100,)
    x = x + pos_embed(positions)                                 # (B, 100, embed_dim)

    # — Transformer encoder —
    for i in range(num_layers):
        x = _transformer_block(x, num_heads, embed_dim, mlp_ratio, dropout)

    # — Output head: project each token to a single probability —
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='final_norm')(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid', name='cell_logit')(x)
    # x: (B, 100, 1)
    out = tf.keras.layers.Reshape((board_size, board_size, 1),
                                   name='output_T_minus_1')(x)

    return tf.keras.Model(inp, out, name='GoLTransformer')


# ════════════════════════════════════════════════════════════════════════════
# 6.  ARCHITECTURE 2 — Physics-Informed CellAttentionNet  (PI-CAN)
#
#   Backbone: toroidal-padded ResConv blocks with CBAM attention modules.
#   Physics:  during training, a custom train_step adds a soft-GoL loss:
#               L_total = L_combo + λ * BCE(soft_GoL(ŷ_T₋₁), x_T)
#             This directly rewards predictions that are "GoL-consistent".
# ════════════════════════════════════════════════════════════════════════════

def _channel_attention(x, ratio=8):
    """Squeeze-and-Excite channel attention."""
    C     = x.shape[-1]
    r     = max(1, C // ratio)
    d1    = tf.keras.layers.Dense(r,   activation='relu',    use_bias=False)
    d2    = tf.keras.layers.Dense(C,   activation='sigmoid', use_bias=False)
    avg   = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(x)
    mx    = tf.keras.layers.GlobalMaxPooling2D(keepdims=True)(x)
    scale = d2(d1(avg)) + d2(d1(mx))      # shared MLPs
    return tf.keras.layers.Multiply()([x, scale])


def _spatial_attention(x, kernel=7):
    """Spatial attention: where to focus."""
    avg    = tf.reduce_mean(x, axis=-1, keepdims=True)
    mx     = tf.reduce_max(x,  axis=-1, keepdims=True)
    concat = tf.keras.layers.Concatenate(axis=-1)([avg, mx])
    attn   = tf.keras.layers.Conv2D(1, kernel, padding='same',
                                     activation='sigmoid')(concat)
    return tf.keras.layers.Multiply()([x, attn])


def _cbam_block(x, ratio=8, kernel=7):
    """CBAM: channel attention then spatial attention."""
    x = _channel_attention(x, ratio)
    x = _spatial_attention(x, kernel)
    return x


def _res_cbam(x, filters, dropout=0.1):
    """Residual block with CBAM attention."""
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, padding='same',
                                           use_bias=False)(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Conv2D(filters, 3, padding='same',
                                use_bias=False,
                                kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.SpatialDropout2D(dropout)(x)

    x = tf.keras.layers.Conv2D(filters, 3, padding='same',
                                use_bias=False,
                                kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = _cbam_block(x)

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x


def _wrap_pad_tf(x, pad):
    """Toroidal (wrap) padding."""
    top = x[:, -pad:, :, :];  bot = x[:, :pad, :, :]
    x   = tf.concat([top, x, bot], axis=1)
    lft = x[:, :, -pad:, :];  rgt = x[:, :, :pad, :]
    return tf.concat([lft, x, rgt], axis=2)


def build_cell_attention_net(board_size=10, filters=96, n_blocks=8,
                              pad=2, dropout=0.1):
    """
    CellAttentionNet backbone (used inside PI-CAN).

    Design rationale
    ----------------
    • Toroidal wrap padding (pad=2) means every 3×3 conv sees correct
      toroidal neighbours — essential for accurate GoL reconstruction.
    • CBAM blocks (channel + spatial attention) let the network focus on
      the sparse alive-cell regions that matter most.
    • 8 residual blocks with 96 filters provide more capacity than the
      current best model (80f, 7 blocks) without exploding parameter count.
    • 1×1 output conv + crop back to board_size.
    """
    inp = tf.keras.Input(shape=(board_size, board_size, 1), name='board_T')

    x = tf.keras.layers.Lambda(lambda t: _wrap_pad_tf(t, pad),
                                name='torus_pad')(inp)  # (B, 14, 14, 1)

    x = tf.keras.layers.Conv2D(filters, 3, padding='same',
                                use_bias=False, name='entry_conv')(x)
    x = tf.keras.layers.BatchNormalization(name='entry_bn')(x)
    x = tf.keras.layers.Activation('relu')(x)

    for i in range(n_blocks):
        x = _res_cbam(x, filters, dropout)

    x = tf.keras.layers.Cropping2D(cropping=((pad, pad), (pad, pad)),
                                    name='crop_to_board')(x)  # (B, 10, 10, filters)
    out = tf.keras.layers.Conv2D(1, 1, activation='sigmoid',
                                  name='output_T_minus_1')(x)

    return tf.keras.Model(inp, out, name='CellAttentionNet')


class PhysicsInformedModel(tf.keras.Model):
    """
    Wraps any backbone model with a physics-informed training step.

    Extra loss term during training:
        L_physics = BCE( soft_GoL(ŷ_{T-1}), x_T )

    Because soft_GoL is differentiable, gradients flow back through the
    GoL approximation into the backbone weights, teaching the network to
    produce predictions that are consistent with GoL rules.

    physics_weight : float — λ scaling the physics loss (0.1–0.3 recommended)
    """

    def __init__(self, backbone, physics_weight=0.15, **kwargs):
        super().__init__(**kwargs)
        self.backbone       = backbone
        self.physics_weight = physics_weight
        self._phys_loss_tracker = tf.keras.metrics.Mean(name='physics_loss')
        self._prim_loss_tracker = tf.keras.metrics.Mean(name='primary_loss')

    def call(self, inputs, training=None):
        return self.backbone(inputs, training=training)

    @property
    def metrics(self):
        return (super().metrics
                + [self._phys_loss_tracker, self._prim_loss_tracker])

    def train_step(self, data):
        x, y = data   # x = board T, y = board T-1

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)

            # Primary task loss (Combo / Tversky / Focal)
            primary_loss = self.compiled_loss(y, y_pred)

            # Physics regularisation: soft GoL(predicted T-1) ≈ input T
            pred_fwd = soft_gol_step(y_pred)
            physics_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(x, pred_fwd))

            total_loss = primary_loss + self.physics_weight * physics_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.compiled_metrics.update_state(y, y_pred)
        self._phys_loss_tracker.update_state(physics_loss)
        self._prim_loss_tracker.update_state(primary_loss)

        result = {m.name: m.result() for m in self.metrics}
        result['loss'] = total_loss
        return result

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        self.compiled_loss(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


# ════════════════════════════════════════════════════════════════════════════
# 7.  DATA AUGMENTATION — D4 symmetry (8× boards)
# ════════════════════════════════════════════════════════════════════════════

class D4Sequence(tf.keras.utils.Sequence):
    """On-the-fly D4 (dihedral-group) augmentation.
    One random transform from the 8 GoL-symmetric transforms per sample."""

    def __init__(self, X, y, batch_size=512):
        self.X, self.y   = X, y
        self.batch_size  = batch_size
        self._reshuffle()

    def _reshuffle(self):
        idx = np.random.permutation(len(self.X))
        self.X, self.y = self.X[idx], self.y[idx]

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        s = idx * self.batch_size
        e = min(s + self.batch_size, len(self.X))
        Xb = self.X[s:e].copy()
        yb = self.y[s:e].copy()

        k    = np.random.randint(4)
        flip = np.random.rand() > 0.5
        if k:
            Xb = np.rot90(Xb, k, axes=(1, 2))
            yb = np.rot90(yb, k, axes=(1, 2))
        if flip:
            Xb = Xb[:, :, ::-1, :]
            yb = yb[:, :, ::-1, :]
        return Xb, yb

    def on_epoch_end(self):
        self._reshuffle()


# ════════════════════════════════════════════════════════════════════════════
# 8.  TEST-TIME AUGMENTATION  (D4 ensemble — free performance boost)
# ════════════════════════════════════════════════════════════════════════════

def predict_tta(model, X, batch_size=512):
    """
    Average predictions over all 8 D4 transforms.
    Each prediction is inverse-transformed before averaging, so the result
    is always in the original board orientation.
    """
    preds = []
    for k in range(4):
        for flip in (False, True):
            Xt = np.rot90(X, k=k, axes=(1, 2))
            if flip:
                Xt = Xt[:, :, ::-1, :]
            p = model.predict(Xt, batch_size=batch_size, verbose=0)
            if flip:
                p = p[:, :, ::-1, :]
            p = np.rot90(p, k=-k, axes=(1, 2))
            preds.append(p)
    return np.mean(preds, axis=0)


# ════════════════════════════════════════════════════════════════════════════
# 9.  THRESHOLD OPTIMISATION
# ════════════════════════════════════════════════════════════════════════════

def find_best_threshold(y_pred_prob, y_true, metric='f1'):
    """
    Search [0.25, 0.75] for the threshold that maximises the chosen metric.
    metric: 'f1', 'alive_recall', 'accuracy'
    Returns (best_threshold, best_score).
    """
    flat_prob = y_pred_prob.flatten()
    flat_true = y_true.flatten().astype(int)
    best_t, best_s = 0.5, 0.0

    for t in np.arange(0.25, 0.76, 0.01):
        pred = (flat_prob > t).astype(int)
        if metric == 'f1':
            s = f1_score(flat_true, pred, zero_division=0)
        elif metric == 'alive_recall':
            s = recall_score(flat_true, pred, zero_division=0)
        else:
            s = float(np.mean(pred == flat_true))
        if s > best_s:
            best_s, best_t = s, t
    return float(best_t), float(best_s)


# ════════════════════════════════════════════════════════════════════════════
# 10.  FULL EVALUATION
# ════════════════════════════════════════════════════════════════════════════

def evaluate(model, X, y, label='', threshold=0.5, use_tta=False):
    """Print a detailed per-class breakdown and return overall pixel accuracy."""
    if use_tta:
        y_prob = predict_tta(model, X)
    else:
        y_prob = model.predict(X, batch_size=512, verbose=0)

    y_bin  = (y_prob > threshold).astype(int)
    y_int  = (y > 0.5).astype(int)
    flat_p = y_bin.flatten()
    flat_t = y_int.flatten()

    overall = float(np.mean(flat_p == flat_t))
    alive_m = flat_t == 1
    dead_m  = flat_t == 0
    alive_r = float(np.mean(flat_p[alive_m] == 1)) if alive_m.any() else 0.0
    dead_r  = float(np.mean(flat_p[dead_m]  == 0)) if dead_m.any() else 0.0

    prec  = precision_score(flat_t, flat_p, zero_division=0)
    rec   = recall_score(flat_t, flat_p, zero_division=0)
    f1    = f1_score(flat_t, flat_p, zero_division=0)

    per_board = np.mean(y_bin.reshape(-1, SIZE * SIZE) ==
                        y_int.reshape(-1, SIZE * SIZE), axis=1)
    boards90  = float(np.mean(per_board >= 0.9))

    tag = f"[{label}]" if label else ''
    tta_tag = ' (TTA)' if use_tta else ''
    print(f"  {tag}{tta_tag}  threshold={threshold:.2f}"
          f"  pixel_acc={overall:.4f}"
          f"  dead_recall={dead_r:.4f}  alive_recall={alive_r:.4f}"
          f"  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}"
          f"  boards≥90%={boards90:.2%}", flush=True)
    return overall, alive_r, f1


# ════════════════════════════════════════════════════════════════════════════
# 11.  TRAINING HELPERS
# ════════════════════════════════════════════════════════════════════════════

def standard_callbacks(metric='val_alive_f1', patience_es=12, patience_lr=4):
    """EarlyStopping on alive_f1 + ReduceLROnPlateau on val_loss."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=metric, patience=patience_es,
            restore_best_weights=True, mode='max', verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=patience_lr,
            min_lr=5e-7, verbose=1),
    ]


def run_phase(name, model, X_tr, y_tr, X_val, y_val,
              loss_fn, lr=1e-3, epochs=30, batch_size=512,
              use_aug=True, physics_weight=0.0):
    """
    Build / wrap → compile → train → return (model, best_val_f1).
    If physics_weight > 0 the model is wrapped in PhysicsInformedModel.
    """
    tf.keras.backend.clear_session()

    if physics_weight > 0:
        wrapped = PhysicsInformedModel(model, physics_weight=physics_weight,
                                        name=f'{name}_physics')
    else:
        wrapped = model

    wrapped.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr,
                                             weight_decay=1e-4),
        loss=loss_fn,
        metrics=alive_metrics(),
    )

    cbs = standard_callbacks()

    if use_aug:
        gen = D4Sequence(X_tr, y_tr, batch_size=batch_size)
        hist = wrapped.fit(gen, validation_data=(X_val, y_val),
                           epochs=epochs, callbacks=cbs, verbose=2)
    else:
        hist = wrapped.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                           epochs=epochs, batch_size=batch_size,
                           callbacks=cbs, verbose=2)

    best_f1 = max(hist.history.get('val_alive_f1', [0]))
    return wrapped, best_f1, hist


# ════════════════════════════════════════════════════════════════════════════
# 12.  MAIN — three-phase training pipeline
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # ── Load data ─────────────────────────────────────────────────────────
    data, alive_ratio = load_data()
    X_train = data['X_train']; y_train = data['y_train']
    X_val   = data['X_val'];   y_val   = data['y_val']
    X_test  = data['X_test'];  y_test  = data['y_test']

    # Combo loss (Tversky + Focal) is used for both architectures.
    # Tversky beta=0.7 > alpha=0.3: heavily penalises FN (missed alive cells).
    LOSS = combo_loss(tv_alpha=0.3, tv_beta=0.7,
                      f_gamma=2.5, f_alpha=0.65, ratio=0.5)

    results = {}

    # ════════════════════════════════════════════════════════════════════
    # PHASE 1 — Quick screen on 60K samples, 15 epochs (pick survivors)
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 70, flush=True)
    print("PHASE 1 — Architecture screening  (60K × 15 ep)", flush=True)
    print("═" * 70, flush=True)

    SCREEN_N = min(60_000, len(X_train))
    idx_s    = np.random.choice(len(X_train), SCREEN_N, replace=False)
    Xs, ys   = X_train[idx_s], y_train[idx_s]

    # — 1A: GoL Transformer —
    print("\n── 1A: GoL Transformer ──", flush=True)
    golt_backbone = build_gol_transformer(
        embed_dim=64, num_heads=4, num_layers=4, mlp_ratio=4, dropout=0.10, pad=1)
    print(f"  params: {golt_backbone.count_params():,}", flush=True)
    golt, f1_golt, _ = run_phase(
        'GoLT', golt_backbone, Xs, ys, X_val, y_val,
        loss_fn=LOSS, lr=3e-4, epochs=15, batch_size=256,
        use_aug=True, physics_weight=0.0)
    evaluate(golt, X_val, y_val, '1A GoLT Val')
    results['1A GoLT'] = f1_golt

    # — 1B: PI-CAN (physics-informed CellAttentionNet) —
    print("\n── 1B: PI-CAN ──", flush=True)
    can_backbone = build_cell_attention_net(
        filters=96, n_blocks=8, pad=2, dropout=0.10)
    print(f"  params: {can_backbone.count_params():,}", flush=True)
    pican, f1_pican, _ = run_phase(
        'PICAN', can_backbone, Xs, ys, X_val, y_val,
        loss_fn=LOSS, lr=1e-3, epochs=15, batch_size=256,
        use_aug=True, physics_weight=0.15)
    evaluate(pican, X_val, y_val, '1B PI-CAN Val')
    results['1B PI-CAN'] = f1_pican

    print("\n── Phase 1 results ──", flush=True)
    for name, score in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:<20} alive_f1={score:.4f}", flush=True)

    # ════════════════════════════════════════════════════════════════════
    # PHASE 2 — Scale up best model on full training set, 40 epochs
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 70, flush=True)
    print("PHASE 2 — Full training  (all samples × 40 ep)", flush=True)
    print("═" * 70, flush=True)

    # Train BOTH models fully — they'll be ensembled later
    # GoLT (full)
    print("\n── 2A: GoLT full train ──", flush=True)
    tf.keras.backend.clear_session()
    golt_full_backbone = build_gol_transformer(
        embed_dim=64, num_heads=4, num_layers=4, mlp_ratio=4, dropout=0.10, pad=1)
    golt_full, f1_golt_full, hist_golt = run_phase(
        'GoLT_full', golt_full_backbone, X_train, y_train, X_val, y_val,
        loss_fn=LOSS, lr=3e-4, epochs=40, batch_size=256,
        use_aug=True, physics_weight=0.0)
    evaluate(golt_full, X_val,  y_val,  '2A GoLT Val')
    evaluate(golt_full, X_test, y_test, '2A GoLT Test')

    # PI-CAN (full)
    print("\n── 2B: PI-CAN full train ──", flush=True)
    tf.keras.backend.clear_session()
    can_full_backbone = build_cell_attention_net(
        filters=96, n_blocks=8, pad=2, dropout=0.10)
    pican_full, f1_pican_full, hist_pican = run_phase(
        'PICAN_full', can_full_backbone, X_train, y_train, X_val, y_val,
        loss_fn=LOSS, lr=1e-3, epochs=40, batch_size=256,
        use_aug=True, physics_weight=0.15)
    evaluate(pican_full, X_val,  y_val,  '2B PI-CAN Val')
    evaluate(pican_full, X_test, y_test, '2B PI-CAN Test')

    # ════════════════════════════════════════════════════════════════════
    # PHASE 3 — Post-processing: TTA + threshold optimisation + ensemble
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 70, flush=True)
    print("PHASE 3 — TTA + Threshold + Ensemble", flush=True)
    print("═" * 70, flush=True)

    # — Threshold search (val set, optimise F1) —
    print("\nThreshold search on validation set …", flush=True)
    for tag, mdl in [('GoLT', golt_full), ('PI-CAN', pican_full)]:
        prob = mdl.predict(X_val, batch_size=512, verbose=0)
        t_f1,  s_f1  = find_best_threshold(prob, y_val, 'f1')
        t_rec, s_rec = find_best_threshold(prob, y_val, 'alive_recall')
        print(f"  {tag:8}  best_f1_threshold={t_f1:.2f} ({s_f1:.4f})"
              f"  best_recall_threshold={t_rec:.2f} ({s_rec:.4f})", flush=True)

    # — TTA evaluation (GoLT) —
    print("\nTTA evaluation — GoLT:", flush=True)
    evaluate(golt_full, X_test, y_test, 'GoLT TTA Test', use_tta=True)

    # — TTA evaluation (PI-CAN) —
    print("\nTTA evaluation — PI-CAN:", flush=True)
    evaluate(pican_full, X_test, y_test, 'PI-CAN TTA Test', use_tta=True)

    # — Ensemble: average probabilities from both models —
    print("\nEnsemble (GoLT + PI-CAN mean probability):", flush=True)
    prob_golt  = predict_tta(golt_full,  X_test)
    prob_pican = predict_tta(pican_full, X_test)
    prob_ens   = 0.5 * prob_golt + 0.5 * prob_pican

    for tag, prob in [('Ensemble', prob_ens)]:
        t_f1, _ = find_best_threshold(
            0.5 * predict_tta(golt_full, X_val) +
            0.5 * predict_tta(pican_full, X_val),
            y_val, 'f1')
        y_bin  = (prob > t_f1).astype(int)
        y_int  = (y_test > 0.5).astype(int)
        flat_p = y_bin.flatten();  flat_t = y_int.flatten()
        overall = float(np.mean(flat_p == flat_t))
        alive_m = flat_t == 1
        ar      = float(np.mean(flat_p[alive_m] == 1)) if alive_m.any() else 0.0
        prec    = precision_score(flat_t, flat_p, zero_division=0)
        f1s     = f1_score(flat_t, flat_p, zero_division=0)
        print(f"  {tag}  t={t_f1:.2f}  pixel_acc={overall:.4f}"
              f"  alive_recall={ar:.4f}  prec={prec:.4f}  f1={f1s:.4f}", flush=True)

    # ════════════════════════════════════════════════════════════════════
    # SAVE models
    # ════════════════════════════════════════════════════════════════════
    print("\nSaving models …", flush=True)

    # Save only the backbone weights for portability
    # (PhysicsInformedModel wraps the backbone; golt_full may be the backbone directly)
    backbone_golt  = (golt_full.backbone
                      if isinstance(golt_full, PhysicsInformedModel)
                      else golt_full)
    backbone_pican = (pican_full.backbone
                      if isinstance(pican_full, PhysicsInformedModel)
                      else pican_full)

    best_val_f1_golt  = f1_golt_full
    best_val_f1_pican = f1_pican_full

    golt_path  = f"models/GoLT_f1{best_val_f1_golt:.4f}.keras"
    pican_path = f"models/PICAN_f1{best_val_f1_pican:.4f}.keras"

    backbone_golt.save(golt_path)
    backbone_pican.save(pican_path)
    print(f"  Saved: {golt_path}", flush=True)
    print(f"  Saved: {pican_path}", flush=True)

    print("\n" + "═" * 70, flush=True)
    print("PHASE 6 COMPLETE", flush=True)
    print(f"  Baseline (Phase 5):   pixel_acc ≈ 0.9059  alive_recall ≈ 0.7282", flush=True)
    print(f"  Target:               pixel_acc > 0.9500  alive_recall > 0.8500", flush=True)
    print("═" * 70, flush=True)
