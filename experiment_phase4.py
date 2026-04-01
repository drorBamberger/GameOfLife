"""
Phase 4: Wider model (128 filters) to break through the 0.90 barrier.
Analysis from Phase 3: 64-filter model plateaued at 0.8974 with near-zero
train-val gap → model is underfitting (capacity-limited).

Strategy:
- 128 filters (4x wider) for richer feature representation
- Weighted loss (pos_weight=3) to improve alive-pixel accuracy
- 1M training samples (more data barely helped vs capacity)
- Cosine LR schedule for smoother convergence
"""

import os, sys, time, gc, numpy as np, pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

SIZE = 10
AMOUNT_BOARDS = 100_000
GEN = 2
PATH_DF = 'C:\\GameOfLifeFiles\\df\\'
os.makedirs("models", exist_ok=True)
np.random.seed(42)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading pickle...", flush=True)
name_df = f'{PATH_DF}\\{SIZE}-{AMOUNT_BOARDS}\\{SIZE}size_{AMOUNT_BOARDS}boards_{GEN}gen_reverse'
df = pd.read_pickle(f'{name_df}.pkl')
print(f"Loaded {len(df):,} rows", flush=True)

nf = (GEN - 1) * SIZE * SIZE  # 100

print("Splitting...", flush=True)
idx_tv, idx_test = train_test_split(np.arange(len(df)), test_size=0.1, random_state=365)
idx_train, idx_val = train_test_split(idx_tv, test_size=0.1, random_state=365)
del idx_tv

def extract(indices):
    sub = df.iloc[indices]
    X = sub.iloc[:, :nf].to_numpy(dtype=np.float32).reshape(-1, SIZE, SIZE, 1)
    y = sub.iloc[:, nf:].to_numpy(dtype=np.float32)
    return X, y

print("Converting...", flush=True)
X_train_full, y_train_full = extract(idx_train)
X_val, y_val = extract(idx_val)
X_test, y_test = extract(idx_test)
del df, idx_train, idx_val, idx_test
gc.collect()

N_TRAIN_FULL = len(X_train_full)
print(f"Full train: {X_train_full.shape}, Val: {X_val.shape}, Test: {X_test.shape}", flush=True)
print(f"Alive fraction: {y_train_full.mean():.3f}", flush=True)

# Use 1M subset for faster training
SUBSET = min(1_000_000, N_TRAIN_FULL)
idx_sub = np.random.choice(N_TRAIN_FULL, SUBSET, replace=False)
X_train = X_train_full[idx_sub]
y_train = y_train_full[idx_sub]
del X_train_full, y_train_full
gc.collect()
print(f"Training subset: {X_train.shape}", flush=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
METRIC_KEY = 'val_binary_accuracy'

def weighted_bce(pos_weight=3.0):
    def loss_fn(y_true, y_pred):
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        weight = y_true * (pos_weight - 1.0) + 1.0
        return tf.reduce_mean(weight * bce, axis=-1)
    return loss_fn


def evaluate_fb(model, X_te, y_te, label=""):
    y_pred = model.predict(X_te, verbose=0)
    y_pred_bin = (y_pred > 0.5).astype(int)
    y_true_int = y_te.astype(int)
    overall = np.mean(y_pred_bin == y_true_int)
    alive = y_true_int == 1
    dead = y_true_int == 0
    alive_acc = np.mean(y_pred_bin[alive] == 1) if alive.sum() > 0 else 0
    dead_acc = np.mean(y_pred_bin[dead] == 0) if dead.sum() > 0 else 0
    per_sample = np.mean(y_pred_bin == y_true_int, axis=1)
    pct_90 = np.mean(per_sample >= 0.90)

    print(f"  {label}:", flush=True)
    print(f"    Pixel acc: {overall:.4f} | Dead: {dead_acc:.4f} | Alive: {alive_acc:.4f} | Boards>=90%: {pct_90:.2%}", flush=True)
    return overall


# ── Architecture: Plain Conv 10 layers × 128 filters ────────────────────────
def build_wide_conv(n_filters=128, n_layers=10):
    inp = tf.keras.layers.Input(shape=(SIZE, SIZE, 1))
    x = inp
    for _ in range(n_layers):
        x = tf.keras.layers.Conv2D(n_filters, 3, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
    out = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
    out = tf.keras.layers.Flatten()(out)
    m = tf.keras.Model(inp, out)
    return m


# ══════════════════════════════════════════════════════════════════════════════
# Experiment A: 128 filters, standard BCE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'#'*60}", flush=True)
print(f"# Experiment A: 128 filters, standard BCE, 1M samples", flush=True)
print(f"{'#'*60}\n", flush=True)

tf.keras.backend.clear_session()
model_a = build_wide_conv(128, 10)
model_a.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.BinaryAccuracy()],
)
print(f"Params: {model_a.count_params():,}", flush=True)

cbs_a = [
    tf.keras.callbacks.EarlyStopping(METRIC_KEY, patience=8,
                                      restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.5, patience=3,
                                          min_lr=1e-6, verbose=1),
]

t0 = time.time()
h_a = model_a.fit(X_train, y_train, validation_data=(X_val, y_val),
                   epochs=40, batch_size=512, callbacks=cbs_a, verbose=2)
elapsed_a = time.time() - t0
best_a = max(h_a.history[METRIC_KEY])
best_ep_a = int(np.argmax(h_a.history[METRIC_KEY])) + 1
print(f">> Exp A: best val_acc={best_a:.4f} @ epoch {best_ep_a}, time={elapsed_a:.0f}s", flush=True)
evaluate_fb(model_a, X_val, y_val, "Exp A Val")
evaluate_fb(model_a, X_test, y_test, "Exp A Test")

if best_a >= 0.90:
    model_a.save("models/best_fullboard_wide.keras")
    print(f"\n*** TARGET 0.90 REACHED with Exp A! ***", flush=True)
else:
    print(f"\n--- Exp A: {best_a:.4f} ---", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Experiment B: 128 filters + weighted loss (pos_weight=3)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*60}", flush=True)
    print(f"# Experiment B: 128 filters, weighted BCE (w=3), 1M samples", flush=True)
    print(f"{'#'*60}\n", flush=True)

    tf.keras.backend.clear_session()
    model_b = build_wide_conv(128, 10)
    model_b.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=weighted_bce(3.0),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )
    print(f"Params: {model_b.count_params():,}", flush=True)

    cbs_b = [
        tf.keras.callbacks.EarlyStopping(METRIC_KEY, patience=8,
                                          restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.5, patience=3,
                                              min_lr=1e-6, verbose=1),
    ]

    t0 = time.time()
    h_b = model_b.fit(X_train, y_train, validation_data=(X_val, y_val),
                       epochs=40, batch_size=512, callbacks=cbs_b, verbose=2)
    elapsed_b = time.time() - t0
    best_b = max(h_b.history[METRIC_KEY])
    best_ep_b = int(np.argmax(h_b.history[METRIC_KEY])) + 1
    print(f">> Exp B: best val_acc={best_b:.4f} @ epoch {best_ep_b}, time={elapsed_b:.0f}s", flush=True)
    evaluate_fb(model_b, X_val, y_val, "Exp B Val")
    evaluate_fb(model_b, X_test, y_test, "Exp B Test")

    best_overall = max(best_a, best_b)
    if best_b > best_a:
        model_b.save("models/best_fullboard_wide.keras")
        print("Saved Exp B model.", flush=True)
    else:
        model_a.save("models/best_fullboard_wide.keras")
        print("Saved Exp A model.", flush=True)

    if best_overall >= 0.90:
        print(f"\n*** TARGET 0.90 REACHED! best={best_overall:.4f} ***", flush=True)
    else:
        print(f"\n--- Best: {best_overall:.4f}. Need bigger model or different approach. ---", flush=True)

print("\nDone.", flush=True)
