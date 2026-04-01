"""
Fast full-board experiment: predict ALL 100 pixels of previous GoL board.
Fully-convolutional architectures (no dense layers → fewer params, spatial output).
Two-phase: quick architecture test on subset → full train on best.

Key insight: GoL rules are local (3×3 neighbourhood). A stack of 3×3 conv
layers naturally captures this.  Predicting all 100 pixels at once provides
100× richer gradient signal than single-pixel prediction.
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

# ── Load data (memory-efficient: split in pandas, convert per-split) ─────────
print("Loading pickle...", flush=True)
name_df = f'{PATH_DF}\\{SIZE}-{AMOUNT_BOARDS}\\{SIZE}size_{AMOUNT_BOARDS}boards_{GEN}gen_reverse'
df = pd.read_pickle(f'{name_df}.pkl')
print(f"Loaded {len(df):,} rows, {len(df.columns)} cols", flush=True)

# Rename columns
nf = (GEN - 1) * SIZE * SIZE   # 100 feature cols

# Split indices via pandas (low memory)
print("Splitting...", flush=True)
idx_tv, idx_test = train_test_split(np.arange(len(df)), test_size=0.1, random_state=365)
idx_train, idx_val = train_test_split(idx_tv, test_size=0.1, random_state=365)
del idx_tv

def extract(indices):
    """Extract X (4D float32) and y (2D float32) for given row indices."""
    sub = df.iloc[indices]
    X = sub.iloc[:, :nf].to_numpy(dtype=np.float32).reshape(-1, SIZE, SIZE, 1)
    y = sub.iloc[:, nf:].to_numpy(dtype=np.float32)
    return X, y

print("Converting to numpy...", flush=True)
X_train, y_train = extract(idx_train)
X_val, y_val = extract(idx_val)
X_test, y_test = extract(idx_test)
del df, idx_train, idx_val, idx_test
gc.collect()

N_TRAIN = len(X_train)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}", flush=True)
print(f"Alive fraction: {y_train.mean():.3f}", flush=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
METRIC_KEY = 'val_binary_accuracy'   # matches BinaryAccuracy metric name

def run(name, model, X_tr, y_tr, X_v, y_v, epochs, batch_size):
    print(f"\n{'='*60}\n{name} — {model.count_params():,} params\n{'='*60}", flush=True)
    cbs = [
        tf.keras.callbacks.EarlyStopping(METRIC_KEY, patience=5,
                                          restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.5, patience=2,
                                              min_lr=1e-6, verbose=1),
    ]
    t0 = time.time()
    h = model.fit(X_tr, y_tr, validation_data=(X_v, y_v),
                  epochs=epochs, batch_size=batch_size, callbacks=cbs, verbose=2)
    elapsed = time.time() - t0
    best_val = max(h.history[METRIC_KEY])
    best_ep = int(np.argmax(h.history[METRIC_KEY])) + 1
    print(f">> Best val_acc={best_val:.4f} @ epoch {best_ep}, time={elapsed:.0f}s", flush=True)
    return best_val, h, model


def evaluate_fullboard(model, X_te, y_te, label=""):
    """Detailed per-class evaluation."""
    y_pred = model.predict(X_te, verbose=0)
    y_pred_bin = (y_pred > 0.5).astype(int)
    y_true_int = y_te.astype(int)

    overall_acc = np.mean(y_pred_bin == y_true_int)
    alive = y_true_int == 1
    dead = y_true_int == 0
    alive_acc = np.mean(y_pred_bin[alive] == 1) if alive.sum() > 0 else 0
    dead_acc = np.mean(y_pred_bin[dead] == 0) if dead.sum() > 0 else 0

    per_sample = np.mean(y_pred_bin == y_true_int, axis=1)
    pct_90 = np.mean(per_sample >= 0.90)

    print(f"\n  {label} Evaluation:", flush=True)
    print(f"    Overall pixel acc : {overall_acc:.4f}", flush=True)
    print(f"    Dead pixel acc    : {dead_acc:.4f}", flush=True)
    print(f"    Alive pixel acc   : {alive_acc:.4f}", flush=True)
    print(f"    Boards with >=90% : {pct_90:.2%}", flush=True)
    return overall_acc


# ── Building blocks ──────────────────────────────────────────────────────────

def res_block(x, filters):
    shortcut = x
    y = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation('relu')(y)
    y = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False)(y)
    y = tf.keras.layers.BatchNormalization()(y)
    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, use_bias=False)(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    return tf.keras.layers.Activation('relu')(tf.keras.layers.Add()([shortcut, y]))


# ── Architecture A: Lean FCN (4 res blocks, 64 filters) ~150K params ────────
def build_lean_fcn():
    inp = tf.keras.layers.Input(shape=(SIZE, SIZE, 1))
    x = tf.keras.layers.Conv2D(64, 3, padding='same', use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    for _ in range(4):
        x = res_block(x, 64)
    out = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
    out = tf.keras.layers.Flatten()(out)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='binary_crossentropy',
              metrics=[tf.keras.metrics.BinaryAccuracy()])
    return m


# ── Architecture B: Plain deep conv (10 conv layers, 64 filters) ────────────
def build_plain_conv():
    inp = tf.keras.layers.Input(shape=(SIZE, SIZE, 1))
    x = inp
    for i in range(10):
        x = tf.keras.layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
    out = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
    out = tf.keras.layers.Flatten()(out)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='binary_crossentropy',
              metrics=[tf.keras.metrics.BinaryAccuracy()])
    return m


# ── Architecture C: U-Net mini (no SE, minimal) ~120K params ────────────────
def build_unet_tiny():
    inp = tf.keras.layers.Input(shape=(SIZE, SIZE, 1))
    e1 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inp)
    e1 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(e1)
    p1 = tf.keras.layers.MaxPooling2D(2)(e1)  # 5×5

    e2 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(p1)
    e2 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(e2)

    b = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(e2)

    d = tf.keras.layers.UpSampling2D(2)(b)
    d = tf.keras.layers.Concatenate()([d, e1])
    d = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(d)
    d = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(d)

    out = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(d)
    out = tf.keras.layers.Flatten()(out)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='binary_crossentropy',
              metrics=[tf.keras.metrics.BinaryAccuracy()])
    return m


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Quick screening on 100K subset, 10 epochs (should run in minutes)
# ══════════════════════════════════════════════════════════════════════════════
QUICK_N = min(100_000, N_TRAIN)
idx = np.random.choice(N_TRAIN, QUICK_N, replace=False)
X_quick, y_quick = X_train[idx], y_train[idx]
print(f"\n{'#'*60}", flush=True)
print(f"# PHASE 1: Quick screening — {QUICK_N:,} samples, 10 epochs", flush=True)
print(f"{'#'*60}\n", flush=True)

architectures = [
    ("Lean FCN (4res,64f)", build_lean_fcn),
    ("Plain Conv (10 layers)", build_plain_conv),
    ("U-Net tiny", build_unet_tiny),
]

results = {}
for name, builder in architectures:
    tf.keras.backend.clear_session()
    m = builder()
    val_acc, _, _ = run(name, m, X_quick, y_quick, X_val, y_val,
                        epochs=10, batch_size=256)
    results[name] = val_acc
    sys.stdout.flush()

print(f"\n{'='*60}", flush=True)
print("PHASE 1 RESULTS:", flush=True)
print(f"{'='*60}", flush=True)
for name, acc in sorted(results.items(), key=lambda x: -x[1]):
    marker = " <-- BEST" if acc == max(results.values()) else ""
    print(f"  {name:<30} val_acc={acc:.4f}{marker}", flush=True)

best_name = max(results, key=results.get)
best_quick = results[best_name]
print(f"\nWinner: {best_name} ({best_quick:.4f})", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Scale up best arch — 500K samples, 20 epochs
# ══════════════════════════════════════════════════════════════════════════════
PHASE2_N = min(500_000, N_TRAIN)
idx2 = np.random.choice(N_TRAIN, PHASE2_N, replace=False)
X_p2, y_p2 = X_train[idx2], y_train[idx2]

print(f"\n{'#'*60}", flush=True)
print(f"# PHASE 2: Scale up — {PHASE2_N:,} samples, 20 epochs", flush=True)
print(f"{'#'*60}\n", flush=True)

builders = dict(architectures)
tf.keras.backend.clear_session()
model = builders[best_name]()
val_p2, h_p2, model_p2 = run(
    f"PHASE2: {best_name}", model, X_p2, y_p2, X_val, y_val,
    epochs=20, batch_size=256,
)

evaluate_fullboard(model_p2, X_val, y_val, "Phase2 Val")
model_p2.save("models/best_fullboard.keras")

if val_p2 >= 0.90:
    evaluate_fullboard(model_p2, X_test, y_test, "Phase2 Test")
    print(f"\n*** TARGET 0.90 REACHED in Phase 2! val_accuracy = {val_p2:.4f} ***", flush=True)
else:
    print(f"\n--- Phase 2 best: {val_p2:.4f} — trying full data ---", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3: Full data, 30 epochs
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*60}", flush=True)
    print(f"# PHASE 3: Full data — {N_TRAIN:,} samples, 30 epochs", flush=True)
    print(f"{'#'*60}\n", flush=True)

    tf.keras.backend.clear_session()
    model_full = builders[best_name]()
    val_full, h_full, model_full = run(
        f"FULL: {best_name}", model_full, X_train, y_train, X_val, y_val,
        epochs=30, batch_size=512,
    )
    evaluate_fullboard(model_full, X_test, y_test, "Phase3 Test")

    if val_full > val_p2:
        model_full.save("models/best_fullboard.keras")
        print(f"Improved model saved. val_acc={val_full:.4f}", flush=True)

    if val_full >= 0.90:
        print(f"\n*** TARGET 0.90 REACHED in Phase 3! ***", flush=True)
    else:
        print(f"\n--- Final best: {max(val_p2, val_full):.4f} ---", flush=True)

print("\nDone.", flush=True)
