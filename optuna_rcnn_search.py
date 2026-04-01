"""
Optuna hyperparameter search for ConvLSTM / RCNN model
to predict Conway's Game of Life backwards.

Target: validation accuracy >= 0.90

Usage:
    python optuna_rcnn_search.py

Uses the existing data pipeline from functions.py:
    load_reverse_df  →  prepare_reverse_dataset  →  to_numpy_4d
"""

import os
import time
import optuna
import numpy as np
import tensorflow as tf
from functions import (
    load_reverse_df,
    prepare_reverse_dataset,
    to_numpy_4d,
)

# ─── Configuration ────────────────────────────────────────────────────────────
SIZE = 10
AMOUNT_BOARDS = 100_000
GEN = 4
TARGET_PIXEL_INDEX = 0

N_TRIALS = 50               # total Optuna trials
MAX_EPOCHS = 40              # max epochs per trial
BATCH_SIZE_OPTIONS = [64, 128, 256]
TARGET_VAL_ACC = 0.90        # stop the whole study when reached
SAVE_DIR = "models"

os.makedirs(SAVE_DIR, exist_ok=True)


# ─── Data loading (done once) ─────────────────────────────────────────────────
def load_data():
    print(f"Loading data: SIZE={SIZE}, AMOUNT_BOARDS={AMOUNT_BOARDS}, GEN={GEN}")
    reverse_df = load_reverse_df(SIZE, AMOUNT_BOARDS, GEN)

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_reverse_dataset(
        reverse_df, SIZE, GEN,
        target_pixel_index=TARGET_PIXEL_INDEX,
        test_size=0.1, val_size=0.1, random_state=365,
    )

    X_train_a, X_val_a, X_test_a, y_train_a, y_val_a, y_test_a = to_numpy_4d(
        X_train, X_val, X_test, y_train, y_val, y_test, SIZE, GEN,
    )

    return X_train_a, X_val_a, X_test_a, y_train_a, y_val_a, y_test_a


# ─── Reshape helpers ──────────────────────────────────────────────────────────
def to_convlstm_input(X, size, gen):
    """Reshape (n, size, size, gen-1) → (n, gen-1, size, size, 1)."""
    timesteps = gen - 1
    return X.reshape((-1, timesteps, size, size, 1)).astype("float32")


# ─── Model builder ────────────────────────────────────────────────────────────
def build_model(trial):
    """Dynamically build a ConvLSTM model with Optuna-suggested hyperparameters."""

    timesteps = GEN - 1

    # --- Search space ---
    n_conv_layers   = trial.suggest_int("n_conv_layers", 1, 2)
    conv_filters    = trial.suggest_categorical("conv_filters", [16, 32, 64])
    lstm_units      = trial.suggest_categorical("lstm_units", [32, 64, 128])
    dropout_rate    = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.05)
    learning_rate   = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    dense_units     = trial.suggest_categorical("dense_units", [32, 64, 128])
    use_bidirectional = trial.suggest_categorical("use_bidirectional", [True, False])
    batch_size      = trial.suggest_categorical("batch_size", BATCH_SIZE_OPTIONS)

    # Store batch_size in trial user_attrs so the training loop can use it
    trial.set_user_attr("batch_size", batch_size)

    # --- Build model ---
    inputs = tf.keras.layers.Input(shape=(timesteps, SIZE, SIZE, 1))
    x = inputs

    # ConvLSTM layers
    for i in range(n_conv_layers):
        is_last = (i == n_conv_layers - 1) and not use_bidirectional
        layer = tf.keras.layers.ConvLSTM2D(
            filters=conv_filters,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            return_sequences=not is_last,
        )
        x = layer(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if not is_last:
            x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Optional second recurrent pass with a standard LSTM (Bidirectional)
    if use_bidirectional:
        x = tf.keras.layers.Reshape((timesteps if n_conv_layers == 0 else -1,
                                      np.prod(x.shape[2:])))(x)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_units, activation="tanh")
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    else:
        x = tf.keras.layers.Flatten()(x)

    # Dense head
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ─── Custom Keras callback: prune bad trials early via Optuna ─────────────────
class OptunaPruningCallback(tf.keras.callbacks.Callback):
    """Report intermediate val_accuracy to Optuna; prune unpromising trials."""

    def __init__(self, trial):
        super().__init__()
        self.trial = trial

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get("val_accuracy")
        if val_acc is None:
            return
        self.trial.report(val_acc, epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


# ─── Custom exception to stop entire study ────────────────────────────────────
class StudyStopException(Exception):
    """Raised when target accuracy is reached to stop the Optuna study."""


# ─── Objective function ───────────────────────────────────────────────────────
def objective(trial, X_train, X_val, y_train, y_val):
    """Train one trial and return validation accuracy."""

    tf.keras.backend.clear_session()

    model = build_model(trial)
    batch_size = trial.user_attrs["batch_size"]

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=0,
        ),
        OptunaPruningCallback(trial),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    val_acc = max(history.history["val_accuracy"])

    # If target reached — save model and stop the study
    if val_acc >= TARGET_VAL_ACC:
        save_path = os.path.join(
            SAVE_DIR,
            f"best_rcnn_optuna_{val_acc:.4f}_trial{trial.number}.keras",
        )
        model.save(save_path)
        print(f"\n*** Target reached! val_accuracy={val_acc:.4f}  →  saved to {save_path}")
        raise StudyStopException(f"Target val_accuracy {TARGET_VAL_ACC} reached at trial {trial.number}")

    return val_acc


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Load and reshape data once
    X_train_a, X_val_a, X_test_a, y_train_a, y_val_a, y_test_a = load_data()

    X_train_cl = to_convlstm_input(X_train_a, SIZE, GEN)
    X_val_cl = to_convlstm_input(X_val_a, SIZE, GEN)
    X_test_cl = to_convlstm_input(X_test_a, SIZE, GEN)

    print(f"Train shape: {X_train_cl.shape}  Val shape: {X_val_cl.shape}  Test shape: {X_test_cl.shape}")

    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        study_name="rcnn_reverse_gol",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )

    start = time.time()

    try:
        study.optimize(
            lambda trial: objective(trial, X_train_cl, X_val_cl, y_train_a, y_val_a),
            n_trials=N_TRIALS,
            catch=(StudyStopException,),
        )
    except StudyStopException:
        pass  # gracefully stopped after target was met

    elapsed = time.time() - start

    # ─── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("OPTUNA SEARCH COMPLETE")
    print(f"Time elapsed:  {elapsed / 60:.1f} min")
    print(f"Trials run:    {len(study.trials)}")
    print(f"Best trial:    #{study.best_trial.number}")
    print(f"Best val_acc:  {study.best_value:.4f}")
    print("Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # ─── Final eval on test set with best model ───────────────────────────
    best = study.best_trial
    print("\nRetraining best model for final test evaluation...")
    tf.keras.backend.clear_session()

    # Rebuild + retrain the best model
    best_model = build_model(best)
    batch_size = best.user_attrs["batch_size"]

    best_model.fit(
        X_train_cl, y_train_a,
        validation_data=(X_val_cl, y_val_a),
        epochs=MAX_EPOCHS,
        batch_size=batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=6,
                restore_best_weights=True, verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=3, min_lr=1e-6, verbose=1,
            ),
        ],
        verbose=1,
    )

    # Save the final model
    final_path = os.path.join(SAVE_DIR, "best_rcnn_optuna_final.keras")
    best_model.save(final_path)
    print(f"Final model saved to {final_path}")

    # Evaluate on held-out test set
    test_loss, test_acc = best_model.evaluate(X_test_cl, y_test_a, verbose=1)
    print(f"\n*** Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
