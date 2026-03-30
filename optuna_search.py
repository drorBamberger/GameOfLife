"""
Optuna hyperparameter search for a CRNN/ConvLSTM model that predicts
Conway's Game of Life backwards (backcasting).

Can be run standalone (loads data from disk) or imported and called from
a notebook with pre-loaded arrays via `run_optuna_search(...)`.
"""

import os
import numpy as np
import tensorflow as tf
import optuna
from optuna.exceptions import OptunaError

# ─── Constants ───────────────────────────────────────────────────────────
TARGET_VAL_ACC = 0.90
SAVE_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(SAVE_DIR, exist_ok=True)


# ─── Optuna callback: stop study when target is reached ──────────────────
class StopWhenTargetReached:
    """Stop the Optuna study when a trial reaches the target accuracy."""

    def __init__(self, target: float = TARGET_VAL_ACC):
        self.target = target

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if trial.value is not None and trial.value >= self.target:
            print(f"\n✓ Target val_accuracy {self.target} reached "
                  f"(trial {trial.number}: {trial.value:.4f}). Stopping study.")
            study.stop()


# ─── Build model from trial parameters ──────────────────────────────────
def _build_model(trial, timesteps, size):
    """Dynamically build a CRNN/ConvLSTM model based on Optuna trial params."""

    n_conv_layers = trial.suggest_int("n_conv_layers", 1, 2)
    conv_filters = trial.suggest_categorical("conv_filters", [16, 32, 64])
    rnn_type = trial.suggest_categorical("rnn_type", ["LSTM", "GRU"])
    rnn_units = trial.suggest_categorical("rnn_units", [32, 64, 128])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.05)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    use_bidirectional = trial.suggest_categorical("use_bidirectional", [True, False])
    dense_units = trial.suggest_categorical("dense_units", [32, 64, 128])

    layers = [tf.keras.layers.Input(shape=(timesteps, size, size, 1))]

    # --- TimeDistributed Conv2D blocks ---
    for i in range(n_conv_layers):
        filters = conv_filters * (2 ** i)  # double filters for 2nd layer
        layers.append(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(filters, (3, 3), activation="relu", padding="same")
            )
        )
        layers.append(tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()))

    layers.append(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))))
    layers.append(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
    layers.append(tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(dropout_rate)))

    # --- Recurrent layer ---
    RNNCell = tf.keras.layers.LSTM if rnn_type == "LSTM" else tf.keras.layers.GRU
    rnn_layer = RNNCell(rnn_units, activation="tanh", return_sequences=False)
    if use_bidirectional:
        rnn_layer = tf.keras.layers.Bidirectional(rnn_layer)
    layers.append(rnn_layer)
    layers.append(tf.keras.layers.BatchNormalization())
    layers.append(tf.keras.layers.Dropout(dropout_rate))

    # --- Dense head ---
    layers.append(tf.keras.layers.Dense(dense_units, activation="relu"))
    layers.append(tf.keras.layers.BatchNormalization())
    layers.append(tf.keras.layers.Dropout(dropout_rate))
    layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))

    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─── Optuna objective ────────────────────────────────────────────────────
def _make_objective(X_train, y_train, X_val, y_val, size, gen):
    """Return an Optuna objective function closed over the data arrays."""

    timesteps = gen - 1

    # Reshape once: (n, size, size, gen-1) → (n, timesteps, size, size, 1)
    X_train_5d = X_train.reshape((-1, timesteps, size, size, 1)).astype("float32")
    y_train_f = y_train.astype("float32")
    X_val_5d = X_val.reshape((-1, timesteps, size, size, 1)).astype("float32")
    y_val_f = y_val.astype("float32")

    def objective(trial: optuna.Trial) -> float:
        # Clear previous Keras session to avoid memory leaks
        tf.keras.backend.clear_session()

        model = _build_model(trial, timesteps, size)

        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=5,
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
        ]

        # Optuna pruning callback — stop hopeless trials early
        class OptunaEpochPruning(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                val_acc = logs.get("val_accuracy")
                if val_acc is not None:
                    trial.report(val_acc, epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

        callbacks.append(OptunaEpochPruning())

        history = model.fit(
            X_train_5d, y_train_f,
            validation_data=(X_val_5d, y_val_f),
            epochs=50,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
        )

        val_acc = max(history.history["val_accuracy"])

        # Save model if target reached
        if val_acc >= TARGET_VAL_ACC:
            save_path = os.path.join(SAVE_DIR, f"optuna_best_trial{trial.number}_{val_acc:.4f}.keras")
            model.save(save_path)
            print(f"  ★ Saved model → {save_path}")

        return val_acc

    return objective


# ─── Public entry point ──────────────────────────────────────────────────
def run_optuna_search(X_train_array, y_train_array,
                      X_val_array, y_val_array,
                      size, gen,
                      n_trials=50, study_name="gol_crnn_search"):
    """
    Run the Optuna hyperparameter search and return the study.

    Parameters
    ----------
    X_train_array : np.ndarray, shape (n, size, size, gen-1)
    y_train_array : np.ndarray, shape (n, 1)
    X_val_array   : np.ndarray, shape (n, size, size, gen-1)
    y_val_array   : np.ndarray, shape (n, 1)
    size          : int, board side length
    gen           : int, number of generations
    n_trials      : int, max number of Optuna trials
    study_name    : str, name for the Optuna study

    Returns
    -------
    optuna.Study
    """
    objective = _make_objective(X_train_array, y_train_array,
                                X_val_array, y_val_array,
                                size, gen)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[StopWhenTargetReached(TARGET_VAL_ACC)],
        show_progress_bar=True,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("OPTUNA SEARCH COMPLETE")
    print("=" * 60)
    print(f"Best trial #{study.best_trial.number}")
    print(f"  Val accuracy : {study.best_trial.value:.4f}")
    print(f"  Params       : {study.best_trial.params}")
    print("=" * 60)

    return study


# ─── Standalone mode ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from functions import load_reverse_df, prepare_reverse_dataset, to_numpy_4d

    SIZE = 10
    AMOUNT_BOARDS = 100000
    gen = 2

    print("Loading data...")
    reverse_df = load_reverse_df(SIZE, AMOUNT_BOARDS, gen)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_reverse_dataset(
        reverse_df, SIZE, gen, target_pixel_index=0, test_size=0.1, val_size=0.1, random_state=365
    )
    X_train_array, X_val_array, X_test_array, y_train_array, y_val_array, y_test_array = (
        to_numpy_4d(X_train, X_val, X_test, y_train, y_val, y_test, SIZE, gen)
    )

    print(f"Train: {X_train_array.shape}, Val: {X_val_array.shape}")
    study = run_optuna_search(X_train_array, y_train_array,
                              X_val_array, y_val_array,
                              SIZE, gen, n_trials=50)
