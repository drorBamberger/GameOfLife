import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import traceback

from functions import (
    load_reverse_df,
    prepare_reverse_dataset,
    to_numpy_4d,
    build_and_train_nn,
    build_and_train_cnn,
    build_and_train_rnn,
    build_and_train_rcnn,
)


def model_accuracy_and_f1(y_true, y_pred_raw, softmax=False):
    y_true = y_true.flatten().astype(int)
    if softmax:
        y_pred = np.argmax(y_pred_raw, axis=1).astype(int)
    else:
        y_pred = (y_pred_raw > 0.5).astype(int).flatten()
    y_true = y_true[: len(y_pred)]
    return float(accuracy_score(y_true, y_pred)), float(f1_score(y_true, y_pred, zero_division=0))


def rnn_input(X, size, gen):
    return X.reshape((-1, gen - 1, size * size)).astype('float32')


def rcnn_input(X, size, gen):
    gen_data = gen - 1
    num_samples = X.shape[0] - gen_data
    X_rcnn = np.zeros((num_samples, gen_data * gen_data, size, size, 1), dtype='float32')
    for i in range(num_samples):
        X_rcnn[i] = X[i : i + gen_data].reshape(gen_data * gen_data, size, size, 1)
    return X_rcnn


def run_cycle(size=5, amount_boards=1000, gen=3, epochs=2, batch_size=32):
    print(f"Starting cycle: size={size}, boards={amount_boards}, gen={gen}, epochs={epochs}", flush=True)
    reverse_df = load_reverse_df(size, amount_boards, gen)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_reverse_dataset(
        reverse_df, size, gen, target_pixel_index=0, test_size=0.1, val_size=0.1, random_state=365
    )
    X_train_a, X_val_a, X_test_a, y_train_a, y_val_a, y_test_a = to_numpy_4d(
        X_train, X_val, X_test, y_train, y_val, y_test, size, gen
    )

    rows = []

    # Baseline NN
    print("Training NN baseline...", flush=True)
    nn_base, _ = build_and_train_nn(X_train_a, y_train_a, size, epochs=epochs, batch_size=batch_size,
                                    use_callbacks=False, use_class_weight=False, fit_verbose=0)
    acc, f1 = model_accuracy_and_f1(y_test_a, nn_base.predict(X_test_a, verbose=0))
    rows.append(("NN", "baseline", acc, f1))

    # Improved NN
    print("Training NN improved...", flush=True)
    nn_imp, _ = build_and_train_nn(X_train_a, y_train_a, size, epochs=epochs, batch_size=batch_size,
                                   use_callbacks=True, use_class_weight=True, fit_verbose=0)
    acc, f1 = model_accuracy_and_f1(y_test_a, nn_imp.predict(X_test_a, verbose=0))
    rows.append(("NN", "improved", acc, f1))

    # Baseline CNN
    print("Training CNN baseline...", flush=True)
    cnn_base, _ = build_and_train_cnn(X_train_a, y_train_a, size, epochs=epochs, batch_size=batch_size,
                                      use_callbacks=False, use_class_weight=False, fit_verbose=0)
    acc, f1 = model_accuracy_and_f1(y_test_a, cnn_base.predict(X_test_a, verbose=0))
    rows.append(("CNN", "baseline", acc, f1))

    # Improved CNN
    print("Training CNN improved...", flush=True)
    cnn_imp, _ = build_and_train_cnn(X_train_a, y_train_a, size, epochs=epochs, batch_size=batch_size,
                                     use_callbacks=True, use_class_weight=True, fit_verbose=0)
    acc, f1 = model_accuracy_and_f1(y_test_a, cnn_imp.predict(X_test_a, verbose=0))
    rows.append(("CNN", "improved", acc, f1))

    # Baseline RNN
    print("Training RNN baseline...", flush=True)
    rnn_base, _ = build_and_train_rnn(X_train_a, y_train_a, size, gen, epochs=epochs, batch_size=batch_size,
                                      use_callbacks=False, use_class_weight=False, fit_verbose=0)
    X_test_rnn = rnn_input(X_test_a, size, gen)
    acc, f1 = model_accuracy_and_f1(y_test_a, rnn_base.predict(X_test_rnn, verbose=0))
    rows.append(("RNN", "baseline", acc, f1))

    # Improved RNN
    print("Training RNN improved...", flush=True)
    rnn_imp, _ = build_and_train_rnn(X_train_a, y_train_a, size, gen, epochs=epochs, batch_size=batch_size,
                                     use_callbacks=True, use_class_weight=True, fit_verbose=0)
    acc, f1 = model_accuracy_and_f1(y_test_a, rnn_imp.predict(X_test_rnn, verbose=0))
    rows.append(("RNN", "improved", acc, f1))

    # Baseline RCNN-sigmoid
    print("Training RCNN-sigmoid baseline...", flush=True)
    rcnn_sig_base, _ = build_and_train_rcnn(gen, X_train_a, y_train_a, size, batch_size=batch_size, epochs=epochs,
                                            active="sigmoid", use_callbacks=False, use_class_weight=False, fit_verbose=0)
    X_test_rcnn = rcnn_input(X_test_a, size, gen)
    acc, f1 = model_accuracy_and_f1(y_test_a[: len(X_test_rcnn)], rcnn_sig_base.predict(X_test_rcnn, verbose=0))
    rows.append(("RCNN-sigmoid", "baseline", acc, f1))

    # Improved RCNN-sigmoid
    print("Training RCNN-sigmoid improved...", flush=True)
    rcnn_sig_imp, _ = build_and_train_rcnn(gen, X_train_a, y_train_a, size, batch_size=batch_size, epochs=epochs,
                                           active="sigmoid", use_callbacks=True, use_class_weight=True, fit_verbose=0)
    acc, f1 = model_accuracy_and_f1(y_test_a[: len(X_test_rcnn)], rcnn_sig_imp.predict(X_test_rcnn, verbose=0))
    rows.append(("RCNN-sigmoid", "improved", acc, f1))

    # Baseline RCNN-softmax
    print("Training RCNN-softmax baseline...", flush=True)
    rcnn_soft_base, _ = build_and_train_rcnn(gen, X_train_a, y_train_a, size, batch_size=batch_size, epochs=epochs,
                                             active="softmax", use_callbacks=False, use_class_weight=False, fit_verbose=0)
    acc, f1 = model_accuracy_and_f1(y_test_a[: len(X_test_rcnn)], rcnn_soft_base.predict(X_test_rcnn, verbose=0), softmax=True)
    rows.append(("RCNN-softmax", "baseline", acc, f1))

    # Improved RCNN-softmax
    print("Training RCNN-softmax improved...", flush=True)
    rcnn_soft_imp, _ = build_and_train_rcnn(gen, X_train_a, y_train_a, size, batch_size=batch_size, epochs=epochs,
                                            active="softmax", use_callbacks=True, use_class_weight=True, fit_verbose=0)
    acc, f1 = model_accuracy_and_f1(y_test_a[: len(X_test_rcnn)], rcnn_soft_imp.predict(X_test_rcnn, verbose=0), softmax=True)
    rows.append(("RCNN-softmax", "improved", acc, f1))

    df = pd.DataFrame(rows, columns=["model", "variant", "accuracy", "f1"])
    pivot_acc = df.pivot(index="model", columns="variant", values="accuracy")
    pivot_f1 = df.pivot(index="model", columns="variant", values="f1")
    df["size"] = size
    df["gen"] = gen
    df["amount_boards"] = amount_boards

    print("\n=== Results ===")
    print(df.to_string(index=False))
    print("\n=== Accuracy Delta (improved - baseline) ===")
    print((pivot_acc["improved"] - pivot_acc["baseline"]).to_string())
    print("\n=== F1 Delta (improved - baseline) ===")
    print((pivot_f1["improved"] - pivot_f1["baseline"]).to_string())

    df.to_csv("improvement_cycle_results.csv", index=False)
    return df


if __name__ == "__main__":
    try:
        run_cycle(size=5, amount_boards=100000, gen=3, epochs=2, batch_size=32)
    except Exception as e:
        print(f"Run failed: {e}", flush=True)
        traceback.print_exc()
