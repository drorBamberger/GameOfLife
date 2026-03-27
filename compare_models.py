import time
import traceback
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from functions import (
    load_reverse_df,
    prepare_reverse_dataset,
    to_numpy_4d,
    build_and_train_nn,
    build_and_train_cnn,
    build_and_train_rnn,
    build_and_train_rcnn,
    dec_tree,
)


def rnn_input(X, size, gen):
    gen_data = gen - 1
    return X.reshape((-1, gen_data, size * size)).astype('float32')


def rcnn_input(X, size, gen):
    gen_data = gen - 1
    num_samples = X.shape[0] - gen_data
    if num_samples <= 0:
        raise ValueError(f"Not enough samples ({X.shape[0]}) for gen={gen}")

    X_rcnn = np.zeros((num_samples, gen_data * gen_data, size, size, 1), dtype='float32')
    for i in range(num_samples):
        block = X[i : i + gen_data]
        X_rcnn[i] = block.reshape(gen_data * gen_data, size, size, 1)
    return X_rcnn


def model_accuracy(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba > threshold).astype(int).flatten()
    return accuracy_score(y_true.flatten(), y_pred)


def evaluate_models_for_config(size, amount_boards, gen, epochs=5, batch_size=32):
    print(f"\n=== size={size}, amount_boards={amount_boards}, gen={gen} ===")
    start = time.time()

    reverse_df = load_reverse_df(size, amount_boards, gen)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_reverse_dataset(
        reverse_df, size, gen, target_pixel_index=0, test_size=0.1, val_size=0.1, random_state=365
    )
    X_train_array, X_val_array, X_test_array, y_train_array, y_val_array, y_test_array = to_numpy_4d(
        X_train, X_val, X_test, y_train, y_val, y_test, size, gen
    )

    results = []

    try:
        model = build_and_train_nn(X_train_array, y_train_array, size, epochs=epochs, batch_size=batch_size)
        y_pred = model[0].predict(X_test_array)
        acc = model_accuracy(y_test_array, y_pred)
        results.append(("MLP", acc))
    except Exception:
        traceback.print_exc()
        results.append(("MLP", None))

    try:
        model = build_and_train_cnn(X_train_array, y_train_array, size, epochs=epochs, batch_size=batch_size)
        y_pred = model[0].predict(X_test_array)
        acc = model_accuracy(y_test_array, y_pred)
        results.append(("CNN", acc))
    except Exception:
        traceback.print_exc()
        results.append(("CNN", None))

    try:
        X_test_rnn = rnn_input(X_test_array, size, gen)
        model = build_and_train_rnn(X_train_array, y_train_array, size, gen, epochs=epochs, batch_size=batch_size)
        y_pred = model[0].predict(X_test_rnn)
        acc = model_accuracy(y_test_array, y_pred)
        results.append(("RNN", acc))
    except Exception:
        traceback.print_exc()
        results.append(("RNN", None))

    try:
        X_test_rcnn = rcnn_input(X_test_array, size, gen)
        model = build_and_train_rcnn(gen, X_train_array, y_train_array, size, batch_size=batch_size, epochs=epochs, active="sigmoid")
        y_pred = model[0].predict(X_test_rcnn)
        acc = model_accuracy(y_test_array[: len(y_pred)], y_pred)
        results.append(("RCNN-sigmoid", acc))
    except Exception:
        traceback.print_exc()
        results.append(("RCNN-sigmoid", None))

    try:
        dt_model, _ = dec_tree(X_train, y_train, X_test, y_test, md=10, rs=42)
        y_pred = dt_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append(("DecisionTree", acc))
    except Exception:
        traceback.print_exc()
        results.append(("DecisionTree", None))

    duration = time.time() - start
    print(f"done size={size}, gen={gen} in {duration:.1f}s")

    result_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
    result_df["size"] = size
    result_df["gen"] = gen
    return result_df


def main():
    sizes = [5, 10]
    gens = [2, 3, 4]
    amount_boards = 100000

    all_results = []

    for size in sizes:
        for gen in gens:
            try:
                df = evaluate_models_for_config(size, amount_boards, gen, epochs=5, batch_size=32)
                all_results.append(df)
                print(df)
            except Exception:
                traceback.print_exc()

    if all_results:
        final = pd.concat(all_results, ignore_index=True)
        print("\n=== Final Summary ===")
        print(final)
        final.to_csv("model_comparison_results.csv", index=False)
        print("Saved to model_comparison_results.csv")


if __name__ == "__main__":
    main()
