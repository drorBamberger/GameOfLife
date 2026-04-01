import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score


def evaluate_depth_impact(model_builder_fn, data_loader_fn, max_generations=5,
                          min_generations=2, epochs=10, batch_size=32, threshold=0.5):
    """Evaluate how prediction quality degrades with increasing history depth.

    Parameters
    ----------
    model_builder_fn : callable(input_shape) -> tf.keras.Model
        A factory that returns a *compiled* Keras model.  It receives the
        input shape ``(H, W, C)`` so it can adapt to each depth's channel count.
    data_loader_fn : callable(generation) -> (X_train, y_train, X_test, y_test)
        Returns data arrays for the given generation depth.
        * ``X_train / X_test``: shape ``(N, H, W, C)``
        * ``y_train / y_test``: shape ``(N, ...)`` matching the model output.
    max_generations : int
        Upper bound of the depth loop (inclusive).
    min_generations : int
        Lower bound of the depth loop (inclusive).  Default 2 (skips gen=1
        which has 0 history channels).
    epochs : int
        Training epochs per depth.
    batch_size : int
        Training batch size.
    threshold : float
        Binarisation threshold applied to model predictions.

    Returns
    -------
    results : dict
        ``{gen: {'accuracy': float, 'f1': float}}`` for every depth.
    """
    results = {}

    for gen in range(min_generations, max_generations + 1):
        print(f"\n{'='*40}")
        print(f"  Generation depth: {gen}")
        print(f"{'='*40}")

        X_train, y_train, X_test, y_test = data_loader_fn(gen)

        input_shape = X_train.shape[1:]
        model = model_builder_fn(input_shape)

        model.fit(X_train, y_train,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_split=0.1,
                  verbose=0)

        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > threshold).astype(int).flatten()
        y_true = y_test.flatten()

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        results[gen] = {'accuracy': acc, 'f1': f1}
        print(f"  Accuracy: {acc:.4f}  |  F1: {f1:.4f}")

    _plot_results(results)
    return results


def _plot_results(results):
    """Plot accuracy and F1 vs generation depth."""
    gens = sorted(results.keys())
    accs = [results[g]['accuracy'] for g in gens]
    f1s = [results[g]['f1'] for g in gens]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gens, accs, 'o-', label='Accuracy')
    ax.plot(gens, f1s, 's--', label='F1-Score')
    ax.set_xlabel('Generation Depth')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance vs. History Depth')
    ax.set_xticks(gens)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('depth_impact.png', dpi=150)
    plt.show()
