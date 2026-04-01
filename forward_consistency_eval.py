import numpy as np


def game_of_life_step(boards):
    """Apply one deterministic Game of Life step to a batch of binary boards.

    Parameters
    ----------
    boards : np.ndarray
        Shape ``(batch, H, W)`` with values 0 or 1.

    Returns
    -------
    np.ndarray
        Shape ``(batch, H, W)`` — the next generation for every board.
    """
    # Pad with zeros (dead borders, no wrapping)
    padded = np.pad(boards, ((0, 0), (1, 1), (1, 1)), mode='constant')

    # Sum the 8 neighbours via shifted slices
    neighbours = (
        padded[:, :-2, :-2] + padded[:, :-2, 1:-1] + padded[:, :-2, 2:] +
        padded[:, 1:-1, :-2]                         + padded[:, 1:-1, 2:] +
        padded[:, 2:, :-2]  + padded[:, 2:, 1:-1]   + padded[:, 2:, 2:]
    )

    # Conway's rules:
    #   alive cell survives if 2 or 3 neighbours
    #   dead cell becomes alive if exactly 3 neighbours
    next_gen = ((boards == 1) & ((neighbours == 2) | (neighbours == 3))) | \
               ((boards == 0) & (neighbours == 3))

    return next_gen.astype(np.int32)


def evaluate_forward_consistency(model, X_test_T, threshold=0.5, ground_truth=None):
    """Measure how often predicted T-1 boards evolve back to the known T boards.

    Workflow
    --------
    1. ``predicted_T_minus_1 = model.predict(X_test_T)``
    2. Binarise with *threshold*.
    3. Step the predicted T-1 boards forward one generation using exact GoL rules.
    4. Compare the forward-stepped result to *ground_truth* (the board at T).

    Parameters
    ----------
    model : tf.keras.Model
        Trained model that takes board(s) at time T and predicts T-1.
        Input shape ``(batch, H, W, C)``; output ``(batch, H, W, 1)``.
    X_test_T : np.ndarray
        Model input — boards at generation T.  Shape ``(N, H, W, C)``.
    threshold : float
        Value above which a predicted cell is considered alive.
    ground_truth : np.ndarray or None
        The single board to compare against after forward-stepping, shape
        ``(N, H, W)`` or ``(N, H, W, 1)``.  If *None*, the **last channel**
        of *X_test_T* is used (the feature board closest in time to the
        predicted target).

    Returns
    -------
    dict
        ``pixel_accuracy``  — fraction of individual cells that match.
        ``board_accuracy``  — fraction of boards that match *exactly*.
    """
    # 1. Predict T-1
    pred_proba = model.predict(X_test_T, verbose=0)  # (N, H, W, 1)

    # 2. Binarise
    pred_binary = (pred_proba > threshold).astype(np.int32)

    # 3. Squeeze channel dim -> (N, H, W) for the GoL step
    pred_boards = pred_binary.squeeze(axis=-1)

    if ground_truth is not None:
        target_boards = np.asarray(ground_truth).astype(np.int32)
    else:
        # Use the last channel of X_test_T as the comparison board
        target_boards = X_test_T[:, :, :, -1].astype(np.int32)

    if target_boards.ndim == 4:
        target_boards = target_boards.squeeze(axis=-1)

    # 4. Forward step
    forward_stepped = game_of_life_step(pred_boards)

    # 5. Compare
    match_map = (forward_stepped == target_boards)
    pixel_accuracy = match_map.mean()
    board_accuracy = match_map.all(axis=(1, 2)).mean()

    print(f"Forward-consistency  |  Pixel acc: {pixel_accuracy:.4f}  |  Board acc: {board_accuracy:.4f}")
    return {
        'pixel_accuracy': float(pixel_accuracy),
        'board_accuracy': float(board_accuracy),
    }
