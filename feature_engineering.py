import numpy as np
import tensorflow as tf


def _make_ones_kernel(radius):
    """Create a square kernel of ones with side length (2*radius+1).

    The center cell is included in the count so the caller can subtract it
    if a *neighbour-only* count is needed.
    """
    side = 2 * radius + 1
    kernel = np.ones((side, side, 1, 1), dtype=np.float32)
    return tf.constant(kernel)


def add_radius_features(boards_batch, radii=(1, 2, 3)):
    """Add neighbourhood-count feature channels for each requested radius.

    Parameters
    ----------
    boards_batch : np.ndarray or tf.Tensor
        Shape ``(batch, 10, 10, C)`` where ``C >= 1``.
        The *first* channel (index 0) is treated as the binary cell grid.
    radii : tuple of int
        Each value ``r`` produces a feature channel counting live cells
        inside the ``(2r+1) x (2r+1)`` window centred on each cell
        (excluding the cell itself).

    Returns
    -------
    tf.Tensor
        Shape ``(batch, 10, 10, C + len(radii))`` — the original channels
        concatenated with one new channel per radius.
    """
    boards_batch = tf.cast(boards_batch, tf.float32)
    # Use only the first channel as the binary life grid
    life_channel = boards_batch[:, :, :, :1]  # (B, H, W, 1)

    extras = []
    for r in radii:
        kernel = _make_ones_kernel(r)
        # 'SAME' pads with zeros so output keeps (H, W) shape
        counts = tf.nn.conv2d(life_channel, kernel, strides=1, padding='SAME')
        # Subtract the centre cell so we only count neighbours
        neighbour_counts = counts - life_channel
        extras.append(neighbour_counts)

    return tf.concat([boards_batch] + extras, axis=-1)
