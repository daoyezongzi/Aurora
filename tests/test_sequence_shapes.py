from __future__ import annotations

import numpy as np

from aurora_ml.dataset import build_sequences


def test_sequence_window_shape_is_20_to_1() -> None:
    features = np.arange(60, dtype=np.float32).reshape(30, 2)
    labels = (np.arange(30) % 2).astype(np.float32)

    x, y = build_sequences(features, labels, sequence_length=20)

    assert x.shape == (11, 20, 2)
    assert y.shape == (11,)
    np.testing.assert_allclose(x[0], features[:20])
    assert y[0] == labels[19]
