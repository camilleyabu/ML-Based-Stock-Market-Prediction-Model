"""
Transformer encoder architecture for time-series regression.
Used by aapl_transformer.py and nvda_transformer.py.

Architecture: input projection → sinusoidal positional encoding →
N encoder blocks (multi-head attention + FFN + LayerNorm) →
global average pooling → Dense head.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


def _positional_encoding(seq_len: int, d_model: int) -> tf.Tensor:
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""
    positions = np.arange(seq_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)  # (1, seq_len, d_model)


def _encoder_block(x, d_model: int, num_heads: int,
                   ff_dim: int, dropout_rate: float):
    """Single Transformer encoder block: attention + FFN with residuals."""
    # Multi-head self-attention
    attn = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads
    )(x, x)
    attn = layers.Dropout(dropout_rate)(attn)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn)

    # Position-wise feed-forward
    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(d_model)(ff)
    ff = layers.Dropout(dropout_rate)(ff)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff)
    return x


def build_transformer(seq_len: int, n_features: int,
                      d_model: int = 64, num_heads: int = 4,
                      ff_dim: int = 128, num_blocks: int = 2,
                      dropout: float = 0.1) -> Model:
    """
    Build a Transformer encoder model for next-day return regression.

    Args:
        seq_len:    Input sequence length (time steps).
        n_features: Number of input features per time step.
        d_model:    Internal embedding dimension.
        num_heads:  Number of attention heads.
        ff_dim:     Feed-forward hidden dimension.
        num_blocks: Number of stacked encoder blocks.
        dropout:    Dropout rate applied after attention and FFN.

    Returns:
        Compiled Keras Model.
    """
    inputs = layers.Input(shape=(seq_len, n_features))

    # Project raw features to d_model dimensions
    x = layers.Dense(d_model)(inputs)

    # Add positional encoding (fixed sinusoidal, not learned)
    x = x + _positional_encoding(seq_len, d_model)
    x = layers.Dropout(dropout)(x)

    # Stacked encoder blocks
    for _ in range(num_blocks):
        x = _encoder_block(x, d_model, num_heads, ff_dim, dropout)

    # Aggregate over time → scalar prediction
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")
    return model
