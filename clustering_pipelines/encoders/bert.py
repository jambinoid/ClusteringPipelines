from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
import tensorflow as tf
import tensorflow_text
import tensorflow_hub as tfhub

from . import Encoder


class SmallBERT(Encoder):
    """Wrapper for the SmallBERT family."""

    def __init__(
        self,
        L: Literal[2, 4, 6, 8, 10, 12] = 8,
        H: Literal[128, 256, 512, 768] = 512,
        base_name: str = "BERT"
    ):
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
        preprocessor = tfhub.KerasLayer(
            f"https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
        encoder_inputs = preprocessor(text_input)
        encoder = tfhub.KerasLayer(
            f"https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-{L}_H-{H}_A-8/2",
            trainable=False)
        outputs = encoder(encoder_inputs)
        pooled_output = outputs["pooled_output"]
    
        self._encoder = tf.keras.Model(text_input, pooled_output)

        super().__init__(name=f"{base_name}_L-{L}_H-{H}_A-8")

    def _encode(self, x: list[str]) -> np.ndarray:
        return self._encoder(tf.constant(x)).numpy()
    
    def _encode_in_batches(
        self,
        x: ArrayLike,
        batch_size: int
    ) -> np.ndarray:
        return self._encoder.predict(
            tf.constant(x),
            batch_size=batch_size
        )
