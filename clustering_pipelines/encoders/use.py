import numpy as np
from numpy.typing import ArrayLike
import tensorflow_hub as tfhub
from tqdm import tqdm

from clustering_pipelines.encoders import Encoder
from clustering_pipelines.utils.dataset_utils import chunkify


class USE(Encoder):
    """Wrapper for the USE."""

    def __init__(self, name: str = "USE") -> None:
        self._encoder = tfhub.load(
            "https://tfhub.dev/google/universal-sentence-encoder/4")
        super().__init__(name=name)

    def _encode(self, x: ArrayLike) -> np.ndarray:
        return self._encoder(x).numpy()
    
    def _encode_in_batches(
        self,
        x: ArrayLike,
        batch_size: int
    ) -> np.ndarray:
        return np.vstack([
            self._encode(batch)
            for batch in tqdm(
                chunkify(x, batch_size),
                total=len(x) // batch_size + 1
            )
        ])
