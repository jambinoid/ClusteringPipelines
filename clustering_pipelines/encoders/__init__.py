from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike


class Encoder(ABC):
    """Base abstract class for encoders."""
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        """Name of an encoder."""
        return self._name

    def __call__(self, x: ArrayLike) -> np.ndarray:
        """
        Get embeddings of an input from encoder.

        Args:
            x (array-like): ArrayLike of data to get embeddings for.

        Return:
            array-like with embeddings of dimension `[len(x), self.dim]`
        
        """
        return self.encode(x)

    @abstractmethod
    def _encode(x: ArrayLike) -> np.ndarray:
        pass

    @abstractmethod
    def _encode_in_batches(x: ArrayLike, batch_size: int) -> np.ndarray:
        pass

    def encode(
        self,
        x: ArrayLike,
        batch_size: int | None = None
    ) -> np.ndarray:
        """
        Get embeddings of an input from encoder.

        Args:
            x (array-like): array of data to get embeddings for.
            batch_size (int, optional): size of a batches used.
                Defaults to None.

        Return:
            array-like with embeddings of dimension `[len(x), self.dim]`
        
        """
        if batch_size is None:
            return self._encode(x)
        else:
            return self._encode_in_batches(x, batch_size)
