from abc import ABC, abstractmethod, abstractproperty
import os
from pathlib import Path

from clustering_pipelines.utils.dataset_utils import load_and_extract


_DEFAULT_DIR = Path(__file__).parent.parent / '.datasets'


class DatasetLoader(ABC):
    """Base abstract class for dataset loaders."""

    def __init__(self, name: str):
        self.name = name
        self._dir = Path(os.environ.get("DATASETS_CACHE_DIR", _DEFAULT_DIR))

    @abstractproperty
    def _dataset_name(self) -> str:
        """Name of a dataset."""
        return

    @property
    def path(self) -> Path:
        """Local path where dataset is stored."""
        return self._dir / self._dataset_name 

    @abstractproperty
    def url(self) -> str:
        """Url address to access dataset."""
        return

    def _load(self) -> None:
        """Extract dataset to the `self.path`"""
        load_and_extract(
            self.url,
            self.path,
            self.url.split("/")[-1].split(".", 1)[-1]
        )

    # TODO: use numpy arrays?
    @abstractmethod
    def _parse(self) -> tuple[list, list]:
        """
        Read dataset from the `self.path` and parse it.

        Returns:
            Tuple of three lists: list of texts and list of classes.

        """
        return

    def __call__(self) -> tuple[list, list]:
        """
        Extract dataset to the `self.path` if it's not already extracted,
        read it and parse it.

        Returns:
            Tuple of two lists: list of texts and list of classes.

        """
        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)
            self._load()
        return self._parse()
