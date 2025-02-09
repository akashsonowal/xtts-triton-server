import abc
from pathlib import Path

class BaseVocalizer(abc.ABC):

    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)

    @abc.abstractmethod
    def load_model(self):
        """Loads the Vocalizer model"""
        pass