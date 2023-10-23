import os
from dataclasses import dataclass


@dataclass
class SlovoDsStructure:
    """Dataclass for accessing slovo dataset properties"""

    root_dir: str

    @property
    def test_folder(self) -> str:
        return os.path.join(self.root_dir, "test")

    @property
    def train_folder(self) -> str:
        return os.path.join(self.root_dir, "train")

    @property
    def annotations_path(self) -> str:
        return os.path.join(self.root_dir, "annotations.csv")

    @property
    def mediapipe_labels(self) -> str:
        return os.path.join(self.root_dir, "slovo_mediapipe.json")
