from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class LabelData:
    label_id: int
    label_name: str


@dataclass
class SlovoDatasetSample:
    video_path: str
    label_data: LabelData
    mediapipe_keypoints: Optional[Dict]
