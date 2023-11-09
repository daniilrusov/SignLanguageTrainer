import pickle
import os
from typing import Any, List, Dict
import numpy as np


def read_pickle(pickle_path: str) -> Any:
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data


def write_pickle(pickle_path: str, data: Any) -> None:
    with open(pickle_path, "wb") as f:
        pickle.dump(data, f)


def get_all_files_paths_in_dir(root_path: str) -> List[str]:
    dir_content_names: List[str] = sorted(os.listdir(root_path))
    dir_content_paths: List[str] = [
        os.path.join(root_path, dir_content_name)
        for dir_content_name in dir_content_names
    ]
    return dir_content_paths


def plot_keypoints_on_frame(frame: np.ndarray, keypoints: Dict) -> np.ndarray:
    viz = frame.copy()
    h, w, _ = frame.shape

    # cv2.circle(image, center_coordinates, radius, color, thickness)
    for hand_name, hand_keypoints in keypoints.items():
        for kp in hand_keypoints:
            x_norm = kp["x"]
            y_norm = kp["y"]
            x = int(x_norm * w)
            y = int(y_norm * h)
            viz = cv2.circle(viz, (x, y), 2, color=(0, 255, 0), thickness=3)
    return viz
