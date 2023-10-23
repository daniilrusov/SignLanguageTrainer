from typing import Any, Sequence
from typing_extensions import Protocol
from tqdm import tqdm
import string
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from os.path import dirname as up
import os
from dataclasses import dataclass
import json
from typing import List, Dict, Any
from pprint import pprint
from tqdm import tqdm
import numpy as np
import random
from typing import Optional
from dataset.slovo_dataset import (
    SlovoDataset,
    SlovoDatasetInterface,
    SLOVO_ANNOTATIONS_TABLE,
    SlovoDatasetSubset,
)
from dataset.slovo_ds_sample import SlovoDatasetSample, LabelData
from dataset.slovo_ds_structure import SlovoDsStructure
from utils import (
    read_pickle,
    write_pickle,
    get_all_files_paths_in_dir,
    plot_keypoints_on_frame,
)

WS_DIR = up(up(__file__))
DATA_DIR = os.path.join(WS_DIR, "data")
SLOVO_DATASET_DIR = os.path.join(DATA_DIR, "slovo")


def dump_mediapipe_keys():
    ds_structure: SlovoDsStructure = SlovoDsStructure(SLOVO_DATASET_DIR)
    with open(ds_structure.mediapipe_labels, "r") as f:
        data = json.load(f)
    mediapipe_keys = list(data.keys())
    output_pickle = "mp_keys.pkl"
    write_pickle(pickle_path=output_pickle, data=mediapipe_keys)
    read_keys = read_pickle(output_pickle)
    print("READ KEYS")
    pprint(read_keys)


def check_mediapipe_labels() -> None:
    ds_structure: SlovoDsStructure = SlovoDsStructure(SLOVO_DATASET_DIR)
    output_pickle = "mp_keys.pkl"
    read_keys = read_pickle(output_pickle)
    train_videos = get_all_files_paths_in_dir(ds_structure.train_folder)
    test_videos = get_all_files_paths_in_dir(ds_structure.test_folder)
    train_videos_names = [
        os.path.splitext(os.path.basename(p))[0] for p in train_videos
    ]
    test_videos_names = [os.path.splitext(os.path.basename(p))[0] for p in test_videos]
    #
    train_stats = [name in read_keys for name in train_videos_names]
    test_stats = [name in read_keys for name in test_videos_names]
    print("train stats")
    print(train_stats.count(True))
    print(train_stats.count(False))

    print("test stats")
    print(test_stats.count(True))
    print(test_stats.count(False))


def make_viz_of_sample(sample: Dict, output_dir: str) -> None:
    video_path = sample["video_path"]
    mediapipe_keypoints = sample["mediapipe_keypoints"]
    video_len = 0
    capture = cv2.VideoCapture(video_path)
    while True:
        ret, bgr_frame = capture.read()
        if not ret:
            break
        rgb_frame = bgr_frame[:, :, ::-1]
        if video_len % 10 == 0:
            viz = plot_keypoints_on_frame(
                frame=rgb_frame, keypoints=mediapipe_keypoints[video_len]
            )
            cv2.imwrite(
                os.path.join(output_dir, f"viz_{video_len}.png"), viz[:, :, ::-1]
            )

        video_len += 1
    print(f"video_len = {video_len}")


def is_word_letter(word: str):
    return word.lower() != word and len(word) == 1


def create_label_name2id_mappig(table: SLOVO_ANNOTATIONS_TABLE) -> Dict[str, int]:
    all_labels = table["text"].to_list()
    uniq_labels: List[str] = list(set(all_labels))
    label2id_mapping = {label: id for id, label in enumerate(uniq_labels)}
    return label2id_mapping


def get_gestures_subset(
    labels: List[str], base_dataset: SlovoDatasetInterface
) -> SlovoDatasetSubset:
    sample: SlovoDatasetSample
    indexes_subset: List[int] = []
    for index, sample in tqdm(enumerate(base_dataset), desc="Creating gestures subset"):
        if sample.label_data.label_name not in labels:
            continue
        indexes_subset.append(index)
    gestures_subset = SlovoDatasetSubset(base_dataset, indexes_subset)
    return gestures_subset


def main():
    ds_structure: SlovoDsStructure = SlovoDsStructure(SLOVO_DATASET_DIR)
    table = pd.read_csv(
        ds_structure.annotations_path, on_bad_lines="skip", delimiter="\t"
    )
    mapping = create_label_name2id_mappig(table)
    dataset: SlovoDatasetInterface = SlovoDataset(
        is_train=False,
        slovo_ds_structure=ds_structure,
        read_mediapipe_keypoints=False,
        label_name2id=mapping,
    )
    print("Base dataset info")
    print(len(dataset))
    pprint(dataset[0])

    print("subset info")
    subset: SlovoDatasetInterface = get_gestures_subset(
        base_dataset=dataset, labels=["А", "Б"]
    )
    print(len(subset))
    pprint(subset[0])
    pprint(subset.label_name2id)


if __name__ == "__main__":
    main()
