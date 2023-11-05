from dataset.slovo_dataset import SlovoDatasetInterface
from dataset.slovo_ds_sample import SlovoDatasetSample, LabelData
from typing import List, Dict
from dataclasses import dataclass
from deepdiff import DeepDiff
import json
import os


@dataclass
class MMActionVideoDatasetDescription:
    files_root: str
    train_data_root: str
    val_data_root: str

    def __post_init__(self):
        os.makedirs(self.files_root, exist_ok=True)

    @property
    def ann_file_train_path(self) -> str:
        return os.path.join(self.files_root, "train_ann.txt")

    @property
    def ann_file_val_path(self) -> str:
        return os.path.join(self.files_root, "val_ann.txt")

    @property
    def labels_mapping_path(self) -> str:
        return os.path.join(self.files_root, "label2id_mapping.json")

    def register_label2id_mapping_asserting_consistense(
        self, label2id_mapping: Dict[str, int]
    ):
        # dump mapping if not already present
        if not os.path.exists(self.labels_mapping_path):
            with open(self.labels_mapping_path, "w", encoding="utf-8") as f:
                json.dump(label2id_mapping, f, indent=2, ensure_ascii=False)
            return
        # if there is already mapping check that new one is consistent
        with open(self.labels_mapping_path, "r", encoding="utf-8") as f:
            existing_mapping: Dict[str, int] = json.load(f)
        if DeepDiff(existing_mapping, label2id_mapping):
            raise ValueError(
                f"Labels mapping {label2id_mapping} is incosistent with existing {existing_mapping}"
            )


def spawn_txt_and_labels_mapping_for_mmaction_VideoDataset_init(
    slovo_dataset: SlovoDatasetInterface,
    mmaction_ds_descr: MMActionVideoDatasetDescription,
    is_train: bool,
):
    mmaction_ds_descr.register_label2id_mapping_asserting_consistense(
        slovo_dataset.label_name2id
    )
    lines: List[str] = []
    sample: SlovoDatasetSample
    videos_root = (
        mmaction_ds_descr.train_data_root
        if is_train
        else mmaction_ds_descr.val_data_root
    )
    for sample in slovo_dataset:
        video_path: str = sample.video_path
        video_rel_path = os.path.relpath(video_path, start=videos_root)
        label_id = sample.label_data.label_id
        line = f"{video_rel_path} {label_id}\n"
        lines.append(line)

    txt_output_path: str = (
        mmaction_ds_descr.ann_file_train_path
        if is_train
        else mmaction_ds_descr.ann_file_val_path
    )
    with open(txt_output_path, "w") as f:
        f.writelines(lines)
