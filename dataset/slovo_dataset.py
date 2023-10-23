import os
from tqdm import tqdm
from typing import Dict, Optional, Sequence, List
from dataset.slovo_ds_sample import SlovoDatasetSample, LabelData
from dataset.slovo_ds_structure import SlovoDsStructure
from torch.utils.data import Dataset, Subset, ConcatDataset
from utils import get_all_files_paths_in_dir
import pandas as pd
import json

SLOVO_ANNOTATIONS_TABLE = pd.DataFrame


class SlovoDatasetInterface(Dataset):
    @property
    def label_name2id(self) -> Dict[str, int]:
        pass

    def __getitem__(self, index) -> SlovoDatasetSample:
        return super().__getitem__(index)


class SlovoDatasetSubset(Subset, SlovoDatasetInterface):
    def __init__(self, dataset: SlovoDatasetInterface, indices: Sequence[int]) -> None:
        super().__init__(dataset, indices)
        subset_labels: List[str] = [dataset[i].label_data.label_name for i in indices]
        uniq_labels = sorted(list(set(subset_labels)))
        self._label_name2id: Dict[str, int] = {
            label: index for index, label in enumerate(uniq_labels)
        }

    @property
    def label_name2id(self) -> Dict[str, int]:
        return self._label_name2id

    def __getitem__(self, idx):
        sample: SlovoDatasetSample = super().__getitem__(idx)
        label_name: str = sample.label_data.label_name
        subset_label_id: int = self.label_name2id[label_name]
        sample.label_data.label_id = subset_label_id
        return sample


class SlovoDatasetConcat(ConcatDataset, SlovoDatasetInterface):
    @property
    def label_name2id(self) -> Dict[str, int]:
        pass


class SlovoDataset(SlovoDatasetInterface):
    def __init__(
        self,
        is_train: bool,
        slovo_ds_structure: SlovoDsStructure,
        read_mediapipe_keypoints: bool,
        label_name2id: Dict[str, int],
    ) -> None:
        self._label_name2id: Dict[str, int] = label_name2id
        self.slovo_ds_strture: SlovoDsStructure = slovo_ds_structure
        self.is_train: bool = is_train
        self.read_mediapipe_keypoints: bool = read_mediapipe_keypoints
        self.mediapipe_keypoints: Optional[Dict] = None
        self.annotations_table = pd.read_csv(
            self.slovo_ds_strture.annotations_path,
            on_bad_lines="skip",
            delimiter="\t",
        )
        if self.read_mediapipe_keypoints:
            with open(self.slovo_ds_strture.mediapipe_labels, "r") as f:
                self.mediapipe_keypoints = json.load(f)
        self.samples_data: List[Dict] = self._prepare_data()
        super().__init__()

    @property
    def label_name2id(self) -> Dict[str, int]:
        return self._label_name2id

    def _get_video_id_in_table(self, video_path: str) -> str:
        table_id = os.path.basename(video_path)
        table_id = os.path.splitext(table_id)[0]
        return table_id

    def _video_is_correctly_labelled_in_table(self, video_path: str) -> bool:
        table_id = self._get_video_id_in_table(video_path)
        attachment_view = self.annotations_table[
            self.annotations_table["attachment_id"] == table_id
        ]
        return len(attachment_view) == 1

    def _prepare_data(self) -> List[Dict]:
        video_paths: List[str]
        if self.is_train:
            video_paths = get_all_files_paths_in_dir(self.slovo_ds_strture.train_folder)
        else:
            video_paths = get_all_files_paths_in_dir(self.slovo_ds_strture.test_folder)
        samples_infos: List[Dict] = []
        for video_path in tqdm(video_paths, desc="Reading samples info", leave=False):
            video_id = self._get_video_id_in_table(video_path)
            if not self._video_is_correctly_labelled_in_table(video_path):
                continue

            sample_info = {"video_path": video_path}
            if not self.read_mediapipe_keypoints:
                samples_infos.append(sample_info)
                continue

            # read keypoints
            if video_id not in self.mediapipe_keypoints:
                continue
            sample_info["mediapipe_keypoints"] = self.mediapipe_keypoints[video_id]
            samples_infos.append(sample_info)
        return samples_infos

    def read_slovo_ds_sample(self, sample_info: Dict) -> SlovoDatasetSample:
        video_path = sample_info["video_path"]
        table_id = self._get_video_id_in_table(video_path)
        attachment_view = self.annotations_table[
            self.annotations_table["attachment_id"] == table_id
        ]
        if len(attachment_view) != 1:
            raise ValueError(f"There is not unique label for video {video_path}")
        label = attachment_view["text"].values[0]
        label_data: LabelData = LabelData(
            label_name=label, label_id=self._label_name2id[label]
        )
        mp_keypoints: Optional[Dict] = None
        if self.read_mediapipe_keypoints:
            mp_keypoints = sample_info["mediapipe_keypoints"]
        slovo_ds_sample = SlovoDatasetSample(
            video_path=video_path,
            label_data=label_data,
            mediapipe_keypoints=mp_keypoints,
        )
        return slovo_ds_sample

    def __len__(self) -> int:
        return len(self.samples_data)

    def __getitem__(self, index) -> SlovoDsStructure:
        sample_info: SlovoDatasetSample = self.read_slovo_ds_sample(
            self.samples_data[index]
        )
        return sample_info
