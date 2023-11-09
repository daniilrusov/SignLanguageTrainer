from typing import List, Dict
from dataset.mmaction_views import (
    MMActionVideoDatasetDescription,
    spawn_txt_and_labels_mapping_for_mmaction_VideoDataset_init,
)
from os.path import dirname as up
from dataset.slovo_dataset import (
    SlovoDataset,
    SlovoDatasetInterface,
    SLOVO_ANNOTATIONS_TABLE_TYPE,
    SlovoDatasetSubset,
)
from dataset.utils import create_label_name2id_mappig
from dataset.slovo_ds_structure import SlovoDsStructure
from dataset.slovo_ds_sample import SlovoDatasetSample
from tqdm import tqdm
import pandas as pd
import os
from pprint import pprint

WS_DIR = up(up(__file__))
DATA_DIR = os.path.join(WS_DIR, "data")
SLOVO_DATASET_DIR = os.path.join(DATA_DIR, "slovo")


EMOTES_PACKAGE: List[str] = [
    "no_event",
    "весёлый",
    "злой",
    "расстроенный",
    "гнев",
    "печаль",
    "удивлён",
    "стыдно",
    "зависть",
    "презрение",
    "ревнивый",
]
TIME_PACKAGE: List[str] = [
    "no_event",
    "минута",
    "часы",
    "год",
    "день",
    "месяц",
    "вечер",
    "утро",
    "завтра",
    "сегодня",
]
FAMILY_PACKAGE: List[str] = [
    "no_event",
    "отец",
    "семья",
    "сын",
    "дочь",
    "жена",
    "брат",
    "сестра",
]
ANIMALS_PACKAGE: List[str] = [
    "no_event",
    "собака",
    "лошадь",
    "курица",
    "медведь",
    "козел",
    "волк",
    "бык",
    "коза",
    "свинья",
    "овца",
]

LABELS_PACKS_REGISTRY: Dict[str, List[str]] = {
    "emotions": EMOTES_PACKAGE,
    "time": TIME_PACKAGE,
    "family": FAMILY_PACKAGE,
    "animals": ANIMALS_PACKAGE,
}


def get_labels_subset(
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


def validate_labels_package(
    labels_pack: List[str], labels_table: SLOVO_ANNOTATIONS_TABLE_TYPE
) -> bool:
    """Tests that all labels are uniqie and present in labels table"""
    uniq_labels = list(set(labels_pack))
    if len(uniq_labels) != len(labels_pack):
        return False
    all_labels = labels_table["text"].to_list()
    for label in labels_pack:
        if label not in all_labels:
            print(f"wrong label = {label}")
            return False
    return True


def validate_all_packages():
    ds_structure: SlovoDsStructure = SlovoDsStructure(SLOVO_DATASET_DIR)
    table: SLOVO_ANNOTATIONS_TABLE_TYPE = pd.read_csv(
        ds_structure.annotations_path,
        on_bad_lines="skip",
        delimiter="\t",
    )
    for name, labels_pack in LABELS_PACKS_REGISTRY.items():
        print(name)
        print(validate_labels_package(labels_pack, table))


def prepare_data_for_MMAction_training_on_labels_package(
    output_dir: str, labels_subset: List[str]
):
    ds_structure: SlovoDsStructure = SlovoDsStructure(SLOVO_DATASET_DIR)
    table = pd.read_csv(
        ds_structure.annotations_path, on_bad_lines="skip", delimiter="\t"
    )
    mapping = create_label_name2id_mappig(table)
    train_dataset: SlovoDatasetInterface = SlovoDataset(
        is_train=True,
        slovo_ds_structure=ds_structure,
        read_mediapipe_keypoints=False,
        label_name2id=mapping,
    )
    val_dataset: SlovoDatasetInterface = SlovoDataset(
        is_train=False,
        slovo_ds_structure=ds_structure,
        read_mediapipe_keypoints=False,
        label_name2id=mapping,
    )
    train_subset_dataset = get_labels_subset(
        base_dataset=train_dataset, labels=labels_subset
    )
    val_subset_dataset = get_labels_subset(
        base_dataset=val_dataset, labels=labels_subset
    )

    mmaction_descr_test = MMActionVideoDatasetDescription(
        files_root=output_dir,
        train_data_root=ds_structure.train_folder,
        val_data_root=ds_structure.test_folder,
    )
    spawn_txt_and_labels_mapping_for_mmaction_VideoDataset_init(
        slovo_dataset=val_subset_dataset,
        mmaction_ds_descr=mmaction_descr_test,
        is_train=False,
    )
    spawn_txt_and_labels_mapping_for_mmaction_VideoDataset_init(
        slovo_dataset=train_subset_dataset,
        mmaction_ds_descr=mmaction_descr_test,
        is_train=True,
    )


if __name__ == "__main__":
    family_package_path: str = "E:\\ITMO\\CVProject\\mmaction_datasets\\family"
    prepare_data_for_MMAction_training_on_labels_package(
        family_package_path, labels_subset=LABELS_PACKS_REGISTRY["family"]
    )
