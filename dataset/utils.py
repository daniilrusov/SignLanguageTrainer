from typing import List, Dict
from dataset.slovo_dataset import SLOVO_ANNOTATIONS_TABLE_TYPE


def create_label_name2id_mappig(table: SLOVO_ANNOTATIONS_TABLE_TYPE) -> Dict[str, int]:
    all_labels = table["text"].to_list()
    uniq_labels: List[str] = list(set(all_labels))
    label2id_mapping = {label: id for id, label in enumerate(uniq_labels)}
    return label2id_mapping
