import os
from os.path import dirname, join
import json
from typing import Dict
from mmaction.structures.action_data_sample import ActionDataSample
from mmaction.apis import inference_recognizer, init_recognizer
import torch
from .processor import Processor


class MmactionTSNProcessor(Processor):
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        label2id_mapping_path: str,
        device: str = "cpu",
    ):
        self.model = init_recognizer(config_path, checkpoint_path, device=device)
        with open(label2id_mapping_path, "r", encoding="utf-8") as f:
            label2id_mapping: Dict[str, int] = json.load(f)
        self.id2label: Dict[int, str] = {
            id: label for label, id in label2id_mapping.items()
        }

    def __call__(self, video_path: str) -> str:
        result: ActionDataSample = inference_recognizer(self.model, video_path)
        prediction: torch.Tensor = result.pred_label
        prediction.to("cpu")
        predicted_class = int(prediction.item())
        predicted_word = self.id2label[predicted_class]
        return predicted_word


def inference_example():
    file_path = os.path.abspath(__file__)
    repo_root = dirname(dirname(dirname(file_path)))
    tsn_assets_root = join(
        repo_root,
        "assets",
        "tsn_animals",
    )
    cfg_path: str = join(
        tsn_assets_root,
        "tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_animals.py",
    )
    label2id_mapping_path: str = join(tsn_assets_root, "label2id_mapping.json")
    weights_path: str = join(tsn_assets_root, "epoch_58.pth")

    processor: MmactionTSNProcessor = MmactionTSNProcessor(
        config_path=cfg_path,
        checkpoint_path=weights_path,
        label2id_mapping_path=label2id_mapping_path,
        device="cpu",
    )

    video = join(tsn_assets_root, "00ea6015-99d6-45fe-8124-33461a27f058.mp4")
    output = processor(video)
    print(f"predicted word = {output}")


if __name__ == "__main__":
    inference_example()
