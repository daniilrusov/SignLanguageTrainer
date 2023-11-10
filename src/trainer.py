import os

from .generators.word_generator import WordGenerator
from .processors.dummy_processor import DummyProcessor
from .processors.mmaction_tsn_processor import MmactionTSNProcessor
from .data_access_managers.filesystem_manager import FileSystemManager



class Trainer:
    def __init__(self, upload_config, word_csv_path):
        self.data_manager = FileSystemManager(**upload_config)
        self.generator = WordGenerator(word_csv_path)
        
        tsn_assets_root = os.path.join(
            "assets",
            "tsn_inference_example_data",
        )
        cfg_path: str = os.path.join(
            tsn_assets_root,
            "tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_animals.py",
        )
        label2id_mapping_path: str = os.path.join(tsn_assets_root, "label2id_mapping.json")
        weights_path: str = os.path.join(tsn_assets_root, "epoch_58.pth")

        self.processors = {'Алфавит': DummyProcessor(),
                           'Животные': MmactionTSNProcessor(
                            config_path=cfg_path,
                            checkpoint_path=weights_path,
                            label2id_mapping_path=label2id_mapping_path,
                            device="cpu",
                        )
                        }

    def submit(self, category, form_file):
        path = self.data_manager.save(form_file)
        processor = self.processors[category]
        label = processor(path)
        self.data_manager.delete(path)
        return label

    def generate(self, category=None, word=None):
        if word:
            task = self.generator.get_word(word)
        elif category:
            task = self.generator.get_category(category)
        else:
            task = self.generator.get_random()
        return task

    def get_categories(self):
        return self.generator.get_categories()
    
    def get_words(self):
        return self.generator.get_words()
