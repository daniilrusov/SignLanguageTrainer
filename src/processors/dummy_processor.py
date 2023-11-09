from .processor import Processor

class DummyProcessor(Processor):
    def __init__(self):
        pass

    def __call__(self, video_path):
        return "А"
