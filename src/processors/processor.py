from abc import ABC, abstractmethod

class Processor(ABC):
    @abstractmethod
    def __call__(self, video_path: str) -> str:
        pass
