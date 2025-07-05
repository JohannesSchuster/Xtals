from abc import ABC, abstractmethod
from typing import Optional
import torch

from .reader import Reader

class FrameHandler(ABC):
    def __init__(self) -> None:
        self._current_frame: Optional[int] = None

    def set_frame(self, index: int) -> torch.Tensor:
        self._current_frame = index
        return self.get_frame(index)

    def current_frame(self) -> Optional[int]:
        return self._current_frame

    @abstractmethod
    def get_frame(self, index: int) -> torch.Tensor:
        ...

    @abstractmethod
    def num_frames(self) -> int:
        ...

class MemoryFrameHandler(FrameHandler):
    def __init__(self, frames: list[torch.Tensor]) -> None:
        super().__init__()
        self._frames = frames

    def get_frame(self, index: int) -> torch.Tensor:
        return self._frames[index]

    def num_frames(self) -> int:
        return len(self._frames)


class FileFrameHandler(FrameHandler):
    def __init__(self, reader: Reader) -> None:
        super().__init__()
        self.reader = reader
        self.__frame_cache: dict[int, torch.Tensor] = {}
        self.__frame_offsets = self.reader.get_all_ifd_offsets()

    def get_frame(self, index: int) -> torch.Tensor:
        if index in self.__frame_cache:
            return self.__frame_cache[index]
        offset = self.__frame_offsets[index]
        torch.Tensor = self.reader.read_frame_at_ifd(offset)
        self.__frame_cache[index] = torch.Tensor
        return torch.Tensor

    def num_frames(self) -> int:
        return len(self.__frame_offsets)

    def preload(self) -> None:
        for i in range(self.num_frames()):
            self.get_frame(i)