import os
import numpy as np
from skimage.io import MultiImage

class ImageHandler:
    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.images = MultiImage(filepath)[0]
        self.frames = [np.array(img) for img in self.images]
        self.current_frame = 0

    def get_frame(self, index=None):
        if index is None:
            index = self.current_frame
        return self.frames[index]

    def frame_count(self):
        return len(self.frames)

    def set_frame(self, index):
        self.current_frame = np.clip(index, 0, self.frame_count() - 1)
