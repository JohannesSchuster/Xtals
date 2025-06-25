import os
import tifffile as tf

class BaseImageHandler:
    def __init__(self):
        self.current_frame = 0
        self.frames = []

    def get_frame(self, index=None):
        if index is None:
            index = self.current_frame
        return self.frames[index]

    def frame_count(self):
        return len(self.frames)

    def set_frame(self, index):
        index = min(max(0, index), self.frame_count() - 1)
        self.current_frame = index

class FileImageHandler(BaseImageHandler):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.name = os.path.basename(filepath)

    def load(self):
        with tf.imread(self.filepath) as image:
            self.frames = [frame for frame in image]

    