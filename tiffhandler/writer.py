import struct
import imagecodecs
import torch

class Writer:
    def __init__(self, filename: str, compress: bool = True):
        self.filename = filename
        self.compress = compress

    def write(self, tensor: torch.Tensor) -> None:
        if tensor.ndim == 4:
            # (F, C, H, W)
            for i, frame in enumerate(tensor):
                fname = self.filename.replace('.tif', f'_{i:04d}.tif')
                self._write_single(fname, frame)
        else:
            self._write_single(self.filename, tensor)

    def _write_single(self, filename: str, img: torch.Tensor) -> None:
        img = img.cpu().contiguous()

        if img.ndim == 2:
            # Grayscale
            height, width = img.shape
            data = img.numpy().tobytes()
            samples = 1
        elif img.ndim == 3:
            # RGB
            c, height, width = img.shape
            assert c in [1, 3], "Only 1 or 3 channels supported"
            samples = c
            data = img.permute(1, 2, 0).numpy().tobytes()
        else:
            raise ValueError("Tensor shape must be (H, W) or (C, H, W)")

        if self.compress:
            encoded = imagecodecs.lzw_encode(data)
            compression = 5  # LZW
            strip_bytes = len(encoded)
        else:
            encoded = data
            compression = 1  # None
            strip_bytes = len(data)

        with open(filename, 'wb') as f:
            # Header: Little endian, TIFF magic number, offset to IFD
            f.write(b'II')  # Little endian
            f.write(struct.pack('<H', 42))  # Magic
            f.write(struct.pack('<I', 8))  # Offset to IFD

            # Image data starts at offset 8 + IFD size (~182 bytes)
            image_offset = 8 + 2 + 12 * 9 + 4
            f.write(encoded)

            # IFD
            f.write(struct.pack('<H', 9))  # number of entries

            def tag(t, type_, count, value):
                return struct.pack('<HHII', t, type_, count, value)

            f.write(tag(256, 4, 1, width))               # ImageWidth
            f.write(tag(257, 4, 1, height))              # ImageLength
            f.write(tag(258, 3, 1, 8))                   # BitsPerSample
            f.write(tag(259, 3, 1, compression))         # Compression
            f.write(tag(273, 4, 1, image_offset))        # StripOffsets
            f.write(tag(277, 3, 1, samples))             # SamplesPerPixel
            f.write(tag(278, 4, 1, height))              # RowsPerStrip
            f.write(tag(279, 4, 1, strip_bytes))         # StripByteCounts
            f.write(tag(284, 3, 1, 1))                   # PlanarConfiguration

            f.write(struct.pack('<I', 0))  # End of IFD chain