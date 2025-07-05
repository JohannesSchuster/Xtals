from typing import BinaryIO
import torch
import imagecodecs

TIFF_DTYPES = {
    1: (1, torch.uint8),     # BYTE
    2: (1, torch.uint8),     # ASCII (we treat as byte)
    3: (2, torch.uint16),    # SHORT
    4: (4, torch.uint32),    # LONG
    5: (8, torch.float32),   # RATIONAL (not handled here)
    6: (1, torch.int8),      # SBYTE
    7: (1, torch.uint8),     # UNDEFINED
    8: (2, torch.int16),     # SSHORT
    9: (4, torch.int32),     # SLONG
    11: (4, torch.float32),  # FLOAT
    12: (8, torch.float64),  # DOUBLE
}

class Reader:
    def __init__(self, filename: str, device: str = 'cpu') -> None:
        self.filename = filename
        self.device = device

    def get_all_ifd_offsets(self) -> list[int]:
        offsets = []
        with open(self.filename, 'rb') as f:
            endian = '<' if f.read(2) == b'II' else '>'
            f.read(2)  # skip magic
            next_ifd = int.from_bytes(f.read(4), endian)
            while next_ifd != 0:
                offsets.append(next_ifd)
                f.seek(next_ifd)
                num_tags = int.from_bytes(f.read(2), endian)
                f.seek(12 * num_tags, 1)
                next_ifd = int.from_bytes(f.read(4), endian)
        return offsets

    def read_frame_at_ifd(self, offset: int) -> torch.Tensor:
        with open(self.filename, 'rb') as f:
            f.seek(0)
            endian = '<' if f.read(2) == b'II' else '>'
            f.read(2)
            f.seek(offset)
            tags, _ = self._parse_ifd(f, endian)

            width = tags[256]
            height = tags[257]
            bits_per_sample = tags.get(258, 8)
            samples_per_pixel = tags.get(277, 1)
            photometric = tags.get(262, 1)
            compression = tags.get(259, 1)

            dtype_info = TIFF_DTYPES.get(11 if bits_per_sample == 32 else 1)
            byte_size, dtype = dtype_info

            offsets = tags[273]
            counts = tags[279]
            if isinstance(offsets, int):
                offsets = [offsets]
            if isinstance(counts, int):
                counts = [counts]

            f_data = bytearray()
            for off, cnt in zip(offsets, counts):
                f.seek(off)
                f_data.extend(f.read(cnt))

            if compression == 5:  # LZW
                raw = imagecodecs.lzw_decode(bytes(f_data))
            elif compression == 1:
                raw = bytes(f_data)
            else:
                raise NotImplementedError(f"Compression {compression} not supported")

            tensor = torch.frombuffer(raw, dtype=dtype).clone()

            shape = (height, width) if samples_per_pixel == 1 else (height, width, samples_per_pixel)
            tensor = tensor.reshape(shape)
            if samples_per_pixel > 1:
                tensor = tensor.permute(2, 0, 1)  # to C, H, W

            return tensor.to(self.device)

    def _parse_ifd(self, f: BinaryIO, endian: str) -> tuple[dict[int, any], str]:
        tags: dict[int, any] = {}
        num_tags = int.from_bytes(f.read(2), endian)
        for _ in range(num_tags):
            tag_bytes = f.read(12)
            tag = int.from_bytes(tag_bytes[0:2], endian)
            dtype = int.from_bytes(tag_bytes[2:4], endian)
            count = int.from_bytes(tag_bytes[4:8], endian)
            value_offset = tag_bytes[8:12]

            size_per_item, _ = TIFF_DTYPES.get(dtype, (1, torch.uint8))
            total_size = size_per_item * count

            if total_size <= 4:
                value = int.from_bytes(value_offset[:total_size], endian)
            else:
                offset = int.from_bytes(value_offset, endian)
                pos = f.tell()
                f.seek(offset)
                if dtype in (3, 8):  # SHORT or SSHORT
                    value = [int.from_bytes(f.read(2), endian) for _ in range(count)]
                elif dtype in (4, 9):  # LONG or SLONG
                    value = [int.from_bytes(f.read(4), endian) for _ in range(count)]
                else:
                    value = f.read(total_size)
                f.seek(pos)

                if isinstance(value, list[int]) and len(value) == 1:
                    value = value[0]
            tags[tag] = value
        next_ifd_offset = int.from_bytes(f.read(4), endian)
        tags['next_ifd'] = next_ifd_offset
        return tags, endian