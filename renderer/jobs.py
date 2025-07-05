import torch
from .renderer import RenderOp as __RenderOpBase

class Upload(__RenderOpBase):
    def __init__(self, slot: int, img: torch.Tensor):
        self.slot = slot
        self.img = img

    def execute(self, slot: int, slots: dict[int, torch.Tensor]) -> torch.Tensor:
        slots[self.slot] = self.img.clone()
        return slots[self.slot]


class CreateFlat(__RenderOpBase):
    def __init__(self, slot: int, size: tuple[int, int], color: tuple[int, int, int] | int):
        self.slot = slot
        self.size = size
        self.color = color

    def execute(self, slot: int, slots: dict[int, torch.Tensor]) -> torch.Tensor:
        # Create a C x H x W tensor filled with color
        c = 3
        w, h = self.size
        if isinstance(self.color, int): c = 1 
        canvas = torch.zeros((c, h, w), dtype=torch.float32, device=next(iter(slots.values())).device if slots else 'cpu')
        for i, col in enumerate(self.color):
            canvas[i].fill_(col / 255.0)
        slots[self.slot] = canvas
        return slots[self.slot]


class Select(__RenderOpBase):
    def __init__(self, slot: int):
        self.slot = slot

    def execute(self, slot: int, slots: dict[int, torch.Tensor]) -> torch.Tensor:
        # Select does nothing but can be a placeholder or ensure slot exists
        if self.slot not in slots:
            raise RuntimeError(f"Slot {self.slot} not available in Select op")
        return slots[self.slot]


class Crop(__RenderOpBase):
    def __init__(self, offset: tuple[int, int], size: tuple[int, int]):
        self.offset = offset  # (x,y)
        self.size = size      # (w,h)

    def execute(self, slot: int, slots: dict[int, torch.Tensor]) -> torch.Tensor:
        # Crop the currently selected slot (slot 0 assumed for simplicity)
        if slot not in slots:
            raise RuntimeError("Slot 0 missing for Crop")
        img = slots[slot]
        x, y = self.offset
        w, h = self.size
        _, H, W = img.shape
        x = max(0, min(x, W-1))
        y = max(0, min(y, H-1))
        w = min(w, W - x)
        h = min(h, H - y)
        cropped = img[:, y:y+h, x:x+w]
        slots[slot] = cropped
        return cropped


class Resize(__RenderOpBase):
    def __init__(self, new_size: tuple[int, int], mode: str = "bilinear"):
        self.new_size = new_size  # (w,h)
        self.mode = mode

    def execute(self, slot: int, slots: dict[int, torch.Tensor]) -> torch.Tensor:
        if slot not in slots:
            raise RuntimeError("Slot 0 missing for Resize")
        img = slots[slot]
        # Resize expects (N,C,H,W), add batch dim
        c, h, w = img.shape
        img_batched = img.unsqueeze(0)
        new_w, new_h = self.new_size
        resized = torch.nn.functional.interpolate(img_batched, size=(new_h, new_w), mode=self.mode, align_corners=False)
        resized = resized.squeeze(0)
        slots[slot] = resized
        return resized


class Blend(__RenderOpBase):
    def __init__(self, slots: list[int], offsets: list[tuple[int, int]], mode: str = "override"):
        self.slots = slots
        self.offsets = offsets
        self.mode = mode

    def execute(self, slot: int, slots: dict[int, torch.Tensor]) -> torch.Tensor:
        base_slot = self.slots[0]
        base_img = slots[base_slot]
        c_bg, h_bg, w_bg = base_img.shape
        result = base_img.clone()

        for slot_idx, offset in zip(self.slots[1:], self.offsets[1:]):
            if slot_idx not in slots:
                raise RuntimeError(f"Slot {slot_idx} missing for Blend")
            overlay = slots[slot_idx]
            c_ov, h_ov, w_ov = overlay.shape

            # Match channel dimensions
            if c_ov != c_bg:
                if c_ov == 1 and c_bg == 3:
                    overlay = overlay.expand(3, h_ov, w_ov).clone()
                elif c_ov == 3 and c_bg == 1:
                    result = result.expand(3, h_bg, w_bg).clone()
                    base_img = result  # Also update base reference
                    c_bg = 3  # Update channel count
                else:
                    raise RuntimeError(f"Incompatible channel sizes: base={c_bg}, overlay={c_ov}")

            oy, ox = offset
            paste_h = min(h_ov, h_bg - oy)
            paste_w = min(w_ov, w_bg - ox)
            if paste_h <= 0 or paste_w <= 0:
                continue  # no overlap

            if self.mode == "override":
                result[:, oy:oy+paste_h, ox:ox+paste_w] = overlay[:, :paste_h, :paste_w]
            else:
                raise NotImplementedError(f"Blend mode {self.mode} not implemented")

        slots[base_slot] = result
        return result