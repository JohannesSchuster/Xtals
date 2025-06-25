import torch
from .jobs import RenderOp


class RenderQueue:
    def __init__(self):
        self.ops: list[RenderOp] = []

    def submit(self, op: RenderOp) -> "RenderQueue":
        self.ops.append(op)
        return self

    def clear(self):
        self.ops.clear()

    def __iter__(self):
        return iter(self.ops)

    def __len__(self):
        return len(self.ops)


class Renderer:
    def __init__(self, device: torch.device = torch.device("cpu")):
        self.device = device
        self.last_result: torch.Tensor | None = None

    def begin(self) -> RenderQueue:
        return RenderQueue()

    def render(self, img: torch.Tensor, queue: RenderQueue) -> torch.Tensor:
        """
        Apply a queue of rendering operations to the input image.
        Args:
            img: torch.Tensor of shape (C, H, W)
            queue: RenderQueue containing RenderOp steps

        Returns:
            Processed image (torch.Tensor)
        """
        if img.device != self.device:
            img = img.to(self.device)

        for op in queue:
            img = op.apply(img)

        self.last_result = img.detach()
        return img