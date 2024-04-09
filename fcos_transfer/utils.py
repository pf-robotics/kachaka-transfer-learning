import math
import torch


def generate_location(width: int, height: int, scale: float) -> torch.Tensor:
    stride = int(1.0 / scale)
    shift_y, shift_x = torch.meshgrid(
        torch.arange(
        0,
        math.ceil(height * scale) * stride,
        step=stride,
        dtype=torch.float32,
    ),
        torch.arange(
        0,
        math.ceil(width * scale) / scale,
        step=stride,
        dtype=torch.float32,
    ))
    locations = torch.stack((shift_y.flatten(),
                             shift_x.flatten()),
                            dim=1) + stride // 2
    return locations
