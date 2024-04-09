import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional


class _Scale(nn.Module):
    def __init__(self, init_value: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([init_value]))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * self.scale


class Output(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        apply_scale: bool = False,
        apply_relu: bool = False,
        n_scale: Optional[int] = None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if apply_scale:
            assert n_scale is not None
            self.scales: Optional[nn.ModuleList] = nn.ModuleList(
                [_Scale(init_value=1.0) for _ in range(n_scale)]
            )
        else:
            self.scales = None
        if apply_relu:
            # Note that we use relu, as in the improved FCOS, instead of exp.
            self.relu: Optional[nn.Module] = nn.ReLU()
        else:
            self.relu = None

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return super().__call__(x)  # type: ignore

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        keys = list(x.keys())
        output_dict: Dict[str, torch.Tensor] = {k: self.conv(v) for k, v in x.items()}
        if self.scales is not None:
            for i, s in enumerate(self.scales):
                output_dict[keys[i]] = s(output_dict[keys[i]])
        if self.relu is not None:
            for k, v in output_dict.items():
                output_dict[k] = self.relu(v)
        return output_dict


class Tower(nn.Module):
    def __init__(self,
                 in_channels: int,
                 n_conv: int = 4,
                 n_group: Optional[int] = 32):
        super().__init__()
        tower: List[nn.Module] = []
        for _ in range(n_conv):
            tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )
            if n_group is None:
                tower.append(nn.BatchNorm2d(in_channels))
            else:
                tower.append(nn.GroupNorm(n_group, in_channels))
            tower.append(nn.ReLU())
        self.tower = nn.Sequential(*tower)

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return super().__call__(x)  # type: ignore

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        keys = list(x.keys())
        outputs: Dict[str, torch.Tensor] = x
        for module in self.tower:
            outputs = {k: module(v) for k, v in outputs.items()}
        return dict([(key, outputs[key]) for key in keys])


class DetHead(nn.Module):
    def __init__(
            self,
            n_fg_class: int,
            n_scale: int,
            use_bn: bool = False,
            in_channels: int = 96,
            n_tower_conv: int = 4
    ) -> None:
        super().__init__()
        n_score_tower_conv = n_tower_conv
        n_loc_tower_conv = n_tower_conv
        n_group = 32 if not use_bn else None

        self.score_tower = Tower(
            in_channels, n_score_tower_conv, n_group=n_group)

        self.bbox_tower = Tower(
            in_channels, n_loc_tower_conv, n_group=n_group)

        self.score_output = Output(in_channels, n_fg_class + 1)
        self.loc_output = Output(in_channels, 4,
                                 apply_scale=True,
                                 apply_relu=True,
                                 n_scale=n_scale)
        self.centerness_output = Output(in_channels, 1)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                torch.nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)

        # initialize the bias for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        assert self.score_output.conv.bias is not None
        torch.nn.init.constant_(self.score_output.conv.bias, bias_value)

    def forward(self, x: Dict[str, torch.Tensor]):
        score_towers = self.score_tower(x)
        bbox_towers = self.bbox_tower(x)
        scores = self.score_output(score_towers)
        locs = self.loc_output(bbox_towers)
        centernesses = self.centerness_output(bbox_towers)
        return (scores, locs, centernesses)
