import torch
from torch import nn
import torchvision # type: ignore[import-untyped]


def giou_loss(pred: torch.Tensor,
              target: torch.Tensor, eps: float = 1e-5):
    p_y1, p_x1, p_y2, p_x2 = torch.chunk(pred, 4, dim=1)
    g_y1, g_x1, g_y2, g_x2 = torch.chunk(target, 4, dim=1)

    area_c = torch.clamp_min((torch.max(p_x1, g_x1) + torch.max(p_x2, g_x2)) *
                             (torch.max(p_y1, g_y1) + torch.max(p_y2, g_y2)),
                             eps)
    area_intersect = ((torch.min(p_x1, g_x1) + torch.min(p_x2, g_x2)) *
                      (torch.min(p_y1, g_y1) + torch.min(p_y2, g_y2)))
    area_union = ((g_x1 + g_x2) * (g_y1 + g_y2) +
                  (p_x1 + p_x2) * (p_y1 + p_y2) -
                  area_intersect)
    ious = (area_intersect + 1.0) / (area_union + 1.0)
    gious = ious - (area_c - area_union) / area_c
    losses = 1 - gious
    return losses.flatten()

_centerness_loss_func = nn.BCEWithLogitsLoss(reduction="none")

def calc_score_loss(scores: torch.Tensor,
                    score_targets: torch.Tensor) -> float:
    return torchvision.ops.sigmoid_focal_loss(
        scores,
        score_targets,
        0.25, 2
    ).sum()

def calc_loc_loss(
    locs: torch.Tensor,
    loc_targets: torch.Tensor,
    centerness_targets: torch.Tensor,
    indices: torch.Tensor,
) -> float:
    loc_loss = giou_loss(locs[0, indices[:, 0] > 0, :],
                         loc_targets[indices[:, 0] > 0]) * centerness_targets[indices[:, 0] > 0]
    return loc_loss.sum() / torch.clamp_min(centerness_targets[indices[:, 0]].sum(), 1e-5)

def calc_centerness_loss(
    centernesses: torch.Tensor,
    centerness_targets: torch.Tensor,
    indices: torch.Tensor,
) -> float:
    centerness_loss = _centerness_loss_func(
        centernesses[indices[:, 0] > 0],
        centerness_targets[indices[:, 0] > 0])
    return centerness_loss.sum() / max(centerness_loss.shape[0], 1)
