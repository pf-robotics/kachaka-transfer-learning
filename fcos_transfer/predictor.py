import torch
import torch.nn as nn
from torchvision.ops.boxes import (  # type: ignore[import-untyped]
    batched_nms,
    clip_boxes_to_image,
    nms,
    remove_small_boxes,
)
from typing import Dict, List, NamedTuple, Tuple

from fcos_transfer.model import DetHead
from fcos_transfer.utils import generate_location


class DetectionOutputs(NamedTuple):
    locs: torch.Tensor
    scores: torch.Tensor
    centernesses: torch.Tensor


class FCOSOutput(NamedTuple):
    score: torch.Tensor
    bbox: torch.Tensor


class DetPrediction(NamedTuple):
    keep: torch.Tensor
    bbox: torch.Tensor
    label: torch.Tensor
    score: torch.Tensor


def adjust_score_by_centerness(
    score: torch.Tensor, centerness: torch.Tensor
) -> torch.Tensor:
    return torch.sqrt(score * centerness)


def location_to_bbox(
    width: int, height: int, scale: float, location: torch.Tensor
) -> torch.Tensor:
    loc = generate_location(width, height, scale)
    return torch.stack(
        [
            loc[:, 0] - location[:, 0],
            loc[:, 1] - location[:, 1],
            loc[:, 0] + location[:, 2],
            loc[:, 1] + location[:, 3],
        ],
        dim=1,
    )


def per_scale(
    det_out: DetectionOutputs,
    width: int,
    height: int,
    scale: float,
) -> List[FCOSOutput]:
    scores = adjust_score_by_centerness(det_out.scores, det_out.centernesses)
    N, c = scores.shape[:2]
    scores = scores.reshape(N, c, -1).permute(0, 2, 1)
    locs = det_out.locs
    locs = locs.reshape(N, 4, -1).permute(0, 2, 1)
    return [
        FCOSOutput(score=score, bbox=location_to_bbox(width, height, scale, loc))
        for loc, score in zip(locs, scores)
    ]


@torch.jit.script
def topk(score: torch.Tensor, k: int) -> torch.Tensor:
    score = score.reshape((-1,))
    # Notes: aten:topk accept integer value as k.
    # thus we have to use TorchScript to make k dynamic.
    if score.shape[0] < k:
        k = score.shape[0]
    _, index = score.topk(k)
    return index  # type: ignore


def calc_det_prediction(
    size: Tuple[int, int],
    bbox: torch.Tensor,
    score: torch.Tensor,
    score_thresh: float,
    iou_threshold: float,
    k: int,
    class_agnostic_nms: bool,
) -> DetPrediction:
    keep = torch.arange(score.shape[0])

    # flatten classwise-score and classwise-loc
    score = score[:, 1:]  # first index is for bg
    nz = torch.nonzero(score >= score_thresh)
    index = nz[:, 0]
    label = nz[:, 1]
    score = score[index, label]
    bbox = bbox[index]
    keep = keep[index]

    _keep = topk(score, k)
    score, label, bbox, keep = score[_keep], label[_keep], bbox[_keep], keep[_keep]

    if class_agnostic_nms:
        _keep = nms(bbox, score, iou_threshold=iou_threshold)
    else:
        _keep = batched_nms(bbox, score, label, iou_threshold=iou_threshold)
    bbox, label, score, keep = bbox[_keep], label[_keep], score[_keep], keep[_keep]

    bbox = clip_boxes_to_image(bbox, size[::-1])  # torchvision uses xy format
    _keep = remove_small_boxes(bbox, min_size=1e-5)
    return DetPrediction(
        keep=keep[_keep], bbox=bbox[_keep], label=label[_keep], score=score[_keep]
    )

class Prediction(NamedTuple):
    bbox: torch.Tensor
    label: torch.Tensor
    score: torch.Tensor

class Predictor(nn.Module):
    def __init__(self, class_num = 1, checkpoint_data = None) -> None:
        super().__init__()
        self.training = False
        self.scales = [0.0625, 0.03125, 0.015625]
        self.det_head = DetHead(
            class_num, len(self.scales),
            n_tower_conv = 1)
        if checkpoint_data:
            self.det_head.load_state_dict(checkpoint_data)

    def __call__(self, x: Dict[str, torch.Tensor]) -> Prediction:
        out: Prediction = super().__call__(x)
        return out

    def forward(self, x: Dict[str, torch.Tensor]) -> Prediction:
        assert not self.training
        with torch.no_grad():
            return self._forward(x)

    def _forward(self, x: Dict[str, torch.Tensor]) -> Prediction:
        score_head_out, loc_head_out, centerness_head_out = self.det_head(x)

        det_out: Dict[str, DetectionOutputs] = dict()
        det_out["p4"] = DetectionOutputs(
            locs=loc_head_out["p4"] / self.scales[0],
            scores=torch.sigmoid(score_head_out["p4"]),
            centernesses=torch.sigmoid(centerness_head_out["p4"])
        )
        det_out["p5"] = DetectionOutputs(
            locs=loc_head_out["p5"] / self.scales[1],
            scores=torch.sigmoid(score_head_out["p5"]),
            centernesses=torch.sigmoid(centerness_head_out["p5"])
        )
        det_out["p6"] = DetectionOutputs(
            locs=loc_head_out["p6"] / self.scales[2],
            scores=torch.sigmoid(score_head_out["p6"]),
            centernesses=torch.sigmoid(centerness_head_out["p6"])
        )

        model_h = 224
        model_w = 398
        resized_img = torch.zeros((1, model_h, model_w), dtype=torch.float32)
        out_per_scale: List[FCOSOutput] = []
        for i, key in enumerate(det_out.keys()):
            out = det_out[key]
            scale = self.scales[i]
            p = per_scale(out, model_w, model_h, scale)
            out_per_scale.append(p[0])
        score = torch.cat([x.score for x in out_per_scale], dim=0)
        bbox = torch.cat([x.bbox for x in out_per_scale], dim=0)

        keep, bboxes, labels, scores = calc_det_prediction(
            (model_h, model_w),
            bbox=bbox,
            score=score,
            score_thresh=0.05,
            iou_threshold=0.6,
            k=1000,
            class_agnostic_nms=False,
        )
        return Prediction(bboxes, labels, scores)
