import cv2
import glob
import json
import math
import numpy as np
import pickle
from typing import Dict, NamedTuple, Union, Optional, List
import torch

from fcos_transfer.utils import generate_location


class SizeOfInterest(NamedTuple):
    min_size: float
    max_size: float

def generate_size_of_interests(
    img: torch.Tensor, scales: List[float], size_ratio: float
) -> List[SizeOfInterest]:
    h, w = img.shape[1:]
    size = math.sqrt(h * w)
    min_scale = min(scales)

    sizes_of_interest: List[SizeOfInterest] = []
    prev = -1.0
    for scale in scales:
        max_size = float(int(size * min_scale / scale * size_ratio))
        assert prev <= max_size
        sizes_of_interest.append(SizeOfInterest(prev, max_size))
        prev = max_size
    return sizes_of_interest

def compute_centerness_targets(loc_targets: torch.Tensor) -> torch.Tensor:
    top_bottom = loc_targets[:, :, [0, 2]]
    top_bottom_min = top_bottom.min(dim=-1)[0]
    top_bottom_max = top_bottom.max(dim=-1)[0]
    top_bottom_max = torch.max(top_bottom_max, torch.full_like(top_bottom_max, 1e-5))
    left_right = loc_targets[:, :, [1, 3]]
    left_right_min = left_right.min(dim=-1)[0]
    left_right_max = left_right.max(dim=-1)[0]
    left_right_max = torch.max(left_right_max, torch.full_like(left_right_max, 1e-5))
    centerness = (top_bottom_min / top_bottom_max) * (left_right_min / left_right_max)
    centerness = torch.max(centerness, torch.zeros_like(centerness))
    return torch.sqrt(centerness)

def get_sample_region(
    bboxes: torch.Tensor,
    location_ys: torch.Tensor,
    location_xs: torch.Tensor,
    radius: float,
    scale: float,
) -> torch.Tensor:
    # bboxes: (bsize, n_gt, 4)
    # location_ys: (n_sample)
    # location_xs: (n_sample)
    assert bboxes.shape[-1] == 4
    assert location_xs.shape[0] == location_ys.shape[0]

    bboxes = bboxes.float()
    center_x = (bboxes[..., 1] + bboxes[..., 3]) / 2  # (bsize, n_gt, 4)
    center_y = (bboxes[..., 0] + bboxes[..., 2]) / 2  # (bsize, n_gt, 4)

    stride = radius / scale
    xmin = center_x - stride
    ymin = center_y - stride
    xmax = center_x + stride
    ymax = center_y + stride

    # limit sample region in gt
    center_bbox = torch.stack(
        [
            torch.max(ymin, bboxes[:, :, 0]),
            torch.max(xmin, bboxes[:, :, 1]),
            torch.min(ymax, bboxes[:, :, 2]),
            torch.min(xmax, bboxes[:, :, 3]),
        ],
        dim=2,
    )
    left = location_xs[None, :, None] - center_bbox[:, None, :, 1]
    right = center_bbox[:, None, :, 3] - location_xs[None, :, None]
    top = location_ys[None, :, None] - center_bbox[:, None, :, 0]
    bottom = center_bbox[:, None, :, 2] - location_ys[None, :, None]
    center_bbox = torch.stack(
        (left, top, right, bottom), -1
    )  # (bsize, n_sample, n_gt, 4)
    inside_bbox_mask: torch.Tensor = center_bbox.min(-1)[0] > 0
    return inside_bbox_mask  # (bsize ,n_sample, n_gt)


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, scales, dirname):
        orig_h = 720
        orig_w = 1280
        model_h = 224
        model_w = 398
        self.input_list = []
        self.target_list = []
        category_list = []
        path = sorted(glob.glob(f'{dirname}/.exports/*.json'))[-1]
        with open(path, 'r') as f:
            datajson = json.loads(f.read())
        for im in datajson['images']:
            filename = im['path'].split('/')[-1]
            img = cv2.imread(f"{dirname}/{filename}",
                             cv2.IMREAD_UNCHANGED)
            img_input = cv2.resize(img, (orig_w, orig_h))
            img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
            img_input = img_input.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            img_input = img_input.astype(np.float32)
            with open(f"{dirname}/{'.'.join(filename.split('.')[:-1])}.pkl", 'rb') as f:
                gt = pickle.load(f)
            score_assigns_list = [None, None, None]
            locs_assigns_list = [None, None, None]
            centerness_assigns_list = [None, None, None]
            locs = [generate_location(model_w, model_h, scale) for scale in scales]
            for anno in datajson['annotations']:
                if anno['image_id'] != im['id']:
                    continue
                category_id = anno['category_id']
                if category_id not in category_list:
                    category_list.append(category_id)
                category_index = category_list.index(category_id) + 1
                x = anno['bbox'][0] * model_w / orig_w
                y = anno['bbox'][1] * model_h / orig_h
                w = anno['bbox'][2] * model_w / orig_w
                h = anno['bbox'][3] * model_h / orig_h
                cy = y + h / 2
                cx = x + w / 2
                bboxes = torch.Tensor([[[y, x, h+y, w+x]]])
                radius = 1.5
                area = (bboxes[..., 2] - bboxes[..., 0]) * (
                    bboxes[..., 3] - bboxes[..., 1]
                )  # (N, n_gt)
                print (anno)
                soi = generate_size_of_interests(img_input, scales, size_ratio=1.0)
                for i in range(3):
                    ys, xs = locs[i][:, 0], locs[i][:, 1]
                    # for loc
                    t = ys[None, :, None] - bboxes[:, None, :, 0]  # (N, n_sample, n_gt)
                    l = xs[None, :, None] - bboxes[:, None, :, 1]  # (N, n_sample, n_gt)
                    b = bboxes[:, None, :, 2] - ys[None, :, None]  # (N, n_sample, n_gt)
                    r = bboxes[:, None, :, 3] - xs[None, :, None]  # (N, n_sample, n_gt)
                    loc_targets_per_im = torch.stack(
                        [t, l, b, r], dim=-1
                    )  # (N, n_sample, n_gt, 4)
                    _, n_sample, n_gt = loc_targets_per_im.shape[:3]
                    max_loc_targets_per_im = loc_targets_per_im.max(dim=3)[0]  # (N, n_sample, n_gt)
                    is_cared_in_the_level = (
                        max_loc_targets_per_im >= soi[i].min_size
                    ) & (
                        max_loc_targets_per_im <= soi[i].max_size
                    )  # (N, n_sample, n_gt)

                    locs_assign = (loc_targets_per_im * scales[i])[:, :, 0, :]
                    if locs_assigns_list[i] is None:
                        locs_assigns_list[i] = locs_assign
                    else:
                        indices = locs_assigns_list[i].min(dim=2).values < 0
                        locs_assigns_list[i][indices] = locs_assign[indices]
                    centerness_assigns = compute_centerness_targets(locs_assign)
                    if centerness_assigns_list[i] is None:
                        centerness_assigns_list[i] = centerness_assigns
                    else:
                        centerness_assigns_list[i] = torch.max(centerness_assigns_list[i],
                                                               centerness_assigns)
                    is_in_boxes = get_sample_region(
                        bboxes,
                        ys,
                        xs,
                        radius,
                        scales[i],
                    )  # (N, n_sample, n_gt)
                    locations_to_gt_area = area[:, None, :].expand(
                        1, 1, 1
                    )  # (N, n_sample, n_gt)
                    inf = 1e10
                    locations_to_gt_area = torch.where(
                        is_cared_in_the_level,
                        locations_to_gt_area,
                        torch.full_like(locations_to_gt_area, fill_value=inf),
                    )
                    locations_to_gt_area = torch.where(
                        is_in_boxes,
                        locations_to_gt_area,
                        torch.full_like(locations_to_gt_area, fill_value=inf),
                    )
                    locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=2)
                    score_assign = locations_to_gt_inds + category_index
                    score_assign = torch.where(
                        locations_to_min_area == inf, torch.zeros_like(score_assign),
                        score_assign
                    )
                    if score_assigns_list[i] is None:
                        score_assigns_list[i] = score_assign
                    else:
                        score_assigns_list[i] = torch.where(score_assigns_list[i] > 0,
                                                            score_assigns_list[i], score_assign)
            self.input_list.append((gt['p4'], gt['p5'], gt['p6']))
            self.target_list.append((np.concatenate(score_assigns_list, axis=1),
                                     np.concatenate(locs_assigns_list, axis=1),
                                     np.concatenate(centerness_assigns_list, axis=1)))

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        return (self.input_list[idx], self.target_list[idx])
