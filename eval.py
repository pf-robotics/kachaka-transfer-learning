#!/usr/bin/env python3
import click
import os

import cv2
import numpy as np
import onnxruntime  # type: ignore[import-untyped]
import pickle

import torch
from tqdm import tqdm  # type: ignore[import-untyped]
from typing import Dict, List, NamedTuple, Tuple

from fcos_transfer.predictor import Predictor

# for up to 15-classes
_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0),
           (0, 255, 255), (255, 255, 0), (255, 0, 255),
           (0, 128, 255), (0, 255, 128), (128, 255, 0),
           (255, 128, 0), (128, 0, 255), (255, 0, 128),
           (0, 128, 128), (128, 128, 0), (128, 0, 128)]


def inference(class_num, checkpoint, onnx, indir, outdir, threshold):
    os.makedirs(outdir, exist_ok=True)

    if checkpoint:
        checkpoint_data = torch.load(checkpoint)
        predictor = Predictor(class_num, checkpoint_data)

    if onnx:
        pred_session = onnxruntime.InferenceSession(onnx)
        pred_input_name_list = [out.name for out in pred_session.get_inputs()]
        pred_output_name_list = [out.name for out in pred_session.get_outputs()]

    imgfiles = [
        os.path.join(indir, f)
        for f in sorted(os.listdir(indir))
        if f.endswith(".jpg") or f.endswith(".png")
    ]
    for imgpath in tqdm(imgfiles):
        img = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
        if img is None or len(img.shape) != 3:
            continue
        pkl_filepath = f"{'.'.join(imgpath.split('.')[:-1])}.pkl"
        with open(pkl_filepath, 'rb') as f:
            data = pickle.load(f)
        p4_out = data['p4'].astype(np.float32)
        p5_out = data['p5'].astype(np.float32)
        p6_out = data['p6'].astype(np.float32)

        if onnx:
            inputs = {pred_input_name_list[0]: p4_out,
                      pred_input_name_list[1]: p5_out,
                      pred_input_name_list[2]: p6_out}
            (pred_bboxes,
             pred_labels,
             pred_scores) = pred_session.run(pred_output_name_list,
                                             inputs)
        elif checkpoint:
            result = predictor({"p4": torch.from_numpy(p4_out),
                                "p5": torch.from_numpy(p5_out),
                                "p6": torch.from_numpy(p6_out)})
            pred_bboxes = result.bbox
            pred_labels = result.label
            pred_scores = result.score

        model_h = 224
        model_w = 398
        for i, (label, score, bbox) in enumerate(zip(pred_labels, pred_scores, pred_bboxes)):
            if score < threshold:
                continue
            crop_y1 = int(bbox[0] * (img.shape[0] / model_h))
            crop_x1 = int(bbox[1] * (img.shape[1] / model_w))
            crop_y2 = int(bbox[2] * (img.shape[0] / model_h))
            crop_x2 = int(bbox[3] * (img.shape[1] / model_w))
            cv2.rectangle(
                img,
                (crop_x1, crop_y1),
                (crop_x2, crop_y2),
                _COLORS[label],
                thickness=3,
            )
            cv2.putText(
                img,
                org=(crop_x1, crop_y1 - 10),
                text=f"{label}_class: {score:0.3f}",
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=_COLORS[label],
                thickness=2,
                lineType=cv2.LINE_4)
        cv2.imwrite(os.path.join(outdir, os.path.basename(imgpath)), img)


@click.command()
@click.option("--class-num", type=int, default=1)
@click.option("--checkpoint", type=click.Path(exists=True, file_okay=True))
@click.option("--onnx", type=click.Path(exists=True, file_okay=True))
@click.option("--indir", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--outdir", type=click.Path(exists=False, file_okay=False), required=True)
@click.option("--threshold", type=float, default=0.65)
def main(class_num, checkpoint, onnx, indir, outdir, threshold):
    if checkpoint is None and onnx is None:
        print ("need to specify either checkpoint or onnx.")
        return
    if checkpoint and onnx:
        print ("both checkpoint and onnx are specified, should be either one.")
        return
    inference(class_num, checkpoint, onnx, indir, outdir, threshold)


if __name__ == "__main__":
    main()
