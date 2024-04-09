#!/usr/bin/env python3
import click
import torch

from fcos_transfer.predictor import Predictor


@click.command()
@click.option("--class-num", type=int, default=1)
@click.option("--checkpoint", type=click.Path(exists=True, file_okay=True), required=True)
@click.option("--out", type=click.Path(exists=False, file_okay=True), required=True)
def main(class_num, checkpoint, out):
    predictor = Predictor(class_num, torch.load(checkpoint))
    predictor.eval()
    dummy_input = {"x":
                   {"p4": torch.randn(1, 96, 14, 25),
                    "p5": torch.randn(1, 96, 7, 13),
                    "p6": torch.randn(1, 96, 4, 7)}}
    torch_out = predictor(dummy_input["x"])
    torch.onnx.export(predictor, dummy_input, out,
                      input_names=["p4", "p5", "p6"],
                      output_names=["bboxes", "labels", "scores"])

if __name__ == "__main__":
    main()
