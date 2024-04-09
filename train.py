#!/usr/bin/env python3
import click
import onnxruntime  # type: ignore[import-untyped]
import os
import torch

from fcos_transfer.dataset import CustomDataset
from fcos_transfer.model import DetHead
from fcos_transfer.loss import (
    calc_centerness_loss,
    calc_loc_loss,
    calc_score_loss
)


def train(inputs, targets, class_num, det_head):
    (p4_outs, p5_outs, p6_outs) = inputs
    (score_assigns_list,
     locs_assigns_list,
     centerness_assigns_list) = targets
    with torch.no_grad():
        loc_gt_labels = locs_assigns_list.view(-1, 4)
        centerness_gt_labels = centerness_assigns_list.view(-1, 1)
        score_labels = [torch.where(score_assigns_list > 0, 0, 1).view(-1, 1)]
        for i in range(class_num):
            score_labels.append(torch.where(score_assigns_list == i+1, 1, 0).view(-1, 1))
        score_gt_labels = torch.cat(score_labels).type(torch.float32)
        fg_labels = torch.where(score_assigns_list > 0, 1, 0).view(-1, 1)
    res_fg_scores = []
    res_bg_scores = []
    res_locs = []
    res_centernesses = []
    for (p4_out, p5_out, p6_out) in zip(p4_outs, p5_outs, p6_outs):
        res_score, res_loc, res_centerness = det_head({"p4": p4_out.type(torch.FloatTensor),
                                                       "p5": p5_out.type(torch.FloatTensor),
                                                       "p6": p6_out.type(torch.FloatTensor)})
        res_bg_scores.extend([res_score["p4"][0][0].view(-1, 1),
                              res_score["p5"][0][0].view(-1, 1),
                              res_score["p6"][0][0].view(-1, 1)])
        for i in range(class_num):
            res_fg_scores.extend([res_score["p4"][0][i+1].view(-1, 1),
                                  res_score["p5"][0][i+1].view(-1, 1),
                                  res_score["p6"][0][i+1].view(-1, 1)])
        res_locs.extend([res_loc["p4"].view(1, 4, -1),
                         res_loc["p5"].view(1, 4, -1),
                         res_loc["p6"].view(1, 4, -1)])
        res_centernesses.extend([res_centerness["p4"].view(-1, 1),
                                 res_centerness["p5"].view(-1, 1),
                                 res_centerness["p6"].view(-1, 1)])
    score_inputs = torch.cat(res_bg_scores + res_fg_scores)
    loc_inputs = torch.cat(res_locs, dim=2).permute(0, 2, 1)
    centerness_inputs = torch.cat(res_centernesses, dim=0)
    score_loss = calc_score_loss(score_inputs, score_gt_labels)
    loc_loss = calc_loc_loss(loc_inputs, loc_gt_labels,
                             centerness_gt_labels,
                             fg_labels)
    centerness_loss = calc_centerness_loss(centerness_inputs,
                                           centerness_gt_labels,
                                           fg_labels)
    total_loss = score_loss + loc_loss + centerness_loss
    print(f"total: {total_loss} (score: {score_loss}, loc: {loc_loss}, centerness: {centerness_loss})")
    total_loss.backward()

@click.command()
@click.option("--dataset", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--class-num", type=int, default=1)
@click.option("--epoch", type=int, default=300)
@click.option("--batch-size", type=int, default=1)
@click.option("--num-workers", type=int, default=1)
@click.option("--learning-rate", type=float, default=0.001)
@click.option("--momentum", type=float, default=0.9)
@click.option("--load-checkpoint-path", type=click.Path(exists=True, file_okay=True), default=None)
def main(dataset, class_num, epoch, batch_size, num_workers,
         learning_rate, momentum, load_checkpoint_path):
    scales = [0.0625, 0.03125, 0.015625]
    custom_dataset = CustomDataset(scales, dataset)
    trainloader = torch.utils.data.DataLoader(custom_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers)
    det_head = DetHead(class_num, len(scales), n_tower_conv=1)
    optimizer = torch.optim.SGD(det_head.parameters(),
                                lr=learning_rate, momentum=momentum)
    if load_checkpoint_path:
        det_head.load_state_dict(torch.load(load_checkpoint_path))
        optimizer.load_state_dict(torch.load(f'{load_checkpoint_path.replace(".pt", "_optimizer.pt")}'))
        prev_epoch_num = int(load_checkpoint_path.split('epoch')[-1].split('.pt')[0]) + 1
    else:
        prev_epoch_num = 0

    os.makedirs('checkpoint', exist_ok=True)

    for i in range(epoch):
        print (f"------ epoch num: {i + prev_epoch_num} ------")
        for inputs, targets in trainloader:
            optimizer.zero_grad()
            train(inputs, targets, class_num, det_head)
            optimizer.step()

        torch.save(optimizer.state_dict(),
                   f"checkpoint/epoch{i + prev_epoch_num}_optimizer.pt")
        torch.save(det_head.state_dict(),
                   f"checkpoint/epoch{i + prev_epoch_num}.pt")
    link_path = "latest_checkpoint.pt"
    if os.path.islink(link_path):
        os.unlink(link_path)
    os.symlink(f"checkpoint/epoch{i + prev_epoch_num}.pt", link_path)



if __name__ == "__main__":
    main()
