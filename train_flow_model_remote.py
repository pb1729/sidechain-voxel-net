from __future__ import annotations

import argparse
import os
from pathlib import Path
import random

import numpy as np
import torch

from density_dataset import make_density_batch_loader
from density_fns import DensFn2
from train_flow_model import FlowConfig, FlowModel
from util import annotate_path


FULL_CHANNELS = [(64, 5), (128, 5), (256, 5), (512, 4)]
SANITY_CHANNELS = [(16, 2), (32, 2)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the flow model on a Lambda Cloud instance.")
    parser.add_argument("--dataset", default="/lambda/nfs/research/datasets/cath-cif")
    parser.add_argument("--output-dir", default="/home/ubuntu/cath_s40_flow_output")
    parser.add_argument("--save-name", default="flownet_remote.pt")
    parser.add_argument("--profile", choices=("sanity", "full"), default="full")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout-percent", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-autocast", action="store_true")
    return parser.parse_args()


def save_checkpoint(flowmodel: FlowModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(flowmodel.to_dict(), tmp_path)
    os.replace(tmp_path, path)


def latest_step(flowmodel: FlowModel) -> int:
    return max((i for values in flowmodel.history.values() for i, _value in values), default=-1) + 1


def make_config(args: argparse.Namespace, device: torch.device) -> FlowConfig:
    if args.profile == "full":
        output_channels = 24
        chan_l_list = FULL_CHANNELS
        blur_list = [2.0, 4.0, 6.0, 8.0]
    else:
        output_channels = 8
        chan_l_list = SANITY_CHANNELS
        blur_list = [2.0, 4.0]

    densfn = DensFn2(
        atom_gaussian_radius=1.5,
        output_channels=output_channels,
        device=str(device),
        conv_method="fft",
        fft_phase_batch_size=2,
    )
    return FlowConfig(
        batch=args.batch,
        densfn=densfn,
        chan_L_list=chan_l_list,
        blur_list=blur_list,
        autocast=not args.no_autocast,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_path = Path(args.dataset)
    if not dataset_path.is_dir():
        raise SystemExit(f"dataset directory does not exist: {dataset_path}")

    save_path = Path(args.output_dir) / args.save_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)
    if device.type == "cuda":
        print("cuda_device:", torch.cuda.get_device_name(0), flush=True)
    print("dataset:", dataset_path, flush=True)
    print("save_path:", save_path, flush=True)
    print("profile:", args.profile, flush=True)

    if args.resume and save_path.exists():
        print("resuming:", save_path, flush=True)
        flowmodel = FlowModel.from_dict(torch.load(save_path, map_location=device)).to(device)
        i = latest_step(flowmodel)
    else:
        flowmodel = FlowModel(make_config(args, device)).to(device)
        i = 0

    stop_training = False
    for epoch in range(args.epochs):
        flowmodel.record(i, "start_epoch", epoch)
        dataloader = make_density_batch_loader(
            dataset_path,
            flowmodel.conf.batch,
            densfn=flowmodel.conf.densfn,
            seed=args.seed + epoch,
            holdout_percent=args.holdout_percent,
            holdout=False,
            num_workers=args.num_workers,
        )
        for x in dataloader:
            x = x.to(device)
            flowmodel.step(i, x)
            _, loss = flowmodel.history["loss"][-1]
            print(f"{i} loss={loss}", flush=True)

            if args.save_every > 0 and i % args.save_every == 0:
                save_checkpoint(flowmodel, save_path)

            i += 1
            if args.max_steps is not None and i >= args.max_steps:
                stop_training = True
                break

        save_checkpoint(flowmodel, Path(annotate_path(str(save_path), f"epoch_{epoch}")))
        if stop_training:
            break

    save_checkpoint(flowmodel, save_path)


if __name__ == "__main__":
    main()
