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
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout-percent", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-autocast", action="store_true")
    parser.add_argument("--torch-profile-steps", type=int, default=0)
    parser.add_argument("--torch-profile-dir", default=None)
    parser.add_argument("--torch-profile-with-stack", action="store_true")
    return parser.parse_args()


def save_checkpoint(flowmodel: FlowModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(flowmodel.to_dict(), tmp_path)
    os.replace(tmp_path, path)


def latest_step(flowmodel: FlowModel) -> int:
    return max((i for values in flowmodel.history.values() for i, _value in values), default=-1) + 1


def is_cuda_oom(error: RuntimeError) -> bool:
    message = str(error).lower()
    return (
        "cuda" in message
        and (
            "out of memory" in message
            or "cublas_status_alloc_failed" in message
            or "cudnn_status_alloc_failed" in message
        )
    )


def reduce_batch_after_oom(flowmodel: FlowModel, min_batch: int, step: int) -> None:
    old_batch = int(flowmodel.conf.batch)
    if old_batch <= min_batch:
        raise RuntimeError(f"CUDA out of memory at minimum batch size {min_batch}")

    new_batch = max(min_batch, old_batch // 2)
    print(f"CUDA OOM at batch={old_batch}; retrying with batch={new_batch}", flush=True)
    flowmodel.record(step, "oom_batch_reduction", {"old": old_batch, "new": new_batch})
    flowmodel.conf.batch = new_batch
    if flowmodel.optim is not None:
        flowmodel.optim.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def profiler_activities(device: torch.device) -> list[torch.profiler.ProfilerActivity]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return activities


def train_one_step(
    *,
    dataloader_iter,
    flowmodel: FlowModel,
    step: int,
    device: torch.device,
    profile_path: Path | None,
    profile_with_stack: bool,
) -> None:
    def run_step() -> None:
        x = next(dataloader_iter)
        x = x.to(device)
        flowmodel.step(step, x)

    if profile_path is None:
        run_step()
        return

    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=profiler_activities(device),
        record_shapes=True,
        profile_memory=True,
        with_stack=profile_with_stack,
    ) as prof:
        run_step()
    prof.export_chrome_trace(str(profile_path))
    print(f"wrote torch profile: {profile_path}", flush=True)


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
    if args.batch < 1:
        raise SystemExit("--batch must be positive")
    if args.min_batch < 1:
        raise SystemExit("--min-batch must be positive")
    if args.min_batch > args.batch:
        raise SystemExit("--min-batch must be less than or equal to --batch")
    if args.torch_profile_steps < 0:
        raise SystemExit("--torch-profile-steps must not be negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_path = Path(args.dataset)
    if not dataset_path.is_dir():
        raise SystemExit(f"dataset directory does not exist: {dataset_path}")

    save_path = Path(args.output_dir) / args.save_name
    profile_dir = Path(args.torch_profile_dir) if args.torch_profile_dir else Path(args.output_dir) / "profiles"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)
    if device.type == "cuda":
        print("cuda_device:", torch.cuda.get_device_name(0), flush=True)
    print("dataset:", dataset_path, flush=True)
    print("save_path:", save_path, flush=True)
    print("profile:", args.profile, flush=True)
    if args.torch_profile_steps:
        print("torch_profile_dir:", profile_dir, flush=True)

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
        while True:
            retry_epoch = False
            dataloader = make_density_batch_loader(
                dataset_path,
                flowmodel.conf.batch,
                densfn=flowmodel.conf.densfn,
                seed=args.seed + epoch,
                holdout_percent=args.holdout_percent,
                holdout=False,
                num_workers=args.num_workers,
            )
            dataloader_iter = iter(dataloader)
            while True:
                profile_path = None
                if i < args.torch_profile_steps:
                    profile_path = profile_dir / f"step_{i:06d}_batch_{flowmodel.conf.batch}.json"
                try:
                    train_one_step(
                        dataloader_iter=dataloader_iter,
                        flowmodel=flowmodel,
                        step=i,
                        device=device,
                        profile_path=profile_path,
                        profile_with_stack=args.torch_profile_with_stack,
                    )
                except StopIteration:
                    break
                except RuntimeError as error:
                    if not is_cuda_oom(error):
                        raise
                    reduce_batch_after_oom(flowmodel, args.min_batch, i)
                    retry_epoch = True
                    break

                _, loss = flowmodel.history["loss"][-1]
                print(f"{i} batch={flowmodel.conf.batch} loss={loss}", flush=True)

                if args.save_every > 0 and i % args.save_every == 0:
                    save_checkpoint(flowmodel, save_path)

                i += 1
                if args.max_steps is not None and i >= args.max_steps:
                    stop_training = True
                    break

            if stop_training or not retry_epoch:
                break

        save_checkpoint(flowmodel, Path(annotate_path(str(save_path), f"epoch_{epoch}")))
        if stop_training:
            break

    save_checkpoint(flowmodel, save_path)


if __name__ == "__main__":
    main()
