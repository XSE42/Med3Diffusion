import os
import sys
import json
import math
import shutil
import logging
import argparse
from datetime import timedelta

# Add `src` dir to `sys.path`
base_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)

import torch
import torch.nn.functional as F
import numpy as np
import accelerate
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from packaging import version
from tqdm.auto import tqdm
import SimpleITK as sitk

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils import is_tensorboard_available, is_accelerate_version

from diffusion3d.models.autoencoders.autoencoder_kl import Autoencoder3D
from utils.data import ADNI_3D_Set


logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE on ADNI 3D dataset")
    parser.add_argument("--data_dir", type=str, default=None, help="Folder containing ADNI dataset")
    parser.add_argument("--meta", type=str, default=None, help="JSON file containing metadata of ADNI training set")
    parser.add_argument(
        "--val_meta",
        type=str,
        default=None,
        help="JSON file containing metadata of ADNI validation set"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/vae_adni3d",
        help="Folder to save checkpoints and log files",
    )
    parser.add_argument(
        "--sampling_dir",
        type=str,
        default="samples",
        help="Validation sample directory. Will default to *output_dir/samples*",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="TensorBoard log directory. Will default to *output_dir/logs/**CURRENT_DATETIME_HOSTNAME***",
    )
    parser.add_argument(
        "--data_shape", type=int, nargs="+", default=[104, 128, 104], help="Shape of ADNI 3D data for training"
    )
    parser.add_argument(
        "--interpolate_mode",
        type=str,
        default="trilinear",
        help='Interpolation mode for resizing. Choose between ["nearest", "trilinear", "area", "nearest-exact"]',
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader"
    )
    parser.add_argument(
        "--val_size", type=int, default=8, help="The number of volumes to generate for validation"
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="The number of subprocesses to use for data loading",
    )
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument(
        "--save_volume_epochs", type=int, default=25, help="How often to save 3D volume during training"
    )
    parser.add_argument(
        "--save_model_epochs", type=int, default=50, help="How often to save the model during training"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler"
    )
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer"
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--kl_weight", type=float, default=1e-7, help="Weight for the KL term in the loss function")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=4,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main(args):
    sampling_dir = os.path.join(args.output_dir, args.sampling_dir)
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # a big number for high resolution or big dataset
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if not is_tensorboard_available():
        raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "vae"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = Autoencoder3D.from_pretrained(input_dir, subfolder="vae")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(sampling_dir, exist_ok=True)

    assert len(args.data_shape) == 3  # data_shape must be 3D
    DATA_SIZE = 128
    DATA_SHAPE = args.data_shape
    interpolate_mode = args.interpolate_mode
    ORIG_VAL_SLICE_IDX = [38 + 3, 58 + 3, 78 + 3, 98 + 3, 118 + 3]
    VAL_SLICE_IDX = [round(idx * DATA_SHAPE[0] / 144) for idx in ORIG_VAL_SLICE_IDX]

    model = Autoencoder3D(
        in_channels = 1,
        out_channels = 1,
        down_block_types = ("DownEncoderBlock3D", "DownEncoderBlock3D", "DownEncoderBlock3D"),
        up_block_types = ("UpDecoderBlock3D", "UpDecoderBlock3D", "UpDecoderBlock3D"),
        block_out_channels = (32, 64, 128),
        layers_per_block = 1,
        act_fn = "silu",
        latent_channels = 2,
        norm_num_groups = 8,
        mid_block_add_attention = False,
        sample_size = DATA_SHAPE,
        scaling_factor = 1.0,
    )
    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets
    with open(args.meta, "r") as handle:
        train_meta = json.load(handle)
    train_dataset = ADNI_3D_Set(
        data_dir=args.data_dir,
        scan_list=train_meta["scan_list"]
    )
    with open(args.val_meta, "r") as handle:
        val_meta = json.load(handle)
    val_dataset = ADNI_3D_Set(
        data_dir=args.data_dir,
        scan_list=val_meta["scan_list"][:args.val_size]
    )

    logger.info(f"Dataset size: {len(train_dataset)}")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.train_batch_size, shuffle=False
    )

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # Train!
    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            x = F.interpolate(batch, size=DATA_SHAPE, mode=interpolate_mode)

            with accelerator.accumulate(model):
                posterior = model.encode(x).latent_dist
                z = posterior.sample()  # TODO: sample() or mode() ?
                pred = model.decode(z).sample

                mse_loss = F.mse_loss(pred, x)
                kl_loss = posterior.kl().mean()
                loss = mse_loss + args.kl_weight * kl_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "loss": loss.detach().item(),
                "mse_loss": mse_loss.detach().item(),
                "kl_loss": kl_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        accelerator.wait_for_everyone()

        # Generate sample volumes for visual inspection
        if accelerator.is_main_process:
            if epoch % args.save_volume_epochs == 0 or epoch == args.num_epochs - 1:
                if is_accelerate_version(">=", "0.17.0.dev0"):
                    tracker = accelerator.get_tracker("tensorboard", unwrap=True)
                else:
                    tracker = accelerator.get_tracker("tensorboard")

                vae = accelerator.unwrap_model(model)

                epoch_str = str(epoch).zfill(3)
                vol_idx = 0  # validation volume index
                orig_slices = {slice_idx: [] for slice_idx in VAL_SLICE_IDX}  # record slices of input vols
                recons_slices = {slice_idx: [] for slice_idx in VAL_SLICE_IDX}  # record slices of reconstructed vols

                for i, batch in enumerate(val_dataloader):
                    batch = F.interpolate(batch, size=DATA_SHAPE, mode=interpolate_mode)
                    orig_vols = batch
                    x = batch.to(accelerator.device)
                    recons_vols = vae(x).sample

                    # denormalize the volumes and save to tensorboard
                    orig_vols = (orig_vols / 2 + 0.5).clamp(0, 1)
                    orig_vols = orig_vols.cpu().numpy()
                    recons_vols = (recons_vols / 2 + 0.5).clamp(0, 1)
                    recons_vols = recons_vols.detach().cpu().numpy()

                    # save volumes
                    num_vols = orig_vols.shape[0]
                    for batch_idx in range(num_vols):
                        orig_img = sitk.GetImageFromArray(orig_vols[batch_idx, 0])
                        recons_img = sitk.GetImageFromArray(recons_vols[batch_idx, 0])
                        sitk.WriteImage(orig_img, os.path.join(sampling_dir, f"orig{vol_idx}_epoch{epoch_str}.nii.gz"))
                        sitk.WriteImage(recons_img, os.path.join(sampling_dir, f"recons{vol_idx}_epoch{epoch_str}.nii.gz"))
                        vol_idx += 1

                    # save slices
                    orig_vols = (orig_vols * 255).round().astype("uint8")
                    recons_vols = (recons_vols * 255).round().astype("uint8")
                    for slice_idx in VAL_SLICE_IDX:
                        orig_slices[slice_idx].append(orig_vols[:, :, slice_idx])
                        recons_slices[slice_idx].append(recons_vols[:, :, slice_idx])

                for slice_idx in VAL_SLICE_IDX:
                    tracker.add_images(f"val_orig_slice_{slice_idx}", np.concatenate(orig_slices[slice_idx]), epoch)
                    tracker.add_images(f"val_recons_slice_{slice_idx}", np.concatenate(recons_slices[slice_idx]), epoch)

            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                # save the model
                vae = accelerator.unwrap_model(model)

                vae.save_pretrained(os.path.join(args.output_dir, "vae"))

    progress_bar.close()

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
