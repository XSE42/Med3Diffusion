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
import accelerate
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from packaging import version
from tqdm.auto import tqdm
import numpy as np
import SimpleITK as sitk

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils import is_tensorboard_available, is_accelerate_version

from utils.data import ADNI_3D_128_NumCond_Set
from utils.models import VxmCondAtlas
from utils.losses import JacobianDeterminant3D
from voxelmorph.networks import VxmDense, SpatialTransformer
from voxelmorph.losses import MSE, Grad


logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Cond Atlas with VoxelMorph on ADNI 128 dataset")
    parser.add_argument("--pretrained_model", type=str, default=None, help="Path to pretrained VxmDense model")
    parser.add_argument("--data_dir", type=str, default=None, help="Folder containing ADNI dataset")
    parser.add_argument("--synth_data_dir", type=str, default=None, help="Folder containing Synthetic dataset")
    parser.add_argument("--init_atlas_dir", type=str, default=None, help="Folder containing initial atlas")
    parser.add_argument("--meta", type=str, default=None, help="JSON file containing metadata of ADNI training set")
    parser.add_argument(
        "--synth_meta",
        type=str,
        default=None,
        help="JSON file containing metadata of Synthetic training set"
    )
    parser.add_argument(
        "--val_meta",
        type=str,
        default=None,
        help="JSON file containing metadata ADNI validation set"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/atlas_vxm_adni3d",
        help="Folder to save checkpoints and log files",
    )
    parser.add_argument(
        "--sample_dir",
        type=str,
        default="samples",
        help="Validation sample directory. Will default to *output_dir/samples*",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="TensorBoard log directory. Will default to *output_dir/logs/**CURRENT_DATETIME_HOSTNAME***",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader"
    )
    parser.add_argument(
        "--val_size", type=int, default=16, help="The number of images to generate for validation"
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="The number of subprocesses to use for data loading",
    )
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--save_volume_epochs", type=int, default=25, help="How often to save images during training")
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
        help="Initial learning rate to use",
    )
    parser.add_argument(
        "--atlas_lr",
        type=float,
        default=1e-4,
        help="Initial learning rate to update atlas",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="linear",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--atlas_lr_scheduler",
        type=str,
        default="linear",
        help=(
            'The scheduler type for atlas. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler"
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-3, help="Weight decay magnitude for the Adam optimizer"
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--sim_weight", type=float, default=1.0, help="Weight for the Sim term in loss ")
    parser.add_argument("--grad_weight", type=float, default=1.0, help="Weight for the Grad term in the loss function")
    parser.add_argument(
        "--deform_weight",
        type=float,
        default=1e-2,
        help="Weight for the Deformation MSE term in the loss function"
    )
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
            ' `--checkpointing_steps` , or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main(args):
    # 1. setup accelerator
    sampling_dir = os.path.join(args.output_dir, args.sample_dir)
    logging_dir = os.path.join(args.output_dir, args.log_dir)
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
                    if isinstance(model, VxmDense):
                        model.save_pretrained(os.path.join(output_dir, "vxm"))
                    elif isinstance(model, VxmCondAtlas):
                        model.save_pretrained(os.path.join(output_dir, "atlas"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                if isinstance(model, VxmDense):
                    load_model = VxmDense.from_pretrained(input_dir, subfolder="vxm")
                elif isinstance(model, VxmCondAtlas):
                    load_model = VxmCondAtlas.from_pretrained(input_dir, subfolder="atlas")
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

    # 2. load data
    with open(args.meta, "r") as handle:
        train_meta = json.load(handle)
    if args.synth_meta is not None:
        assert args.synth_data_dir is not None
        with open(args.synth_meta, "r") as handle:
            synth_meta = json.load(handle)
        synth_scan_list = synth_meta["scan_list"]
    else:
        synth_scan_list = None
    train_dataset = ADNI_3D_128_NumCond_Set(
        data_dir=args.data_dir,
        scan_list=train_meta["scan_list"],
        synth_data_dir=args.synth_data_dir,
        synth_scan_list=synth_scan_list,
    )
    with open(args.val_meta, "r") as handle:
        val_meta = json.load(handle)
    val_dataset = ADNI_3D_128_NumCond_Set(
        data_dir=args.data_dir,
        scan_list=val_meta["scan_list"][:args.val_size],
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.val_size, shuffle=False
    )

    logger.info(f"Dataset size: {len(train_dataset)}")

    # 3. constants & model
    DATA_SHAPE = (112, 128, 112)
    PAD_SIZE = 4
    PAD_SEQ = (PAD_SIZE, PAD_SIZE, 0, 0, PAD_SIZE, PAD_SIZE)
    ORIG_VAL_SLICE_IDX = [38 + 3, 58 + 3, 78 + 3, 98 + 3, 118 + 3]
    VAL_SLICE_IDX = [round(idx * 104 / 144) + PAD_SIZE for idx in ORIG_VAL_SLICE_IDX]

    SIM_WEIGHT = args.sim_weight
    GRAD_WEIGHT = args.grad_weight
    DEFOR_WEIGHT = args.deform_weight

    if args.pretrained_model is not None:
        vxm: VxmDense = VxmDense.from_pretrained(args.pretrained_model, subfolder="vxm")
        # vxm.bidir = True
        # vxm.config["bidir"] = True
    else:
        vxm: VxmDense = VxmDense(
        inshape=DATA_SHAPE,
        nb_unet_features=[
            [16, 32, 32, 32],
            [32, 32, 32, 32, 32, 16, 16],
        ],
        int_downsize=1,
    )

    init_atlas = torch.load(os.path.join(args.init_atlas_dir, "init_atlas_128.pt")).unsqueeze(0)  # (1, 1, 104, 128, 104)
    init_atlas = F.pad(init_atlas, pad=PAD_SEQ, mode="constant", value=-1)  # (1, 1, 112, 128, 112)
    init_atlas = init_atlas.to(accelerator.device)
    atlas_model = VxmCondAtlas(
        cond_features=1,
        atlas_shape=DATA_SHAPE,
        out_channels=1,
        hidden_channels=4,
        extra_conv_layers=3,
    )

    # 4. set optimizer and loss function
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch
    optimizer = torch.optim.AdamW(
        vxm.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )
    atlas_optimizer = torch.optim.AdamW(
        atlas_model.parameters(),
        lr=args.atlas_lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    atlas_lr_scheduler = get_scheduler(
        args.atlas_lr_scheduler,
        optimizer=atlas_optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )
    mse_loss_fn = MSE()
    grad_loss_fn = Grad("l2", loss_mult=2)

    # 5. accelerator prepare & initialize trackers
    (
        vxm, atlas_model, optimizer, atlas_optimizer, train_dataloader, lr_scheduler, atlas_lr_scheduler
    ) = accelerator.prepare(
        vxm, atlas_model, optimizer, atlas_optimizer, train_dataloader, lr_scheduler, atlas_lr_scheduler
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

    # 6. potentially load in the weights and states from a previous save
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

    # 7. prepare SpatialTransformer and grid for validation
    if accelerator.is_main_process:
        spatial_transformer = SpatialTransformer(vxm.inshape).to(accelerator.device)
        grid_shape = (len(val_dataset), 1, *(vxm.inshape))
        grid = torch.zeros(grid_shape, dtype=torch.float32, device=accelerator.device)
        d_index = torch.arange(start=1, end=vxm.inshape[0], step=6)
        h_index = torch.arange(start=1, end=vxm.inshape[1], step=6)
        w_index = torch.arange(start=1, end=vxm.inshape[2], step=6)
        # grid[:, :, d_index, :, :] = 1
        grid[:, :, :, h_index, :] = 1
        grid[:, :, :, :, w_index] = 1

        jacob_det = JacobianDeterminant3D()

    # 8. train
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    for epoch in range(first_epoch, args.num_epochs):
        vxm.train()
        atlas_model.train()
        for step, batch in enumerate(train_dataloader):
            # skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(vxm, atlas_model):
                # data
                batch_vols = F.pad(batch["scan"], pad=PAD_SEQ, mode="constant", value=-1)
                batch_conds = batch["num_cond"]
                if batch_vols.shape[0] % 2 == 1:
                    batch_vols = batch_vols.repeat_interleave(repeats=2, dim=0)
                    batch_conds = batch_conds.repeat_interleave(repeats=2, dim=0)
                atlas_tensor = atlas_model(batch_conds)
                atlas_tensor += init_atlas  # (BS, 1, *DATA_SHAPE)
                # forward
                atlas_warped, disp_flow = vxm(atlas_tensor, batch_vols, registration=True)
                # loss
                sim_loss = mse_loss_fn.loss(batch_vols, atlas_warped)
                grad_loss = grad_loss_fn.loss(None, disp_flow)
                deform_loss = mse_loss_fn.loss(0, disp_flow)
                loss = SIM_WEIGHT * sim_loss + GRAD_WEIGHT * grad_loss + DEFOR_WEIGHT * deform_loss
                # backward
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(vxm.parameters(), 1.0)
                optimizer.step()
                atlas_optimizer.step()
                lr_scheduler.step()
                atlas_lr_scheduler.step()
                optimizer.zero_grad()
                atlas_optimizer.zero_grad()

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
                "sim_loss": sim_loss.detach().item(),
                "grad_loss": grad_loss.detach().item(),
                "deform_loss": deform_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        accelerator.wait_for_everyone()

        # generate sample images for visual inspection
        if accelerator.is_main_process:
            if epoch % args.save_volume_epochs == 0 or epoch == args.num_epochs - 1:
                vxm = accelerator.unwrap_model(vxm)
                atlas_model = accelerator.unwrap_model(atlas_model)

                with torch.no_grad():
                    for i, batch in enumerate(val_dataloader):
                        batch_vols = batch["scan"].to(accelerator.device)
                        batch_conds = batch["num_cond"].to(accelerator.device)
                        batch_vols = F.pad(batch_vols, pad=PAD_SEQ, mode="constant", value=-1)
                        atlas_tensor = atlas_model(batch_conds) + init_atlas
                        atlas_warped, displacement_flow = vxm(atlas_tensor, batch_vols, registration=True)

                    # visualization
                    grid_warped = spatial_transformer(grid, displacement_flow).clamp(0, 1).detach().cpu().numpy()

                    num_foldings = jacob_det(displacement_flow).cpu()
                    tag_scalar_dict = dict()
                    for j in range(num_foldings.shape[0]):
                        tag_scalar_dict[f"num_foldings_{j}"] = num_foldings[j].item()

                    source = atlas_tensor / 2 + 0.5
                    target = batch_vols / 2 + 0.5
                    source_warped = atlas_warped / 2 + 0.5
                    source = source.clamp(0, 1).cpu().numpy()
                    target = target.clamp(0, 1).cpu().numpy()
                    source_warped = source_warped.clamp(0, 1).detach().cpu().numpy()

                    for batch_idx in range(source.shape[0]):
                        src_img = sitk.GetImageFromArray(source[batch_idx, 0])
                        trg_img = sitk.GetImageFromArray(target[batch_idx, 0])
                        src_warped_img = sitk.GetImageFromArray(source_warped[batch_idx, 0])
                        grid_warped_img = sitk.GetImageFromArray(grid_warped[batch_idx, 0])
                        age = batch_conds[batch_idx].item()
                        sitk.WriteImage(
                            src_img,
                            os.path.join(sampling_dir, f"atlas_epoch{epoch}_id{batch_idx}_age{age}.nii.gz")
                        )
                        sitk.WriteImage(
                            trg_img,
                            os.path.join(sampling_dir, f"trg_epoch{epoch}_id{batch_idx}_age{age}.nii.gz")
                        )
                        sitk.WriteImage(
                            src_warped_img,
                            os.path.join(sampling_dir, f"atlas_warped_epoch{epoch}_id{batch_idx}_age{age}.nii.gz")
                        )
                        sitk.WriteImage(
                            grid_warped_img,
                            os.path.join(sampling_dir, f"grid_warped_epoch{epoch}_id{batch_idx}_age{age}.nii.gz")
                        )

                    for idx in VAL_SLICE_IDX:
                        diff_pre_warped = (source[:, :, idx] - target[:, :, idx]).squeeze(axis=1)
                        diff_post_warped = (source_warped[:, :, idx] - target[:, :, idx]).squeeze(axis=1)
                        color_diff_pre = np.ones((*diff_pre_warped.shape, 3))
                        color_diff_post = np.ones((*diff_post_warped.shape, 3))
                        color_diff_pre[diff_pre_warped > 0, 1:3] -= diff_pre_warped[diff_pre_warped > 0, np.newaxis]  # Red
                        color_diff_pre[diff_pre_warped < 0, 0:2] += diff_pre_warped[diff_pre_warped < 0, np.newaxis]  # Blue
                        color_diff_post[diff_post_warped > 0, 1:3] -= diff_post_warped[diff_post_warped > 0, np.newaxis]  # Red
                        color_diff_post[diff_post_warped < 0, 0:2] += diff_post_warped[diff_post_warped < 0, np.newaxis]  # Blue

                        source_img = source[:, :, idx]
                        source_img = (source_img * 255).round().astype("uint8")
                        target_img = target[:, :, idx]
                        target_img = (target_img * 255).round().astype("uint8")
                        source_warped_img = source_warped[:, :, idx]
                        source_warped_img = (source_warped_img * 255).round().astype("uint8")

                        color_diff_pre = (color_diff_pre * 255).round().astype("uint8")
                        color_diff_post = (color_diff_post * 255).round().astype("uint8")

                        grid_warped_img = grid_warped[:, :, idx]
                        grid_warped_img = (grid_warped_img * 255).round().astype("uint8")

                        if is_accelerate_version(">=", "0.17.0.dev0"):
                            tracker = accelerator.get_tracker("tensorboard", unwrap=True)
                        else:
                            tracker = accelerator.get_tracker("tensorboard")
                        tracker.add_images(f"atlas_slice{idx}", source_img, epoch)
                        tracker.add_images(f"target_slice{idx}", target_img, epoch)
                        tracker.add_images(f"atlas_warped_slice{idx}", source_warped_img, epoch)
                        tracker.add_images(f"color_diff_pre_slice{idx}", color_diff_pre, epoch, dataformats="NHWC")
                        tracker.add_images(f"color_diff_post_slice{idx}", color_diff_post, epoch, dataformats="NHWC")
                        tracker.add_images(f"grid_warped_slice{idx}", grid_warped_img, epoch)
                    tracker.add_scalars("num_foldings", tag_scalar_dict, epoch)

            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                # save the model
                vxm = accelerator.unwrap_model(vxm)
                atlas_model = accelerator.unwrap_model(atlas_model)

                vxm.save_pretrained(os.path.join(args.output_dir, "vxm"))
                atlas_model.save_pretrained(os.path.join(args.output_dir, "atlas"))

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
