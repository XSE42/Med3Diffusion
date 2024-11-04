import os
import sys
import math
import json
import shutil
import logging
import argparse
from contextlib import nullcontext
from typing import Optional, Union, Tuple, List

# Add `src` dir to `sys.path`
base_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)

import accelerate
import datasets
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm
from transformers.utils import ContextManagers

import diffusers
from diffusers import DDPMScheduler, EulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from diffusion3d.models.autoencoders.autoencoder_kl import Autoencoder3D
from diffusion3d.models.unets.unet_3d_condition import UNet3DConditionModel
from utils.models import NumCondEncoder
from utils.data import ADNI_3D_128_NumEdge_Set, EdgeSampler
from utils.pipeline_numedge3d import NumEdge3DDiffusionPipeline


if is_wandb_available():
    import wandb


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.22.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def log_validation(
    sampling_dir,
    vae: Autoencoder3D,
    cond_encoder: NumCondEncoder,
    unet: UNet3DConditionModel,
    scheduler: DDPMScheduler,
    args,
    validation_conds: Union[torch.Tensor, List[torch.Tensor]],
    accelerator: Accelerator,
    weight_dtype,
    epoch: int,
    init_atlas_latent: Optional[torch.Tensor] = None,
    val_slice_idx: Union[Tuple[int], List[int]] = (30, 44, 58, 73, 87),
    edge_sampler: Optional[EdgeSampler] = None,
    val_loader: torch.utils.data.DataLoader = None,
):
    logger.info("Running validation... ")

    pipeline = NumEdge3DDiffusionPipeline(
        vae=accelerator.unwrap_model(vae),
        cond_encoder=accelerator.unwrap_model(cond_encoder),
        unet=accelerator.unwrap_model(unet),
        scheduler=scheduler,
    )
    pipeline = pipeline.to(accelerator.device, dtype=weight_dtype)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    vol_origin_list = []
    vol_generate_list = []
    edge_list = []
    for _, batch in enumerate(val_loader):
        batch_vols = batch["scan"]  # 1, 1, 104, 128, 104
        batch_conds = batch["num_cond"].to(accelerator.device)  # 1, 2
        edge = batch["edge"].to(accelerator.device)  # 1, 1, 104, 128, 104

        gend = batch_conds[0][1].item()
        if gend < 0.5:
            gend = "F"
        else:
            gend = "M"
        age = batch_conds[0][0].item() * 100

        vol = pipeline(
            batch_conds,
            num_inference_steps=20,
            generator=generator,
            base_latents=init_atlas_latent,
            edges=edge,
            edge_fuse="concat",
            return_dict=False,
        )[0]  # (1, 1, *DATA_SHAPE)

        vol_origin_list.append(batch_vols)
        vol_generate_list.append(vol)
        edge_list.append(edge)

    vol_list = []
    val_edge_list = []
    for i in range(len(validation_conds)):
        if torch.backends.mps.is_available():
            autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(accelerator.device.type)

        with autocast_ctx:
            gend = validation_conds[i][0][1].item()
            if gend < 0.5:
                gend = "F"
            else:
                gend = "M"
            age = validation_conds[i][0][0].item() * 100
            edge = edge_sampler.get_edge(gend, age).to(accelerator.device)  # 104, 128, 104
            edge = edge.unsqueeze(0).unsqueeze(0)  # 1, 1, 104, 128, 104

            vol = pipeline(
                validation_conds[i],
                num_inference_steps=20,
                generator=generator,
                base_latents=init_atlas_latent,
                edges=edge,
                edge_fuse="concat",
                return_dict=False,
            )[0]  # (1, 1, *DATA_SHAPE)

        vol_list.append(vol)
        val_edge_list.append(edge)

    vol_origin = torch.cat(vol_origin_list, dim=0)  # (N, 1, *DATA_SHAPE)
    vol_origin = (vol_origin / 2 + 0.5).clamp(0, 1)
    vol_origin = vol_origin.cpu().numpy().astype(np.float32)
    vol_generate = torch.cat(vol_generate_list, dim=0)  # (N, 1, *DATA_SHAPE)
    vol_generate = vol_generate.cpu().numpy().astype(np.float32)
    edges = torch.cat(edge_list, dim=0)  # (N, 1, *DATA_SHAPE)
    edges = edges.cpu().numpy().astype(np.int32)
    num_vols = vol_origin.shape[0]

    epoch_str = str(epoch).zfill(3)
    for i in range(num_vols):
        origin_img = sitk.GetImageFromArray(vol_origin[i, 0])
        generate_img = sitk.GetImageFromArray(vol_generate[i, 0])
        edge_img = sitk.GetImageFromArray(edges[i, 0])
        sitk.WriteImage(origin_img, os.path.join(sampling_dir, f"origin{i}_epoch{epoch_str}.nii.gz"))
        sitk.WriteImage(generate_img, os.path.join(sampling_dir, f"generate{i}_epoch{epoch_str}.nii.gz"))
        sitk.WriteImage(edge_img, os.path.join(sampling_dir, f"edge{i}.nii.gz"))

    result_vols = torch.cat(vol_list, dim=0)  # (N, 1, *DATA_SHAPE)
    result_vols = result_vols.cpu().numpy().astype(np.float32)
    val_edges = torch.cat(val_edge_list, dim=0)  # (N, 1, *DATA_SHAPE)
    val_edges = val_edges.cpu().numpy().astype(np.int32)
    num_vols = result_vols.shape[0]

    for i in range(num_vols):
        result_img = sitk.GetImageFromArray(result_vols[i, 0])
        edge_img = sitk.GetImageFromArray(val_edges[i, 0])
        sitk.WriteImage(result_img, os.path.join(sampling_dir, f"val{i}_epoch{epoch_str}.nii.gz"))
        sitk.WriteImage(edge_img, os.path.join(sampling_dir, f"val_edge{i}_epoch{epoch_str}.nii.gz"))

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for slice_idx in val_slice_idx:
                val_slice = (result_vols[:, :, slice_idx] * 255).round().astype(np.uint8)
                tracker.writer.add_images(f"val_slice_{slice_idx}", val_slice, epoch)
                val_edge_slice = (val_edges[:, :, slice_idx] * 255).round().astype(np.uint8)
                tracker.writer.add_images(f"val_edge_slice_{slice_idx}", val_edge_slice, epoch)
                origin_slice = (vol_origin[:, :, slice_idx] * 255).round().astype(np.uint8)
                tracker.writer.add_images(f"origin_slice_{slice_idx}", origin_slice, epoch)
                generate_slice = (vol_generate[:, :, slice_idx] * 255).round().astype(np.uint8)
                tracker.writer.add_images(f"generate_slice_{slice_idx}", generate_slice, epoch)
                edge_slice = (edges[:, :, slice_idx] * 255).round().astype(np.uint8)
                tracker.writer.add_images(f"edge_slice_{slice_idx}", edge_slice, epoch)
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    del pipeline
    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Numeric condition, edge detection, concat, downsampling edge")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Folder containing ADNI dataset"
    )
    parser.add_argument(
        "--init_atlas_dir",
        type=str,
        default=None,
        help="Folder containing initial atlas"
    )
    parser.add_argument(
        "--meta",
        type=str,
        default=None,
        help="JSON file containing metadata of ADNI dataset"
    )
    parser.add_argument(
        "--val_meta",
        type=str,
        default=None,
        help="JSON file containing metadata of ADNI validation dataset"
    )
    parser.add_argument(
        "--meta_sampler",
        type=str,
        default=None,
        help="JSON file containing ADNI metadata sampler"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/numedge_ldm_adni3d",
        help="The output directory where the model predictions and checkpoints will be written.",
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
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
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
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--val_size", type=int, default=8, help="Num of validation data"
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
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
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="numcond_ldm_adni3d",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )

    # 1. directories, accelerator, and logging
    sampling_dir = os.path.join(args.output_dir, args.sampling_dir)
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(sampling_dir, exist_ok=True)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # 2. set seed
    if args.seed is not None:
        set_seed(args.seed)

    # 3. variables
    assert len(args.data_shape) == 3  # data_shape must be 3D
    DATA_SHAPE = args.data_shape
    interpolate_mode = args.interpolate_mode
    ORIG_VAL_SLICE_IDX = [38 + 3, 58 + 3, 78 + 3, 98 + 3, 118 + 3]
    VAL_SLICE_IDX = [round(idx * DATA_SHAPE[0] / 144) for idx in ORIG_VAL_SLICE_IDX]

    # 4. load noise schedulers
    noise_scheduler = DDPMScheduler.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="scheduler"
    )
    pipeline_scheduler = EulerDiscreteScheduler.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="scheduler"
    )

    # 5. load models
    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and Autoencoder3D will not enjoy the parameter sharding
    # across multiple gpus and only UNet3DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        vae = Autoencoder3D.from_pretrained(
            args.pretrained_model, subfolder="vae"
        )

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    if isinstance(vae.config.sample_size, (list, tuple)):
        sample_depth = vae.config.sample_size[0] // vae_scale_factor
        sample_height = vae.config.sample_size[1] // vae_scale_factor
        sample_width = vae.config.sample_size[2] // vae_scale_factor
    else:
        sample_depth = vae.config.sample_size // vae_scale_factor
        sample_height = vae.config.sample_size // vae_scale_factor
        sample_width = vae.config.sample_size // vae_scale_factor
    unet_sample_size = (sample_depth, sample_height, sample_width)


    cond_encoder = NumCondEncoder()
    unet = UNet3DConditionModel(
        sample_size=unet_sample_size,
        in_channels=vae.config.latent_channels + 1,
        out_channels=vae.config.latent_channels,
        center_input_sample=False,
        down_block_types=(
            "DownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
        ),
        mid_block_type="UNetMidBlock3DCrossAttn",
        up_block_types=(
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "UpBlock3D",
        ),
        block_out_channels=(320, 640, 640),
        layers_per_block=1,
        dropout=0.0,
        act_fn="silu",
        norm_num_groups=32,
        norm_eps=1e-5,
        cross_attention_dim=cond_encoder.config.out_features,
        transformer_layers_per_block=(1, 2, 2),
        encoder_hid_dim=None,
        encoder_hid_dim_type=None,
        attention_head_dim=(5, 10, 20),
        use_linear_projection=True,
    )

    # 6. model setup
    # 6.1 freeze or train
    vae.requires_grad_(False)
    unet.train()
    cond_encoder.train()

    # 6.2 gradient checkpointing
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        cond_encoder.enable_gradient_checkpointing()

    # 6.3 create EMA for the models
    if args.use_ema:
        ema_unet = UNet3DConditionModel.from_config(unet.config)
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet3DConditionModel, model_config=ema_unet.config)
        ema_cond_encoder = NumCondEncoder()
        ema_cond_encoder = EMAModel(
            ema_cond_encoder.parameters(),
            model_cls=NumCondEncoder,
            model_config=ema_cond_encoder.config
        )

    # 6.4 enable xformers memory efficient attention for UNet
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # 6.5 accelerate saveing and loading hooks for customized saving
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))
                    ema_cond_encoder.save_pretrained(os.path.join(output_dir, "cond_encoder_ema"))

                for i, model in enumerate(models):
                    if isinstance(model, UNet3DConditionModel):
                        model.save_pretrained(os.path.join(output_dir, "unet"))
                    elif isinstance(model, NumCondEncoder):
                        model.save_pretrained(os.path.join(output_dir, "cond_encoder"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet3DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "cond_encoder_ema"), NumCondEncoder)
                ema_cond_encoder.load_state_dict(load_model.state_dict())
                ema_cond_encoder.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                if isinstance(model, UNet3DConditionModel):
                    load_model = UNet3DConditionModel.from_pretrained(input_dir, subfolder="unet")
                elif isinstance(model, NumCondEncoder):
                    load_model = NumCondEncoder.from_pretrained(input_dir, subfolder="cond_encoder")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # 6.6 allow TF32
    # enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # 7. dataset and dataloader
    with open(args.meta, "r") as handle:
        train_meta = json.load(handle)
    train_dataset = ADNI_3D_128_NumEdge_Set(
        data_dir=args.data_dir,
        scan_list=train_meta["scan_list"],
        edge_name="t1_canny",
        edge_scale=False,
    )

    with open(args.val_meta, "r") as handle:
        val_meta = json.load(handle)
    val_dataset = ADNI_3D_128_NumEdge_Set(
        data_dir=args.data_dir,
        scan_list=val_meta["scan_list"][:args.val_size],
        edge_name="t1_canny",
        edge_scale=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
    )

    with open(args.meta_sampler, "r") as handle:
        sampler = json.load(handle)["sampler"]
    edge_sampler = EdgeSampler(
        args.data_dir,
        sampler=sampler,
        edge_name="t1_canny",
        edge_scale=False,
    )

    # 8. optimizer and scheduler
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # 8.1 choose the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # 8.2 initialize the optimizer
    unet_optim = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    cond_encoder_optim = optimizer_cls(
        cond_encoder.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # 8.3 math around the number of training steps
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # 8.4 initialize the scheduler
    unet_lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=unet_optim,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    cond_encoder_lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=cond_encoder_optim,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # 9. accelerator prepare
    (
        unet, cond_encoder, unet_optim, cond_encoder_optim,
        train_loader, unet_lr_scheduler, cond_encoder_lr_scheduler
    ) = accelerator.prepare(
        unet, cond_encoder, unet_optim, cond_encoder_optim, train_loader, unet_lr_scheduler, cond_encoder_lr_scheduler
    )

    # 9.1 move ema models to device
    if args.use_ema:
        ema_unet.to(accelerator.device)
        ema_cond_encoder.to(accelerator.device)

    # 9.2 cast non-trainable weights to half-precision
    # for mixed precision training we cast all non-trainable weights (vae and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # 9.3 move vae to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)

    # 9.4 initialize trackers
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name)

    # 9.5 function for unwrapping
    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # 10. math around the number of training steps and epochs
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    global_step = 0
    first_epoch = 0

    # 11. resume from checkpoint
    # potentially load in the weights and states from a previous save
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
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    # 12. load initial atlas
    use_init_atlas = False
    init_atlas_latent = None
    if args.init_atlas_dir is not None:
        use_init_atlas = True
        init_atlas = torch.load(os.path.join(args.init_atlas_dir, f"init_atlas.pt")).unsqueeze(0)
        init_atlas = F.interpolate(init_atlas, size=DATA_SHAPE, mode=interpolate_mode)
        init_atlas = init_atlas.to(device=accelerator.device, dtype=weight_dtype)
        init_atlas_latent = vae.encode(init_atlas).latent_dist.mean

    # 13. validation conditions
    VALIDATION_CONDS = [
        torch.tensor([[0.55, 0.0]], dtype=weight_dtype, device=accelerator.device),
        torch.tensor([[0.60, 0.0]], dtype=weight_dtype, device=accelerator.device),
        torch.tensor([[0.65, 0.0]], dtype=weight_dtype, device=accelerator.device),
        torch.tensor([[0.70, 0.0]], dtype=weight_dtype, device=accelerator.device),
        torch.tensor([[0.75, 0.0]], dtype=weight_dtype, device=accelerator.device),
        torch.tensor([[0.80, 0.0]], dtype=weight_dtype, device=accelerator.device),
        torch.tensor([[0.85, 0.0]], dtype=weight_dtype, device=accelerator.device),
        torch.tensor([[0.90, 0.0]], dtype=weight_dtype, device=accelerator.device),
        torch.tensor([[0.55, 1.0]], dtype=weight_dtype, device=accelerator.device),
        torch.tensor([[0.60, 1.0]], dtype=weight_dtype, device=accelerator.device),
        torch.tensor([[0.65, 1.0]], dtype=weight_dtype, device=accelerator.device),
        torch.tensor([[0.70, 1.0]], dtype=weight_dtype, device=accelerator.device),
        torch.tensor([[0.75, 1.0]], dtype=weight_dtype, device=accelerator.device),
        torch.tensor([[0.80, 1.0]], dtype=weight_dtype, device=accelerator.device),
        torch.tensor([[0.85, 1.0]], dtype=weight_dtype, device=accelerator.device),
        torch.tensor([[0.90, 1.0]], dtype=weight_dtype, device=accelerator.device),
    ]

    # 14. training
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Use initial Atlas: {use_init_atlas}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        cond_encoder.train()
        train_loss = 0.0
        # (1) training epoch
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(unet, cond_encoder):
                # a. convert images to latent space
                batch_vols = batch["scan"]
                latents = vae.encode(batch_vols.to(weight_dtype)).latent_dist.mean
                if use_init_atlas:
                    latents = latents - init_atlas_latent
                latents = latents * vae.config.scaling_factor

                # b. sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1, 1), device=latents.device
                    )
                if args.input_perturbation:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)

                # c. sample a random timestep for each image
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # d. add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # e. get the edge latents
                edge_latents = batch["edge"]  # (BS, 1, 104, 128, 104)
                edge_latents = F.avg_pool3d(edge_latents, kernel_size=2, stride=2)  # (BS, 1, 52, 64, 52)
                edge_latents = F.avg_pool3d(edge_latents, kernel_size=2, stride=2)  # (BS, 1, 26, 32, 26)

                # f. concatenate noisy latents and edge latents
                fuse_latents = torch.cat([noisy_latents, edge_latents], dim=1)

                # g. condition embedding
                encoder_hidden_states = cond_encoder(batch["num_cond"])

                # h. get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # i. predict the noise residual and compute loss
                model_pred = unet(fuse_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                # j. compute loss
                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # k. gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # l. backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                    accelerator.clip_grad_norm_(cond_encoder.parameters(), args.max_grad_norm)
                unet_optim.step()
                cond_encoder_optim.step()
                unet_lr_scheduler.step()
                cond_encoder_lr_scheduler.step()
                unet_optim.zero_grad()
                cond_encoder_optim.zero_grad()

            # m. checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                    ema_cond_encoder.step(cond_encoder.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # n. save checkpoint
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
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

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            # o. progress bar logging
            logs = {"step_loss": loss.detach().item(), "lr": unet_lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        # (2) validation
        if accelerator.is_main_process:
            if VALIDATION_CONDS is not None and epoch % args.validation_epochs == 0:
                if args.use_ema:
                    # Store the UNet and CondEncoder parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())
                    ema_cond_encoder.store(cond_encoder.parameters())
                    ema_cond_encoder.copy_to(cond_encoder.parameters())
                log_validation(
                    sampling_dir=sampling_dir,
                    vae=vae,
                    cond_encoder=cond_encoder,
                    unet=unet,
                    scheduler=pipeline_scheduler,
                    args=args,
                    validation_conds=VALIDATION_CONDS,
                    accelerator=accelerator,
                    weight_dtype=weight_dtype,
                    epoch=epoch,
                    init_atlas_latent=init_atlas_latent,
                    val_slice_idx=VAL_SLICE_IDX,
                    edge_sampler=edge_sampler,
                    val_loader=val_loader,
                )
                if args.use_ema:
                    # switch back to the original UNet and CondEncoder parameters.
                    ema_unet.restore(unet.parameters())
                    ema_cond_encoder.restore(cond_encoder.parameters())

    # 15. save the final model
    # create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        cond_encoder = unwrap_model(cond_encoder)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())
            ema_cond_encoder.copy_to(cond_encoder.parameters())

        pipeline = NumEdge3DDiffusionPipeline(
            vae=vae,
            cond_encoder=cond_encoder,
            unet=unet,
            scheduler=pipeline_scheduler,
        )
        pipeline.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
