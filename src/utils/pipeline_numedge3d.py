import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
import SimpleITK as sitk

from diffusers.configuration_utils import FrozenDict
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    deprecate,
    logging,
    replace_example_docstring,
    BaseOutput,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from diffusion3d.models.autoencoders.autoencoder_kl import Autoencoder3D
from diffusion3d.models.unets.unet_3d_condition import UNet3DConditionModel
from .models import NumCondEncoder


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


@dataclass
class NumEdge3DDiffusionOutput(BaseOutput):
    """
    Output class for NumEdge 3D Diffusion pipelines.

    Args:
        images (`List[sitk.Image]` or `torch.Tensor`)
            List of denoised SimpleITK Images of length `batch_size` or PyTorch tensor of shape `(batch_size, depth,
            height, width, num_channels)`.
    """

    volumes: Union[List[sitk.Image], torch.Tensor]


class NumEdge3DDiffusionPipeline(DiffusionPipeline, LoraLoaderMixin, FromSingleFileMixin):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.LoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.LoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files

    Args:
        vae ([`Autoencoder3D`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        cond_encoder ([`NumCondEncoder`]):
            Numeric condition encoder that encodes the numeric condition into a condition embedding.
        unet ([`UNet3DConditionModel`]):
            A `UNet3DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """
    model_cpu_offload_seq = "cond_encoder->unet->vae"

    def __init__(
        self,
        vae: Autoencoder3D,
        cond_encoder: NumCondEncoder,
        unet: UNet3DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            cond_encoder=cond_encoder,
            unet=unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # self.register_to_config(requires_safety_checker=requires_safety_checker)

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    # def _encode_prompt(
    #     self,
    #     prompt,
    #     device,
    #     num_images_per_prompt,
    #     do_classifier_free_guidance,
    #     negative_prompt=None,
    #     prompt_embeds: Optional[torch.FloatTensor] = None,
    #     negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    #     lora_scale: Optional[float] = None,
    # ):
    #     deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
    #     deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

    #     prompt_embeds_tuple = self.encode_prompt(
    #         prompt=prompt,
    #         device=device,
    #         num_images_per_prompt=num_images_per_prompt,
    #         do_classifier_free_guidance=do_classifier_free_guidance,
    #         negative_prompt=negative_prompt,
    #         prompt_embeds=prompt_embeds,
    #         negative_prompt_embeds=negative_prompt_embeds,
    #         lora_scale=lora_scale,
    #     )

    #     # concatenate for backwards comp
    #     prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

    #     return prompt_embeds

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`torch.Tensor` or `List[torch.Tensor]`, *optional*):
                numeric condition to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            adjust_lora_scale_text_encoder(self.cond_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, list):
            prompt = torch.stack(prompt)
        elif prompt is not None and isinstance(prompt, torch.Tensor) and len(prompt.shape) == 1:
            prompt = prompt.unsqueeze(0)

        if prompt_embeds is None:
            prompt_embeds = self.cond_encoder(prompt)

        if self.cond_encoder is not None:
            prompt_embeds_dtype = self.cond_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, channel_embed, _ = prompt_embeds.shape
        # duplicate embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, channel_embed, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds, dtype=prompt_embeds_dtype, device=device)
        else:
            negative_prompt_embeds = None

        return prompt_embeds, negative_prompt_embeds

    def decode_latents(self, latents, base_latents=None):
        latents = latents / self.vae.config.scaling_factor
        if base_latents is not None:
            latents = latents + base_latents
        vol = self.vae.decode(latents, return_dict=False)[0]
        vol = (vol / 2 + 0.5).clamp(0, 1)
        return vol

    def _to_nii(self, vol):
        vol = vol.detach().cpu().numpy()
        vol_list = []
        for i in range(vol.shape[0]):
            vol_list.append(sitk.GetImageFromArray(vol[i, 0]))
        return vol_list

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        depth,
        height,
        width,
        callback_steps,
        prompt_embeds=None,
    ):
        if depth % 8 != 0 or height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`depth`, `height` and `width` have to be divisible by 8.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, torch.Tensor) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `torch.Tensor` or `list` but is {type(prompt)}")

    def prepare_latents(
        self, batch_size, num_channels_latents, depth, height, width, dtype, device, generator, latents=None
    ):
        shape = (
            batch_size,
            num_channels_latents,
            depth // self.vae_scale_factor,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[torch.Tensor, List[torch.Tensor]] = None,
        depth: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        base_latents: Optional[torch.FloatTensor] = None,
        edges: Optional[torch.FloatTensor] = None,
        edge_encode: Optional[str] = "downsample",
        edge_fuse: Optional[str] = "concat",
        edge_add_fact: float = 0.1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pt",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`torch.Tensor` or `List[torch.Tensor]`, *optional*):
                The numeric condition or conditions to guide image generation. If not defined, you need to pass `prompt_embeds`.
            depth (`int`, *optional*):
                The depth in pixels of the generated volume.
            height (`int`, *optional*):
                The height in pixels of the generated volume.
            width (`int`, *optional*):
                The width in pixels of the generated volume.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            base_latents (`torch.FloatTensor`, *optional*):
                Base latents to be added after the denoising process and before the VAE decoding.
                Used when the diffusion model generates residual latents.
            edges (`torch.FloatTensor`, *optional*):
                Edge detection results whose latents will be concatenated or added to noisy latents during diffusion process.
            edge_encode (`str`, *optional*, defaults to `"downsample"`):
                How to encode edge detection results to latents. Choose between `downsample`, `vae`, or `vae_adapter`.
            edge_fuse (`str`, *optional*, defaults to `"concat"`):
                Whether edge latents should be concatenated or added to noisy latents. Choose between `concat`, or `add`.
            edge_add_fact (`float`, *optional*):
                Factor for adding edge latents to noisy latents, factor * edge_latents + (1 - factor) * noisy_latents.
                Used when `edge_fuse` is `add`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pt"`):
                The output format of the generated image. Choose between `pt`, `nii`, or `latent`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipeline_numedge3d.NumEdge3DDiffusionOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.

        Examples:

        Returns:
            [`~pipeline_numedge3d.NumEdge3DDiffusionOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipeline_numedge3d.NumEdge3DDiffusionOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated volumes.
        """
        # 0. Default depth, height and width to unet
        if isinstance(self.unet.config.sample_size, (list, tuple)):
            depth = depth or self.unet.config.sample_size[0] * self.vae_scale_factor
            height = height or self.unet.config.sample_size[1] * self.vae_scale_factor
            width = width or self.unet.config.sample_size[2] * self.vae_scale_factor
        else:
            depth = depth or self.unet.config.sample_size * self.vae_scale_factor
            height = height or self.unet.config.sample_size * self.vae_scale_factor
            width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, depth, height, width, callback_steps, prompt_embeds)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, torch.Tensor):
            if len(prompt.shape) == 1:
                prompt = prompt.unsqueeze(0)
            batch_size = prompt.shape[0]
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.out_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            depth,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare edge latents
        if edge_encode == "downsample":
            edge_latents = F.avg_pool3d(edges, kernel_size=2, stride=2)
            edge_latents = F.avg_pool3d(edge_latents, kernel_size=2, stride=2)
        elif edge_encode == "vae":
            edge_latents = self.vae.encode(edges).latent_dist.mean
            edge_latents = edge_latents * self.vae.config.scaling_factor
        elif edge_encode == "vae_adapter":
            raise ValueError(f"`vae_adapter` has not been implemented yet.")
        repeat_num = 2 if do_classifier_free_guidance else 1
        edge_latents = edge_latents.repeat((repeat_num * num_images_per_prompt, 1, 1, 1, 1))

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if edge_fuse == "concat":
                    latent_model_input = torch.cat([latent_model_input, edge_latents], dim=1)
                elif edge_fuse == "add":
                    latent_model_input = (1 - edge_add_fact) * latent_model_input + edge_add_fact * edge_latents

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not output_type == "latent":
            latents = latents / self.vae.config.scaling_factor
            if base_latents is not None:
                latents = latents + base_latents
            vol = self.vae.decode(latents, return_dict=False)[0]
            vol = (vol / 2 + 0.5).clamp(0, 1)
        else:
            vol = latents

        if output_type == "nii":
            vol = self._to_nii(vol)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (vol,)

        return NumEdge3DDiffusionOutput(volumes=vol)
