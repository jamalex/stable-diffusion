import argparse, os, sys, glob
import cv2
import torch
import numpy as np
import os
from attrdict import AttrDict
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.ksampler import KSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

from utils import load_model_from_config, check_for_nsfw, slerp

from image_classes import SDImage, SDImageList, SDImageListList

from matplotlib import pyplot as plt
plt.ion()

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


default_options = AttrDict({
    "steps": 50,
    "n_iter": 2,
    "n_samples": 3,
    "scale": 7.5,
    "sampler": "k_euler_a",
    "outdir": "outputs/txt2img-samples",
    "seed": None,
    "fixed_code": False,
    "ddim_eta": 0.0,
    "H": 512,
    "W": 512,
    "C": 4,
    "f": 8,
    "from_file": None,
    "n_rows": 0,
    "check_nsfw": False,
    "extract_intermediates": False,
    "init_image": None,
    "strength": 0.3,
    "x0": None,
    "mask": None,
})


class SDController:

    opt = None

    def __init__(self, ckpt="models/ldm/stable-diffusion-v1/model.ckpt", config="configs/stable-diffusion/v1-inference.yaml", **options):
        self.opt = AttrDict(default_options.copy())
        self.opt.update(options)
        self.load_model(ckpt, config)

    def load_model(self, ckpt, config):
        config = OmegaConf.load(config)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = load_model_from_config(config, ckpt, self.device)

    def seed_init(self, seed=None):
        if seed is None:
            seed = int(time.time())
        seed_everything(seed)
        return seed

    @contextmanager
    def _default_scopes(self):
        with torch.no_grad(), autocast("cuda"), self.model.ema_scope():
            yield None

    def encode_prompt_to_numpy(self, prompt):
        return self.encode_prompt(prompt)[0].cpu().detach().numpy()

    def encode_prompt(self, prompt):
        return self.model.get_learned_conditioning([prompt])

    def decode_to_torch(self, latents):
        torch_images = self.model.decode_first_stage(latents)
        torch_images = torch.clamp((torch_images + 1.0) / 2.0, min=0.0, max=1.0)
        return torch_images

    def decode_to_images(self, latents):
        torch_images = self.decode_to_torch(latents)
        return SDImageList.fromtorchlist(torch_images)

    def encode_to_torch(self, images):
        if not isinstance(images, list):
            images = [images]
        image_layers = []
        for image in images:
            image = np.array(image).astype(np.float32) / 255.0
            image = image[None].transpose(0, 3, 1, 2)
            image = torch.from_numpy(image)[0]
            image_layers.append(2.*image - 1.)
        stack = torch.stack(image_layers, dim=0).to(self.device)
        encoding = self.model.encode_first_stage(stack)
        return self.model.get_first_stage_encoding(encoding)

    def get_sampler(self, opt):
        if opt.sampler == "plms":
            return PLMSSampler(self.model)
        elif opt.sampler == "ddim":
            return DDIMSampler(self.model)
        elif opt.sampler == "k_dpm_2_a":
            return KSampler(self.model, "dpm_2_ancestral")
        elif opt.sampler == "k_dpm_2":
            return KSampler(self.model, "dpm_2")
        elif opt.sampler == "k_euler_a":
            return KSampler(self.model, "euler_ancestral")
        elif opt.sampler == "k_euler":
            return KSampler(self.model, "euler")
        elif opt.sampler == "k_heun":
            return KSampler(self.model, "heun")
        elif opt.sampler == "k_lms":
            return KSampler(self.model, "lms")
        else:
            raise ValueError(f"Invalid sampler: {opt.sampler}")

    def prep_input_image(self, image, opt=default_options):
        if image is None:
            return None
        else:
            if isinstance(image, str):
                # load the image if a path was provided
                image = SDImage.from_file(image, opt.H, opt.W)
            elif isinstance(image, SDImage):
                image = image
            elif isinstance(image, torch.Tensor):
                if image.shape[1] == 3:  # RGB image
                    image = SDImageList.fromtorchlist(image)[0]
                elif image.shape[1] == 4:  # latent
                    image = self.decode_to_images(image)[0]
            if image.size != (opt.H, opt.W):
                image = image.resize((opt.H, opt.W), resample=Image.LANCZOS)
            image = self.encode_to_torch(image)
            image = repeat(image, '1 ... -> b ...', b=opt.n_samples)
            return image

    def prep_mask(self, mask, opt=default_options):
        if isinstance(mask, SDImage):
            channels, latent_height, latent_width = opt.C, opt.H // opt.f, opt.W // opt.f
            mask = ImageOps.grayscale(mask).resize((latent_height, latent_width), resample=Image.LANCZOS)
            mask = np.array(mask).astype(np.float32) / 255.0
            mask = torch.from_numpy(mask).to(self.device)
            mask = mask[None, None].repeat(opt.n_samples, channels, 1, 1)
            # mask[mask < 0.5] = 0.0
            # mask[mask >= 0.5] = 1.0
        return mask

    def generate(self, prompt, **options):
        opt = AttrDict(self.opt.copy())
        opt.update(options)

        opt.seed = self.seed_init(opt.seed)

        sampler = self.get_sampler(opt)

        batch_size = opt.n_samples
        n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

        latent_shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

        init_image = self.prep_input_image(opt.init_image, opt)
        x0 = self.prep_input_image(opt.x0, opt)
        mask = self.prep_mask(opt.mask, opt)

        if init_image is not None:

            # set up the sampler
            if not isinstance(sampler, DDIMSampler):
                print("WARNING: init_image is only supported for DDIMSampler, so we have switched")
                opt.sampler = "ddim"
                sampler = self.get_sampler(opt)
            sampler.make_schedule(ddim_num_steps=opt.steps, ddim_eta=opt.ddim_eta, verbose=False)

            assert 0. <= opt.strength <= 1., 'img2img can only work with strength in [0.0, 1.0]'
            t_enc = int(opt.strength * opt.steps)

        if opt.fixed_code is True:
            start_code = torch.randn([opt.n_samples] + latent_shape, device=self.device)
        elif opt.fixed_code is False:
            start_code = None
        elif isinstance(opt.fixed_code, torch.Tensor):
            start_code = opt.fixed_code

        with self._default_scopes():

            all_samples = []
            intermediates = SDImageListList([])
            all_nsfw_flags = []

            # unconditional conditioning
            uc = self.model.get_learned_conditioning(batch_size * [""])

            # conditioning
            if isinstance(prompt, str):
                c = self.model.get_learned_conditioning(batch_size * [prompt])
            elif isinstance(prompt, torch.Tensor):
                assert prompt.shape[1] == 77 and prompt.shape[2] == 768, "prompt tensor must have shape [*, 77, 768]"
                if prompt.shape[0] == 1:
                    c = repeat(prompt, '1 ... -> b ...', b=opt.n_samples)
                elif prompt.shape[0] == batch_size:
                    c = prompt
                else:
                    raise ValueError(f"Invalid prompt shape: {prompt.shape} (expected first dimension to be 1 or {opt.n_samples})")

            for batch in range(opt.n_iter):

                if opt.extract_intermediates:
                    def img_callback(pred_x0, i):
                        batch_index_offset = batch * opt.n_samples
                        batch_images = self.decode_to_images(pred_x0)
                        for j, img in enumerate(batch_images):
                            index = batch_index_offset + j
                            if index >= len(intermediates):
                                intermediates.append(SDImageList([]))
                            intermediates[index].append(img)
                else:
                    img_callback = None

                if init_image is None:  # MODE: txt2img

                    samples, _ = sampler.sample(
                        S=opt.steps,
                        conditioning=c,
                        batch_size=opt.n_samples,
                        shape=latent_shape,
                        verbose=False,
                        img_callback=img_callback,
                        unconditional_guidance_scale=opt.scale,
                        unconditional_conditioning=uc,
                        eta=opt.ddim_eta,
                        x_T=start_code,
                        x0=x0,
                        mask=mask,
                    )

                else:  # MODE: img2img
                    
                    # encode (scaled latent)
                    noise = torch.randn_like(init_image)
                    z_enc = sampler.stochastic_encode(init_image, torch.tensor([t_enc]*batch_size).to(self.device), noise=noise)
                    
                    # decode it
                    samples = sampler.decode(
                        z_enc,
                        c,
                        t_enc,
                        unconditional_guidance_scale=opt.scale,
                        unconditional_conditioning=uc,
                        img_callback=img_callback,
                        mask=mask,
                        x0=x0,
                    )

                x_samples = self.decode_to_torch(samples)

                if opt.check_nsfw:
                    all_nsfw_flags += check_for_nsfw(x_samples)

                all_samples.append(x_samples)

        results = AttrDict({
            "images": SDImageList.fromtorchlist(all_samples, opts=[opt]*len(all_samples), nsfw_flags=all_nsfw_flags),
            "grid": SDImage.fromtorchgrid(all_samples, n_rows=n_rows),
            "seed": opt.seed,
            "nsfw_flags": all_nsfw_flags or None,
            "intermediates": intermediates,
            "options": opt,
        })
        
        return results

c = SDController()

