import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from attrdict import AttrDict
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

from utils import load_model_from_config, chunk, numpy_to_pil, check_safety, torch_to_pil, torch_to_grid_pil


default_options = {
    "ddim_steps": 50,
    "n_iter": 2,
    "n_samples": 6,
    "scale": 7.5,
    "plms": False,
    "outdir": "outputs/txt2img-samples",
    "seed": None,
    "filter_nsfw": True,
    "fixed_code": False,
    "ddim_eta": 0.0,
    "H": 512,
    "W": 512,
    "C": 4,
    "f": 8,
    "precision": "autocast",
    "from_file": None,
    "n_rows": 0
}

class SDController:

    opt = None

    def __init__(self, ckpt="models/ldm/stable-diffusion-v1/model.ckpt", config="configs/stable-diffusion/v1-inference.yaml", **options):
        self.opt = AttrDict(default_options.copy())
        self.opt.update(options)
        self.load_model(ckpt, config)

    def load_model(self, ckpt, config):
        config = OmegaConf.load(config)
        self.model = load_model_from_config(config, ckpt)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)
        self.plms_sampler =  PLMSSampler(self.model)
        self.ddim_sampler = DDIMSampler(self.model)

    def seed_init(self, seed=None):
        if seed is None:
            seed = int(time.time())
        seed_everything(seed)

    def txt2img(self, prompt, **options):
        opt = AttrDict(self.opt.copy())
        opt.update(options)

        self.seed_init(opt.seed)

        sampler = self.plms_sampler if opt.plms else self.ddim_sampler
        
        os.makedirs(opt.outdir, exist_ok=True)
        outpath = opt.outdir

        batch_size = opt.n_samples
        n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

        data = [batch_size * [prompt]]

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)

        base_count = len(os.listdir(sample_path))
        grid_count = len(os.listdir(outpath)) - 1

        start_code = None
        if opt.fixed_code:
            start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=self.device)

        precision_scope = autocast if opt.precision=="autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    all_samples = list()
                    for n in trange(opt.n_iter, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if opt.scale != 1.0:
                                uc = self.model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = self.model.get_learned_conditioning(prompts)
                            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                            samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                            conditioning=c,
                                                            batch_size=opt.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=opt.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=opt.ddim_eta,
                                                            x_T=start_code)

                            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                            if opt.filter_nsfw:
                                x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                            else:
                                x_checked_image = x_samples_ddim.copy()
                                for i, nsfw in enumerate(check_safety(x_samples_ddim)[1]):
                                    if nsfw:
                                        print("Unsafe image was found at index: ", n * opt.n_samples + i)

                            x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                            # save the images
                            for img in torch_to_pil(x_checked_image_torch):
                                img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                                base_count += 1
                            
                            all_samples.append(x_checked_image_torch)

                    # additionally, save as grid
                    grid = torch_to_grid_pil(all_samples, n_rows=n_rows)
                    torch_to_pil(grid).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    grid_count += 1
