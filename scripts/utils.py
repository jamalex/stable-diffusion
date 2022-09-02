import logging
import argparse, os, sys, glob
import cv2
import functools
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import transformers
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

CURRENT_MODEL = None

CLIP_TOKEN_NORM = torch.tensor([[[43.8203]] + [[28]] * 76])

transformers.logging.set_verbosity_error()

def get_neighbors_and_weight(index_dict, position):
    """
    Get the before and after value, and weight between them (0 = exactly at "before",
    1 = exactly at "after") for a given position.
    """
    positions = sorted(index_dict.keys())
    before_pos_i, after_pos_i = find_indices(position, positions)
    if before_pos_i is None or before_pos_i == after_pos_i:
        return index_dict[positions[after_pos_i]], index_dict[positions[after_pos_i]], 0
    elif after_pos_i is None:
        return index_dict[positions[before_pos_i]], index_dict[positions[before_pos_i]], 1
    before_val, after_val = index_dict[positions[before_pos_i]], index_dict[positions[after_pos_i]]
    weight = remap(position, positions[before_pos_i], positions[after_pos_i], 0.0, 1.0)
    return before_val, after_val, weight


def convert_to_index_dict(items, start=0.0, end=1.0, default=None):
    """
    Convert a list of items to a dictionary of indices, where the keys are the position floats between start and end.
    """
    index_dict = {}
    if isinstance(items, dict):
        index_dict = items
    elif isinstance(items, list):
        if len(items) > 1:
            index_dict = {remap(i, 0, len(items) - 1, start, end): val for i, val in enumerate(items)}
        elif len(items) > 0:
            index_dict = {start: items[0], end: items[0]}
    else:
        index_dict = {start: items, end: items}

    if start not in index_dict:
        index_dict[start] = default
    if end not in index_dict:
        index_dict[end] = default

    return index_dict


def find_indices(val, arr):
    """
    Function to find position (before/after indices) of "val" in sorted list "arr"
    """
 
    arr = sorted(arr)
    assert list(arr) == sorted(arr), "List 'arr' must be pre-sorted"

    start = -1
    end = len(arr)
 
    while start < end - 1:

        mid = (start + end) // 2
 
        if val == arr[mid]:
            return (mid, mid)
        elif val > arr[mid]:
            start = mid
        else:
            end = mid
 
    return (start if start >= 0 else None, end if end < len(arr) else None)


def remap(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def renormalize_clip_embedding(x):
    """ Unclear whether this is useful, but it helps to keep the token norms matching actual embeddings """
    x = x * CLIP_TOKEN_NORM.to(x.device) / torch.norm(x, dim=2, keepdim=True)
    return x


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """ helper function to spherically interpolate two arrays, with "t" being the fraction
    of the arc to traverse between v0 and v1 """

    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().detach().numpy()
        v1 = v1.cpu().detach().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def get_current_model():
    return CURRENT_MODEL


@functools.lru_cache()
def load_model_from_config(config, ckpt, device, verbose=False):
    
    global CURRENT_MODEL
    
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()

    model = model.to(device)

    CURRENT_MODEL = model

    return model


@functools.lru_cache()
def _load_safety_model(safety_model_id="CompVis/stable-diffusion-safety-checker"):
    safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)
    return safety_feature_extractor, safety_checker


def check_for_nsfw(x_samples):
    x_image = x_samples.cpu().permute(0, 2, 3, 1).numpy()
    safety_feature_extractor, safety_checker = _load_safety_model()
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    _, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    return has_nsfw_concept
