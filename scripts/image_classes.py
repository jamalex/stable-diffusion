import argparse, os, sys, glob
import base64
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
import skvideo.io
import subprocess
import tempfile
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from IPython.display import display, HTML, Video
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
import numpy as np
import warnings
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from utils import get_current_model
from attrdict import AttrDict

import functools

warnings.filterwarnings("ignore", category=DeprecationWarning, message=r"tostring\(\) is deprecated\. Use tobytes\(\) instead\.")

IS_NOTEBOOK = False
try:
    from IPython import get_ipython
    if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
        IS_NOTEBOOK = True
except:
    pass

class SDImage:

    def __init__(self, image, metadata=None):
        self.image = image
        self.metadata = AttrDict(metadata.copy() if metadata else {})

    def update_metadata(self, **new_data):
        self.metadata.update(new_data)

    @classmethod
    def fromtorchgrid(cls, images, n_rows=None):
        """
        Convert torch image tensor(s) to a single PIL image grid.
        """
        if n_rows is None:
            n_rows = images.shape[0]
        grid = torch.stack(images, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=n_rows)
        return SDImageList.fromtorchlist([grid])[0]

    @classmethod
    def from_file(cls, path, target_width=None, target_height=None):
        image = Image.open(path).convert("RGB")
        w, h = target_width or image.size[0], target_height or image.size[1]
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        image = image.resize((w, h), resample=Image.LANCZOS)
        return cls(image)        

    def show(self, filename=None):
        if filename is None:
            filename = tempfile.mktemp(suffix='.png')
        self.image.save(filename)
        subprocess.Popen(["eog", filename])

    def save(self, filename=None, text=None, **kwargs):
        image = self.image
        if filename is None:
            filename = tempfile.mktemp(suffix='.png')
        if text is not None:
            image = image.copy()
            if isinstance(text, str):
                text = {"text": text}
            text_str = text["text"]
            color = text.get("color", (255, 0, 0))
            font = ImageFont.truetype("scripts/fonts/SpecialElite-YOGj.ttf", text.get("size", 20))
            ImageDraw.Draw(image).text((0, 0), text_str, font=font, fill=color)
        image.save(filename)
        return filename

    def __getattr__(self, name):
        return getattr(self.image, name)


class SDImageList(list):

    @classmethod
    def fromfilelist(cls, filelist, target_width=None, target_height=None):
        return cls([SDImage.from_file(path, target_width, target_height) for path in filelist])

    @classmethod
    def fromdirectory(cls, directory, pattern="*.png", target_width=None, target_height=None):
        filelist = sorted(glob.glob(os.path.join(directory, pattern)))
        return cls.fromfilelist(filelist, target_width, target_height)

    @classmethod
    def fromvideo(cls, filename, target_width=None, target_height=None):
        """
        Convert a video to a list of images.
        """
        video = skvideo.io.vread(filename)
        
        # crop the video to the center, making it square
        h, w = video.shape[1:3]
        if h > w:
            left = (h - w) // 2
            right = left + w
            video = video[:, left:right, :]
        elif w > h:
            top = (w - h) // 2
            bottom = top + h
            video = video[:, :, top:bottom]
        
        images = []
        for frame in video:
            image = Image.fromarray(frame)
            w, h = target_width or image.size[0], target_height or image.size[1]
            if w != image.size[0] or h != image.size[1]:
                image = image.resize((w, h), resample=Image.LANCZOS)
            images.append(SDImage(image))

        return cls(images)

    @classmethod
    def fromtorchlist(cls, images, **kwargs):
        """
        Convert torch image tensor(s) to SDImageList.
        """
        pil_images = cls([])
        for x_samples in images:
            if len(x_samples.shape) == 3:
                x_samples = x_samples[None, ...]
            for x_sample in x_samples:
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                pil_images.append(SDImage(Image.fromarray(x_sample.astype(np.uint8))))
        return pil_images

    def togif(self, filename=None, fps=10, loop=0):
        """
        Convert images to a gif.
        """
        if filename is None:
            filename = tempfile.mktemp(suffix='.gif')
        temp_dir = tempfile.mkdtemp()
        # self[0].save(filename, save_all=True, append_images=self[1:], duration=duration, loop=loop)
        self.save_to_dir(temp_dir)
        subprocess.call(f"ffmpeg -framerate {fps} -i {temp_dir}/%05d.png -c:v ffv1 -r 24 -y {filename}.avi".split(), stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        subprocess.call(f"ffmpeg -y -i {filename}.avi -vf palettegen {filename}-palette.png".split(), stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        subprocess.call(f"ffmpeg -y -i {filename}.avi -i {filename}-palette.png -lavfi paletteuse -loop {loop} {filename}".split(), stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        return filename

    def displaygif(self, fps=10, loop=0):
        """
        Display images as a gif.
        """
        filename = self.togif(fps=fps, loop=loop)
        # if we're in a notebook, display the gif inline
        if IS_NOTEBOOK:
            b64 = base64.b64encode(open(filename,'rb').read()).decode('ascii')
            return HTML(f'<img style="width: 512px; float: left; padding: 5px;" src="data:image/gif;base64,{b64}" />')
        else:
            subprocess.Popen(["eog", filename])

    def save_to_dir(self, dirpath, name_format="{i:05d}.png"):
        """
        Save images to a directory.
        """
        os.makedirs(dirpath, exist_ok=True)
        for i, image in enumerate(self):
            image.save(filename=os.path.join(dirpath, name_format.format(i=i, **image.metadata)))

    def tomp4(self, filename=None, fps=5):
        # convert list of PIL images into a 4-dimensional numpy array
        # with dimensions (n_images, height, width, n_channels)

        if filename is None:
            filename = f"temp_{time.time()}.mp4"

        image_array = np.stack([np.array(image) for image in self], axis=0)

        # save array to output file
        skvideo.io.vwrite(filename, image_array, outputdict={
            "-r": str(fps),
            "-c:v": "libx264",
            "-profile:v": "baseline",
            "-level": "3.0",
            "-preset": "veryslow",
            "-crf": "16",
            "-pix_fmt": "yuv420p"
        })

        return filename

    def displaymp4(self, fps=10):
        """
        Display a list of PIL images as an mp4.
        """
        filename = "scripts/temp.mp4"
        self.tomp4(fps=fps, filename=filename)
        if IS_NOTEBOOK:
            display(Video("temp.mp4", embed=True))
        else:
            subprocess.Popen(["mplayer", filename])
        return filename

    def __getitem__(self, val):
        if isinstance(val, int):
            return super().__getitem__(val)
        else:
            return SDImageList(super().__getitem__(val))

    def __add__(self, other):
        return SDImageList(super().__add__(other))

    def __reversed__(self):
        return SDImageList(super().__reversed__())

    def __mul__(self, other):
        return SDImageList(super().__mul__(other))

    def __rmul__(self, other):
        return SDImageList(super().__rmul__(other))


class SDImageListList(list):

    def displaygifs(self, duration=10, loop=0):
        """
        Display a list of SDImageLists as gifs.
        """
        return HTML(" ".join(SDImageList(images).displaygif(duration=duration, loop=loop).__html__() for images in self))

    def __getitem__(self, val):
        if isinstance(val, int):
            return super().__getitem__(val)
        else:
            return SDImageListList(super().__getitem__(val))


"""
img_peng = SDImage.from_file("peng.png")
lat_peng = c.encode_to_torch(img_peng)
img_rose = SDImage.from_file("rose.png")
lat_rose = c.encode_to_torch(img_rose)
frames = SDImageList([])
range_vals = list(np.arange(0.1, 0.8, 0.01))
for strength in range_vals:
    print("Running with strength", strength)
    results = c.txt2imgcls("a rose growing in the sand, left side of image, long shot", n_iter=1, n_samples=1, scale=10, steps=50, extract_intermediates=False, init_image=lat_peng + strength * lat_rose, strength=strength, sampler="ddim", seed=1661642081);
    frames.append(results["images"][0])
frames.pop()
# frames.displaygif(duration=800)
SDImageList(frames * 100).tomp4("temp.mp4", fps=3)
"""
