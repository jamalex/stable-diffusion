from matplotlib import pyplot as plt
plt.ion()

masks = SDImageList.fromdirectory("masks")
frames = SDImageList.fromdirectory("projects/vid2vid_test")

# c.generate("man with crazy hair and a beard wearing sunglasses, plain grey wall in background", x0=SDImage(frames[120].image), mask=masks[0], n_samples=1, n_iter=1, strength=0.6, steps=25, seed=221451).images[0].show()
# c.generate("man with crazy hair and a beard wearing sunglasses, plain grey wall in background", x0=SDImage(frames[20].image), mask=masks[0], n_samples=1, n_iter=1, strength=0.6, steps=25, seed=221451).images[0].show()

blackball = SDImage.from_file("masks/black_circle_ball_on_white.png")
whiteball = SDImage.from_file("masks/white_circle_ball_on_black.png")
allblack = SDImage.from_file("masks/allblack.png")
allwhite = SDImage.from_file("masks/allwhite.png")

img = SDImage(frames[30].image)
results = c.generate("man with crazy hair and a beard wearing sunglasses", sampler="ddim", x0=img, init_image=img, mask=blackball, n_samples=1, extract_intermediates=True, n_iter=1, strength=0.7, steps=50, seed=102340222)
results.images[0].show()
SDImageList(results.intermediates[0]).displaygif(fps=10)

out_frames = []
for frame in frames[20:40]:
    out_frame = c.generate("man with crazy hair and a beard wearing sunglasses", sampler="ddim", x0=frame, init_image=frame, mask=blackball, n_samples=1, extract_intermediates=False, n_iter=1, strength=0.8, steps=15, seed=102340222).images[0]
    out_frames.append(out_frame)
vid = SDImageList(out_frames)
vid.displaygif(fps=5)

img.show()

def show(x):
    img = c.decode_to_images(x)[0]
    img.show()

# show(mask)
# show(x0)

# ts = torch.full((temp_opts.n_samples,), 25, device=c.device, dtype=torch.long)
# img_orig = c.model.q_sample(x0, ts)
# img = img_orig * mask + (1. - mask) * x0


from matplotlib import pyplot as plt; plt.imshow(mask_dec.squeeze().mean(dim=0).detach().cpu()); plt.show()



mask = c.prep_input_image(blackball)
from matplotlib import pyplot as plt; plt.imshow((mask[0].squeeze().mean(dim=0).detach().cpu().numpy()).astype(np.uint8)); plt.show()


# show the decoded image
import os; os.chdir("scripts")
def show(x): from image_classes import SDImage, rearrange; x = img_dec = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0); from PIL import Image; SDImage(Image.fromarray((255. * rearrange(x.squeeze().detach().cpu().numpy(), "c w h -> w h c")).astype(np.uint8))).show()

# show the latent image
def show(self, x): from image_classes import SDImageList; torch_images = torch.clamp((self.model.decode_first_stage(x) + 1.0) / 2.0, min=0.0, max=1.0); SDImageList.fromtorchlist(torch_images)[0].show()

show(img_orig_dec); show(img_dec); show(img_masked_dec)


def visualize_latent_image(latent, index=0, cmap="seismic"):
    latent = latent[index].detach().cpu()
    minval, maxval = torch.min(latent), torch.max(latent)
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    im1 = axs[0, 0].imshow(latent[0], cmap=cmap, vmin=minval, vmax=maxval)
    im2 = axs[0, 1].imshow(latent[1], cmap=cmap, vmin=minval, vmax=maxval)
    im3 = axs[1, 0].imshow(-latent[2], cmap=cmap, vmin=minval, vmax=maxval)
    im4 = axs[1, 1].imshow(-latent[3], cmap=cmap, vmin=minval, vmax=maxval)
    plt.figure()
    plt.imshow(latent[0] + latent[1] - latent[2] - latent[3], cmap=cmap, vmin=minval, vmax=maxval)
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im1, cax=cbar_ax)

#visualize_latent_image(x0)
