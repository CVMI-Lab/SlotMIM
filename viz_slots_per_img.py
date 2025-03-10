import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

from models import vision_transformer
from models.slotmim import SlotMIMEval


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

def denorm(img):
    mean, val = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
    img = (img * val[:, None, None] + mean[:, None, None]) * torch.tensor([255, 255, 255])[:, None, None]
    return img.permute(1, 2, 0).cpu().type(torch.uint8)

def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image

def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = fig.add_axes([0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = len(mask)
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return

def get_model(args):
    encoder = vision_transformer.__dict__[args.arch]
    model = SlotMIMEval(encoder, args)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    weights = {k.replace('module.', ''):v for k, v in checkpoint['model'].items()}
    model.load_state_dict(weights, strict=False)
    model = model.eval()
    return model, checkpoint['epoch']

def get_attns(model, img, temp=0.07):
    dots = model.forward_viz(img.cuda())
    attns = (dots / temp).softmax(dim=1)
    if len(attns.shape) == 4:
        scores = attns.sum(-1).sum(-1)
    else:
        scores = attns.sum(-1)
    return attns, scores

def filter_attentions(attentions, scores, threshold):
    filtered_attns = []
    filtered_scores = []
    for attn, score in zip(attentions, scores):
        if score > threshold:
            filtered_attns.append(attn)
            filtered_scores.append(score.item())
    return filtered_attns, filtered_scores

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # viz-related
    parser.add_argument('--dpi', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.6, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    parser.add_argument('--score_threshold', type=float, default=1.0, help='Minimum score for a slot to be visualized')
    # Data.
    parser.add_argument("--image_size", default=(960, 960), type=int, nargs="+", help="Resize image.")
    parser.add_argument("--image_path", default='./imgs/', type=str, help="Path of the image to load.")
    parser.add_argument("--output_dir", default='./tmp', help='Path where to save visualizations.')
    # Model.
    parser.add_argument('--model_path', type=str, default='output/slotcon_cocoplus_vitb16_800ep_lr1.5e-4+norm_oriproj/ckpt_epoch_800.pth')
    parser.add_argument('--dim_hidden', type=int, default=2048)
    parser.add_argument('--dim_out', type=int, default=256)
    parser.add_argument('--arch', type=str, default='vit_base_patch16')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--num_prototypes', type=int, default=512)
    parser.add_argument('--head-type', type=str, default='early_return', help='choose head type')
    parser.add_argument('--drop-path-rate', type=float, default=0, help="stochastic depth rate")
    parser.add_argument('--use-bn-in-head', type=bool_flag, default=False, help='use batch norm in head')
    parser.add_argument('--use-slot-decoder', type=bool_flag, default=False, help='use slot decoder')
    parser.add_argument('--decoder-depth', type=int, default=8, help='number of decoder layers')
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    model, epoch = get_model(args)
    model.to(device)

    # open image
    if args.image_path is None:
        # user has not specified any image - we use our own image
        print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
        print("Since no image path have been provided, we take the first image in our paper.")
        response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
        imgs = [img]
        img_paths = ["https://dl.fbaipublicfiles.com/dino/img.png"]
    elif os.path.isfile(args.image_path):
        with open(args.image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            imgs = [img]
            img_paths = [args.image_path]
    elif os.path.isdir(args.image_path):
        img_paths = [os.path.join(args.image_path, f) for f in os.listdir(args.image_path) if f.endswith(('png', 'jpg', 'jpeg'))]
        imgs = [Image.open(img_path).convert('RGB') for img_path in img_paths]
    else:
        print(f"Provided image path {args.image_path} is non valid.")
        sys.exit(1)
    transform = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    for img, img_path in zip(imgs, img_paths):
        img = transform(img)

        # make the image divisible by the patch size
        w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
        img = img[:, :w, :h].unsqueeze(0)

        w_featmap = img.shape[-2] // args.patch_size
        h_featmap = img.shape[-1] // args.patch_size

        attentions, scores = get_attns(model, img, temp=0.07)

        # Filter attentions based on the score threshold
        filtered_attentions, filtered_scores = filter_attentions(attentions[0], scores[0], args.score_threshold)
        nh = len(filtered_attentions)  # number of filtered heads
        filtered_attentions = torch.stack(filtered_attentions).reshape(nh, -1)

        if args.threshold is not None:
            # we keep only a certain percentage of the mass
            val, idx = torch.sort(filtered_attentions)
            val /= torch.sum(val, dim=1, keepdim=True)
            cumval = torch.cumsum(val, dim=1)
            th_attn = cumval > (1 - args.threshold)
            idx2 = torch.argsort(idx)
            for head in range(nh):
                th_attn[head] = th_attn[head][idx2[head]]
            th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
            # interpolate
            th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()
        
        filtered_attentions = filtered_attentions.reshape(nh, w_featmap, h_featmap)
        filtered_attentions = nn.functional.interpolate(filtered_attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

        # save attentions heatmaps
        output_dir = os.path.join(args.output_dir, os.path.basename(img_path).split('.')[0], os.path.basename(os.path.dirname(args.model_path)))
        os.makedirs(output_dir, exist_ok=True)
        torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(output_dir, "img.png"))
        for j, (attn, score) in enumerate(zip(filtered_attentions, filtered_scores)):
            score = (score / scores.sum()).item()
            fname = os.path.join(output_dir, f"attn-head{j}-score{round(score*100)}%.png")
            plt.imsave(fname=fname, arr=attn, format='png')
            print(f"{fname} saved.")

        if args.threshold is not None:
            image = skimage.io.imread(os.path.join(output_dir, "img.png"))
            for j, (attn, score) in enumerate(zip(filtered_attentions, filtered_scores)):
                score = (score / scores.sum()).item()
                attn = attn[None, :, :]
                display_instances(image, attn, fname=os.path.join(output_dir, f"mask_th{args.threshold}_head{j}_score{round(score*100)}%.png"), blur=False)
            display_instances(image, filtered_attentions, fname=os.path.join(output_dir, f"mask_th{args.threshold}_all.png"), blur=False)
            black_image = np.zeros_like(image)
            display_instances(black_image, filtered_attentions, fname=os.path.join(output_dir, f"mask_th{args.threshold}_all_black.png"), blur=False, contour=False, alpha=1)
