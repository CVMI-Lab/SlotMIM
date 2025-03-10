import colorsys
import argparse

import numpy as np
import torch
from torchvision import transforms
from torchvision import datasets as torchvision_datasets
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from skimage.measure import find_contours
from torch.nn.functional import interpolate
from tqdm import tqdm
from viz_slots_per_img import bool_flag, get_model, filter_attentions

def get_voc_dataset(voc_root=None):
    if voc_root is None:
        voc_root = "datasets/voc"  # path to VOCdevkit for VOC2012
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    def load_target(image):
        image = np.array(image)
        image = torch.from_numpy(image)
        return image

    target_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(load_target),
    ])

    dataset = torchvision_datasets.VOCSegmentation(root=voc_root, image_set="val", transform=data_transform,
                                                   target_transform=target_transform, download=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, drop_last=False)

    return dataset, data_loader


def get_per_sample_jaccard(pred, target):
    jac = 0
    object_count = 0
    for mask_idx in torch.unique(target):
        if mask_idx in [0, 255]:  # ignore index
            continue
        cur_mask = target == mask_idx
        intersection = (cur_mask * pred) * (cur_mask != 255)  # handle void labels
        intersection = torch.sum(intersection, dim=[1, 2])  # handle void labels
        union = ((cur_mask + pred) > 0) * (cur_mask != 255)
        union = torch.sum(union, dim=[1, 2])
        jac_all = intersection / union
        jac += jac_all.max().item()
        object_count += 1
    return jac / object_count


def run_eval(args, data_loader, model, device):
    model.to(device)
    model.eval()
    total_jac = 0
    image_count = 0
    for idx, (sample, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
        sample, target = sample.to(device), target.to(device)
        attention_mask = get_attention_masks(args, sample, model)
        jac_val = get_per_sample_jaccard(attention_mask, target)
        total_jac += jac_val
        image_count += 1
    return total_jac / image_count


def apply_mask_last(image, mask, color=(0.0, 0.0, 1.0), alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    image = image.permute(1, 2, 0).cpu().numpy()
    mask = mask.cpu().numpy()

    plt.ioff()
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]

    # Generate random colors

    def random_colors(N, bright=True):
        """
        Generate random colors.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        return colors

    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = (image * 255).astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            pass
            # _mask = cv2.blur(_mask, (10, 10))
        # Mask
        masked_image = apply_mask_last(masked_image, _mask, color, alpha)
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
    plt.close(fig)

def get_attns(model, img, shapes, temp=0.07):
    dots = model.forward_viz(img, shapes)
    attns = (dots / temp).softmax(dim=1)
    if len(attns.shape) == 4:
        scores = attns.sum(-1).sum(-1)
    else:
        scores = attns.sum(-1)
    return attns, scores

def get_attention_masks(args, image, model):
    # make the image divisible by the patch size
    w, h = image.shape[2] - image.shape[2] % args.patch_size, image.shape[3] - image.shape[3] % args.patch_size
    image = image[:, :w, :h]
    w_featmap = image.shape[-2] // args.patch_size
    h_featmap = image.shape[-1] // args.patch_size

    # ------------ FORWARD PASS -------------------------------------------
    attentions, scores = get_attns(model, image, (w_featmap, h_featmap), temp=0.07)

    # Filter attentions based on the score threshold
    attentions, filtered_scores = filter_attentions(attentions[0], scores[0], args.score_threshold)
    nh = len(attentions)  # number of filtered heads
    attentions = torch.stack(attentions).reshape(nh, -1)

    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cum_val = torch.cumsum(val, dim=1)
    th_attn = cum_val > (1 - args.threshold)
    idx2 = torch.argsort(idx)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]

    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
    # interpolate
    th_attn = interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0]

    return th_attn


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Unsupervised object discovery with LOST.")
    parser.add_argument("--threshold", type=float, default=.99, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    # Model.
    parser.add_argument('--model_path', type=str, default='../SlotCon/output/slotcon_cocoplus_vitb16_800ep_lr1.5e-4+norm_oriproj/ckpt_epoch_800.pth')
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
    parser.add_argument('--score_threshold', type=float, default=1, help='Minimum score for a slot to be visualized')

    args = parser.parse_args()

    # -------------------------------------------------------------------------------------------------------
    # Dataset

    dataset, loader = get_voc_dataset()

    # -------------------------------------------------------------------------------------------------------
    # Model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    model, epoch = get_model(args)
    model.to(device)

    args.model_name = args.model_path.split('/')[-2]

    model_accuracy = run_eval(args, loader, model, device)
    print(f"Jaccard index for {args.model_name}: {model_accuracy}")
    # save log
    with open("log.txt", "a") as f:
        f.write(f"{args.model_name}, {model_accuracy}\n")