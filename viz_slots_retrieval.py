import os
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from data.datasets import ImageFolder
from models import vision_transformer
from models.slotmim import SlotMIMEval

def denorm(img):
    mean, val = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
    img = (img * val[:, None, None] + mean[:, None, None]) * torch.tensor([255, 255, 255])[:, None, None]
    return img.permute(1, 2, 0).cpu().type(torch.uint8)

def get_model(args):
    encoder = vision_transformer.__dict__[args.arch]
    model = SlotMIMEval(encoder, args)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    weights = {k.replace('module.', ''):v for k, v in checkpoint['model'].items()}
    model.load_state_dict(weights, strict=False)
    model = model.eval()
    return model, checkpoint['epoch']

def get_features(model, dataset, bs):
    memory_loader = torch.utils.data.DataLoader(
        dataset, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    bank = []
    for data in tqdm(memory_loader, desc='Feature extracting', leave=False, disable=False):
        feature = model.projector_k(model.encoder_k(data.cuda(non_blocking=True)))#.mean(dim=(-2, -1))
        feature = F.normalize(feature, dim=2)
        bank.append(feature)
    bank = torch.cat(bank, dim=0)
    return bank

def prepare_knn(model, dataset, args):
    prototypes = F.normalize(model.grouping_k.prototypes_patch.weight, dim=1) # k x d
    memory_bank = get_features(model, dataset, args.batch_size) # n x l x d
    dots = torch.einsum('kd,nld->nkl', [prototypes, memory_bank]) # n x k x l
    masks = torch.zeros_like(dots).scatter_(1, dots.argmax(1, keepdim=True), 1)
    masks_adder = masks + 1.e-6
    scores = (dots * masks_adder).sum(-1) / masks_adder.sum(-1) # n x k
    _, idxs = scores.t().topk(dim=1, k=args.topk)
    dots = dots.reshape(dots.shape[0], dots.shape[1], int(dots.shape[2] ** 0.5), int(dots.shape[2] ** 0.5))
    return dots, idxs

def viz_slots(dataset, dots, idxs, slot_idxs, save_path, args):
    color = np.array([255, 0, 0]).reshape(1, 1, 3)
    fig, ax = plt.subplots(args.topk, len(slot_idxs), figsize=(len(slot_idxs)*2, args.topk*2), squeeze=False, dpi=args.dpi)
    
    for i, slot_idx in enumerate(tqdm(slot_idxs, desc='KNN retreiving', leave=False, disable=False)):
        # ax[0, i].set_title(i)
        for j in range(args.topk):
            idx = idxs[slot_idx, j]
            image = denorm(dataset[idx]).numpy()
            pred = TF.resize(dots[idx], image.shape[:2], TF.InterpolationMode.BICUBIC)
            mask = torch.zeros_like(pred).scatter_(0, pred.argmax(0, keepdim=True), 1)
            mask = mask[slot_idx].unsqueeze(-1).cpu().numpy()
            image = np.int32((args.alpha * (image * mask) + (1 - args.alpha) * color * mask) + (image * (1 - mask)))
            ax[j, i].imshow(image)
            ax[j, i].axis('off')
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')

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

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # viz-related
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--dpi', type=int, default=100)
    parser.add_argument('--sampling', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--idxs', type=list, default=[])
    parser.add_argument('--save_path', type=str, default='viz_slots.jpg')
    # dataset
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--dataset', type=str, default='COCOval', help='dataset type')
    parser.add_argument('--data_dir', type=str, default='datasets/coco', help='dataset directory')
    parser.add_argument('--batch_size', type=int, default=64)
    # Model.
    parser.add_argument('--model_path', type=str, default='output/slotmim_coco_vitb16_800ep/current.pth')
    parser.add_argument('--dim_hidden', type=int, default=2048)
    parser.add_argument('--dim_out', type=int, default=256)
    parser.add_argument('--arch', type=str, default='vit_base_patch16')
    parser.add_argument('--num_prototypes', type=int, default=256)
    # below are legacy arguments, just keep them
    parser.add_argument('--head-type', type=str, default='early_return', help='choose head type')
    parser.add_argument('--drop-path-rate', type=float, default=0, help="stochastic depth rate")
    parser.add_argument('--use-bn-in-head', type=bool_flag, default=False, help='use batch norm in head')
    args = parser.parse_args()

    mean_vals, std_vals = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=mean_vals, std=std_vals)])

    dataset = ImageFolder(args.dataset, args.data_dir, transform)
    model, epoch = get_model(args)
    model = model.cuda()

    np.random.seed(args.seed)
    dots, idxs = prepare_knn(model, dataset, args)
    if args.sampling > 0:
        slot_idxs = np.random.randint(0, args.num_prototypes, args.sampling)
    elif len(args.idxs) > 0:
        slot_idxs = args.idxs
    else:
        slot_idxs = range(args.num_prototypes)
    
    if not os.path.exists(os.path.join(os.path.dirname(args.model_path), 'visualizations')):
        os.mkdir(os.path.join(os.path.dirname(args.model_path), 'visualizations'))
    
    save_path = os.path.join(os.path.dirname(args.model_path), 'visualizations', 'viz_slots_ep{}.jpg'.format(epoch))
    viz_slots(dataset, dots, idxs, slot_idxs, save_path, args)
