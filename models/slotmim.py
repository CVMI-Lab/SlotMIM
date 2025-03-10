import math
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision

def _build_mlp_dino(nlayers, in_dim, bottleneck_dim, hidden_dim=None):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim)
    else:
        layers = [nn.Linear(in_dim, hidden_dim)]
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        return nn.Sequential(*layers)

class DINOHead(nn.Module):
    def __init__(self, in_dim, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp_dino(nlayers, in_dim, bottleneck_dim, hidden_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        return x

class SemanticGrouping(nn.Module):
    def __init__(self, num_prototypes_patch, dim_prototype, enable_global=False, num_prototypes_slot=8192):
        super().__init__()
        self.num_prototypes_patch = num_prototypes_patch
        self.dim_prototype = dim_prototype

        self.prototypes_patch = nn.Embedding(num_prototypes_patch, dim_prototype)
        self.enable_global = enable_global
        if self.enable_global:
            self.num_prototypes_slot = num_prototypes_slot
            self.prototypes_slot = nn.Embedding(num_prototypes_slot, dim_prototype)
        
    def get_dots_patches(self, patches):
        prototypes = self.prototypes_patch.weight
        eps = 1e-6 if patches.dtype == torch.float16 else 1e-12
        dots = torch.einsum('kd,bld->bkl', 
            F.normalize(prototypes, dim=-1, eps=eps), 
            F.normalize(patches, dim=-1, eps=eps))
        return dots

    def get_dots_slots(self, slots):
        prototypes = self.prototypes_slot.weight
        eps = 1e-6 if slots.dtype == torch.float16 else 1e-12
        dots = torch.einsum('kd,bd->bk', 
            F.normalize(prototypes, dim=-1, eps=eps), 
            F.normalize(slots, dim=-1, eps=eps))
        return dots

    def forward(self, key, val=None, temp=0.07, onehot=False):
        val = key if val is None else val
        dots = self.get_dots_patches(key)
        
        eps = 1e-6 if key.dtype == torch.float16 else 1e-12
        if not onehot:
            attn = (dots / temp).softmax(dim=1) + eps
        else:
            attn = torch.zeros_like(dots).scatter_(1, dots.argmax(1, keepdim=True), 1) + eps
        slots = torch.einsum('bld,bkl->bkd', val, attn / attn.sum(dim=2, keepdim=True))
        return slots, dots

class SlotMIM(nn.Module):
    def __init__(self, encoder, args):
        super().__init__()

        self.dim_hidden = args.dim_hidden
        self.dim_hidden_slot = args.dim_hidden_slot
        self.dim_out = args.dim_out
        self.teacher_momentum = args.teacher_momentum
        self.group_loss_weight = args.group_loss_weight
        self.use_sinkhorn = args.use_sinkhorn
        self.use_cross_patch_loss = args.use_cross_patch_loss
        self.use_masked_patch_loss = args.use_masked_patch_loss
        self.mask_loss_weight = args.mask_loss_weight
        self.cross_loss_weight = args.cross_loss_weight

        assert 'vit' in args.arch, 'Only vit is supported'
        self.encoder_q = encoder(head_type='early_return', mask_im_modeling=True, drop_path_rate=args.drop_path_rate)
        self.encoder_k = encoder(head_type='early_return')
        self.num_channels = self.encoder_q.num_features

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.student_temp = args.student_temp
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(args.warmup_teacher_temp,
                        args.teacher_temp, args.warmup_teacher_temp_epochs),
            np.ones(args.epochs - args.warmup_teacher_temp_epochs) * args.teacher_temp
        ))
            
        self.projector_q = DINOHead(self.num_channels, hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out)
        self.projector_k = DINOHead(self.num_channels, hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out)

        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.num_prototypes = args.num_prototypes
        self.center_momentum = args.center_momentum
        self.register_buffer("center", torch.zeros(1, self.num_prototypes))

        self.grouping_q = SemanticGrouping(self.num_prototypes, self.dim_out)
        self.grouping_k = SemanticGrouping(self.num_prototypes, self.dim_out)
        self.predictor_slot = DINOHead(self.dim_out, nlayers=2, hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out)

        for param_q, param_k in zip(self.grouping_q.parameters(), self.grouping_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.K = int(args.num_instances * 1. / args.world_size / args.batch_size * args.epochs)
        self.k = int(args.num_instances * 1. / args.world_size / args.batch_size * (args.start_epoch - 1))

    def re_init(self, args):
        self.k = int(args.num_instances * 1. / args.world_size / args.batch_size * (args.start_epoch - 1))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        momentum = 1. - (1. - self.teacher_momentum) * (math.cos(math.pi * self.k / self.K) + 1.) * 0.5
        self.k += 1
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
        for param_q, param_k in zip(self.grouping_q.parameters(), self.grouping_k.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum) 

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        teacher_output = teacher_output.float()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        Q = torch.exp(teacher_output / teacher_temp).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def self_distill(self, q, k, sinkhorn=False):
        q = F.log_softmax(q / self.student_temp, dim=-1)
        if not sinkhorn:
            k = F.softmax((k - self.center) / self.teacher_temp, dim=-1)
        else:
            k = self.sinkhorn_knopp_teacher(k, self.teacher_temp)
        return torch.sum(-k * q, dim=-1).mean()

    @torch.no_grad()
    def get_slot_mask(self, score):
        return (torch.zeros_like(score).scatter_(1, score.argmax(1, keepdim=True), 1).sum(-1) > 0).long().detach()

    def ctr_loss_filtered(self, q, k, mask_q, mask_k, tau=0.2):
        eps = 1e-6 if q.dtype == torch.float16 else 1e-12
        q = q.flatten(0, 1)
        k = F.normalize(k.flatten(0, 1), dim=1, eps=eps)

        mask_intersection = (mask_q * mask_k).view(-1)
        idxs_q = mask_intersection.nonzero().squeeze(-1)

        mask_k = concat_all_gather(mask_k.view(-1))
        idxs_k = mask_k.nonzero().squeeze(-1)

        N = k.shape[0]
        logits = torch.einsum('nc,mc->nm', [F.normalize(self.predictor_slot(q[idxs_q]), dim=1, eps=eps), concat_all_gather(k)[idxs_k]]) / tau
        labels = mask_k.cumsum(0)[idxs_q + N * torch.distributed.get_rank()] - 1
        return F.cross_entropy(logits, labels) * (2 * tau)

    def self_distill_slot(self, q, k, mask_q, mask_k, sinkhorn=False):
        mask_intersection = (mask_q * mask_k).view(-1)
        idxs = mask_intersection.nonzero().squeeze(-1)
        scores_q = self.grouping_q.get_dots_slots(q.flatten(0, 1)[idxs])
        scores_k = self.grouping_k.get_dots_slots(k.flatten(0, 1)[idxs])
        logits_q = F.log_softmax(scores_q / self.student_temp, dim=-1)
        if not sinkhorn:
            logits_k = F.softmax((scores_k - self.center_slot) / self.teacher_temp_slot, dim=-1)
        else:
            logits_k = self.sinkhorn_knopp_teacher(scores_k, self.teacher_temp_slot)
        return torch.sum(-logits_k * logits_q, dim=-1).mean(), scores_k

    def invaug(self, x, coords, flags):
        if len(x.shape) == 3:
            x = x.reshape(x.shape[0], x.shape[1], int(x.shape[2] ** 0.5), int(x.shape[2] ** 0.5))
        N, C, H, W = x.shape

        batch_idxs = torch.arange(N, device=coords.device).view(N, 1)
        coords_rescaled = coords.clone()
        coords_rescaled[:, 0] = coords_rescaled[:, 0] * W  # x1
        coords_rescaled[:, 2] = coords_rescaled[:, 2] * W  # x2
        coords_rescaled[:, 1] = coords_rescaled[:, 1] * H  # y1
        coords_rescaled[:, 3] = coords_rescaled[:, 3] * H  # y2
        coords_with_idxs = torch.cat([batch_idxs, coords_rescaled], dim=1)
        
        x_aligned = torchvision.ops.roi_align(x, coords_with_idxs, (H, W), aligned=True)
        x_flipped = torch.stack([feat.flip(-1) if flag else feat for feat, flag in zip(x_aligned, flags)])
        return x_flipped

    def forward(self, crops, coords, flags, masks, epoch):
        self.teacher_temp = self.teacher_temp_schedule[epoch]
        losses = {}

        if len(masks[0].shape) == 3:
            masks[0], masks[1] = masks[0].flatten(1), masks[1].flatten(1)

        x1, x2 = self.encoder_q(crops[0], masks[0]), self.encoder_q(crops[1], masks[1])
        x1_proj, x2_proj = self.projector_q(x1), self.projector_q(x2)
        
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            y1, y2 = self.encoder_k(crops[0]), self.encoder_k(crops[1])
            y1_proj, y2_proj = self.projector_k(y1), self.projector_k(y2)
        
        (q1, score_q1), (q2, score_q2) = self.grouping_q(x1_proj, x1_proj, self.teacher_temp), self.grouping_q(x2_proj, x2_proj, self.teacher_temp)
        if self.use_cross_patch_loss:
            q1_aligned = self.invaug(score_q1, coords[0], flags[0]).permute(0, 2, 3, 1).flatten(0, 2)
            q2_aligned = self.invaug(score_q2, coords[1], flags[1]).permute(0, 2, 3, 1).flatten(0, 2)
        if self.use_masked_patch_loss:
            q1_masked = score_q1.transpose(1, 2).flatten(0, 1)[masks[0].flatten().nonzero().squeeze(-1)]
            q2_masked = score_q2.transpose(1, 2).flatten(0, 1)[masks[1].flatten().nonzero().squeeze(-1)]
        with torch.no_grad():
            (k1, score_k1), (k2, score_k2) = self.grouping_k(y1_proj, y1_proj, self.teacher_temp), self.grouping_k(y2_proj, y2_proj, self.teacher_temp)
            if self.use_cross_patch_loss:
                k1_aligned = self.invaug(score_k1, coords[0], flags[0]).permute(0, 2, 3, 1).flatten(0, 2)
                k2_aligned = self.invaug(score_k2, coords[1], flags[1]).permute(0, 2, 3, 1).flatten(0, 2)
            if self.use_masked_patch_loss:
                k1_masked = score_k1.transpose(1, 2).flatten(0, 1)[masks[0].flatten().nonzero().squeeze(-1)]
                k2_masked = score_k2.transpose(1, 2).flatten(0, 1)[masks[1].flatten().nonzero().squeeze(-1)]
        
        slot_mask_q1, slot_mask_q2 = self.get_slot_mask(score_q1), self.get_slot_mask(score_q2)
        slot_mask_k1, slot_mask_k2 = self.get_slot_mask(score_k1), self.get_slot_mask(score_k2)
        if self.use_cross_patch_loss:
            patch_loss_cross = self.self_distill(q1_aligned, k2_aligned, self.use_sinkhorn) \
                            + self.self_distill(q2_aligned, k1_aligned, self.use_sinkhorn)
            losses['patch_loss_cross'] = patch_loss_cross * self.cross_loss_weight * self.group_loss_weight * .5
        
        if self.use_masked_patch_loss:
            patch_loss_masked = self.self_distill(q1_masked, k1_masked, self.use_sinkhorn) \
                            + self.self_distill(q2_masked, k2_masked, self.use_sinkhorn)
            losses['patch_loss_masked'] = patch_loss_masked * self.mask_loss_weight * self.group_loss_weight * .5
        
        nan_detected = torch.tensor(torch.isnan(losses['patch_loss_masked']) or torch.isnan(losses['patch_loss_cross']), device=q1.device)
        dist.all_reduce(nan_detected, op=dist.ReduceOp.MAX)
        if not nan_detected:
            self.update_center(torch.cat([score_k1, score_k2]).transpose(1, 2).flatten(0, 1))
        
        if self.group_loss_weight < 1:
            if not nan_detected:
                ctr_slot_loss = self.ctr_loss_filtered(q1, k2, slot_mask_q1, slot_mask_k2) \
                        + self.ctr_loss_filtered(q2, k1, slot_mask_q2, slot_mask_k1)
                losses['ctr_slot_loss'] = ctr_slot_loss * (1. - self.group_loss_weight) * .5
            else:
                losses['ctr_slot_loss'] = torch.tensor(0.0, device=q1.device)

        n_slot1, n_slot2 = slot_mask_k1.sum(-1).float().mean(), slot_mask_k2.sum(-1).float().mean()
        losses['n_slot'] = (n_slot1 + n_slot2) * .5
        
        return losses
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    @torch.no_grad()
    def update_center_slot(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center_slot = self.center_slot * self.center_momentum + batch_center * (1 - self.center_momentum)
    
    @torch.no_grad()
    def forward_viz(self, x, shape=None):
        feats = self.encoder_k(x)
        slots, probs = self.grouping_k(self.projector_k(feats), feats)
        return probs

class SlotMIMEval(nn.Module):
    def __init__(self, encoder, args):
        super().__init__()

        self.dim_hidden = args.dim_hidden
        self.dim_out = args.dim_out

        self.encoder_k = encoder(head_type=args.head_type, drop_path_rate=args.drop_path_rate)
        self.num_channels = self.encoder_k.num_features

        self.projector_k = DINOHead(self.num_channels, hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out)

        self.num_prototypes = args.num_prototypes
        self.grouping_k = SemanticGrouping(self.num_prototypes, self.dim_out)

    def forward_viz(self, x, shape=None):
        with torch.no_grad():
            feats = self.encoder_k(x)
            slots, probs = self.grouping_k(self.projector_k(feats), feats)
            return probs


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
