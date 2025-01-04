import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange, repeat
import numpy as np
from gluefactory.models.base_model import BaseModel
from torch.nn.functional import cosine_similarity

@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def normalize_keypoints(kpts, size=None, shape=None):
    if size is None:
        assert shape is not None
        _, _, h, w = shape
        one = kpts.new_tensor(1)
        size = torch.stack([one * w, one * h])[None]

    shift = size.float() / 2
    scale = size.max(1).values.float() / 2  # actual SuperGlue mult by 0.7
    kpts = (kpts - shift[:, None]) / scale[:, None, None]
    return kpts


def masked_mean(value, weights):
    pos = (value * weights.clamp(min=0.0)).sum(-1) / weights.clamp(min=0.0).sum(
        -1
    ).clamp(min=1.0)
    neg = (value * weights.clamp(max=0.0)).sum(-1) / weights.clamp(max=0.0).sum(
        -1
    ).clamp(max=-1.0)
    return pos + neg


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, dim: int, F_dim=None, gamma=1.0, H_dim=0):
        """
        Learnable Fourier Features from https://arxiv.org/pdf/2106.02795.pdf
        """
        super().__init__()
        self.M = M
        self.F_dim = F_dim if F_dim is not None else dim
        self.H_dim = H_dim
        self.D = dim
        self.gamma = gamma
        self.add_mlp = H_dim is not None and H_dim > 0

        # Projection matrix on learned lines (used in eq. 2)
        self.Wr = nn.Linear(self.M, self.F_dim // 2, bias=False)
        # MLP (GeLU(F @ W1 + B1) @ W2 + B2 (eq. 6)
        if self.add_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(self.F_dim, self.H_dim, bias=True),
                nn.GELU(),
                nn.Linear(self.H_dim, self.D),
            )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x):
        B, N, M = x.shape
        # Step 1. Compute Fourier features (eq. 2)
        projected = self.Wr(x)
        cosines = torch.cos(projected)
        sines = torch.sin(projected)
        if not self.add_mlp:
            emb = torch.stack([cosines, sines], 0).unsqueeze(-2)
            return repeat(emb, "... n -> ... (n r)", r=2)
        else:
            F = 1 / np.sqrt(self.F_dim) * torch.cat([cosines, sines], dim=-1)
            # Step 2. Compute projected Fourier features (eq. 6)
            return self.mlp(F)


class FastAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim**-0.5

    def forward(self, q, k, v, mask=None):
        b, n, h, d = q.shape
        query, key, value = [x.permute(0, 2, 1, 3).contiguous() for x in [q, k, v]]

        context = F.scaled_dot_product_attention(query, key, value).to(q)

        context = context.permute(0, 2, 1, 3)
        return context


class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, flash=False, bias=True) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0
        self.head_dim = self.embed_dim // num_heads

        self.Wkv = nn.Linear(embed_dim, 2 * embed_dim, bias=bias)

        self.Wq = nn.Linear(embed_dim, embed_dim, bias=bias)

        attn = FastAttention
        self.inner_attn = attn(self.head_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, x, source=None, mask=None):
        if source is None:
            source = x  # self attn
        q = rearrange(self.Wq(x), "b s (h d) -> b s h d", h=self.num_heads)
        kv = self.Wkv(source)
        kv = rearrange(kv, "b s (h d two) -> b s h d two", two=2, h=self.num_heads)
        k, v = kv[..., 0], kv[..., 1]

        context = self.inner_attn(q, k, v)
        message = self.out_proj(rearrange(context, "b s h d -> b s (h d)"))
        return x + self.ffn(torch.cat([x, message], -1))


class OverlapPredictor(BaseModel):
    default_conf = {
        "n_layers": 4,
        "input_dim": 384,
        "descriptor_dim": 256,
        "num_heads": 4,
        "patch_size": 14,
        "flash": False,
        "dropout_prob": 0.,
        "n": 2
    }
    required_data_keys = ["descriptors0", "descriptors1", "keypoints0", "keypoints1"]

    def _init(self, conf):
        self.keypoint_encoder = nn.Sequential(
            nn.Linear(2, 2 * conf.descriptor_dim),
            nn.LayerNorm(2 * conf.descriptor_dim),
            nn.GELU(),
            nn.Dropout(conf.dropout_prob),
            nn.Linear(2 * conf.descriptor_dim, conf.descriptor_dim),
        ) if conf.dropout_prob != 0 else nn.Sequential(
            nn.Linear(2, 2 * conf.descriptor_dim),
            nn.LayerNorm(2 * conf.descriptor_dim),
            nn.GELU(),
            nn.Linear(2 * conf.descriptor_dim, conf.descriptor_dim),
        )
        self.input_proj = nn.Linear(conf.input_dim, conf.descriptor_dim)

    def _forward(self, data):
        """
            input:
                data["keypoints0"] - grid keypoints,
                data["descriptors0"] - patch descriptors, built from frozen backbones,
                data["global_descriptor0"] - cls tokens, built from frozen backbones,
                image sizes
            output:
                patch-level/global embeddings

        """
        pred = {}

        if "image_size0" in data.keys():
            image_size0 = data.get("image_size0")
            image_size1 = data.get("image_size1")
        else:
            image_size0 = data['view0']['image_size']
            image_size1 = data['view1']['image_size']

        kpts0 = normalize_keypoints(data["keypoints0"], size=image_size0) # B, N, 2 = 64, 256, 3
        kpts1 = normalize_keypoints(data["keypoints1"], size=image_size1)

        # reduce the dim of the patch-level descriptors, B, N, D, eg, to 256
        desc0, desc1 = self.input_proj(data["descriptors0"]), self.input_proj(data["descriptors1"])

        # DINOv2 patch tokens + encoded keypoints (patch centers)
        pred["desc0"] = desc0 + self.keypoint_encoder(kpts0)
        pred["desc1"] = desc1 + self.keypoint_encoder(kpts1)

        if "gt_labels" in data.keys(): pred["gt_labels"] = data["gt_labels"].squeeze()
        if "label_confs" in data.keys(): pred["label_confs"] = data["label_confs"].squeeze()
        if "neg_labels" in data.keys(): pred["neg_labels"] = data["neg_labels"].squeeze()

        return pred

    def loss(self, pred, data):
        losses = {}
        losses["total"] = 0
        contrastive_results = self.contrastive_loss_patches(pred)
        for key in contrastive_results.keys():
            losses[key] = contrastive_results[key]

        losses["total"] += losses["local_neg"]
        losses["total"] += losses["local_pos"]

        return losses, self.metrics(losses, data)

    def metrics(self, pred, data):

        return {'local_neg_sim': pred["local_neg_sim"], 'local_pos_sim': pred["local_pos_sim"]}

    def contrastive_loss_patches(self, pred, margin=1, cos=True, conf=False):
        """
            contrastive loss on N patches pairs on each image pair
            embed1, embed2 in a shape (B, N, D)
            gt_labels: the matched patches where at least one correspondence are matched, (B, N, N)
            label_confs: how many correspondences in the matched patches, confidence of the labels
            cls_token1, cls_token1: cls tokens in shape (B, D)
        """

        # patch difference, local, on B*256 patches, normalize the descriptors for each patch
        x0_normalized = torch.nn.functional.normalize(pred['desc0'], dim=2) # B, 256, 256
        x1_normalized = torch.nn.functional.normalize(pred['desc1'], dim=2)
        # distances between each patch in each image pair, (B, N, N), [-1, 1]
        local_cos_sim = cosine_similarity(x0_normalized.unsqueeze(2), x1_normalized.unsqueeze(1), dim=3) if cos==True else torch.nn.PairwiseDistance()(x0_normalized.unsqueeze(1), x1_normalized.unsqueeze(2))
        local_neg, local_pos, local_neg_sim, local_pos_sim = self.contrastive_loss(local_cos_sim, pred["gt_labels"], pred["neg_labels"], pred["label_confs"], margin, conf)

        return {
            "local_neg": local_neg,
            "local_pos": local_pos,
            'local_neg_sim':local_neg_sim,
            'local_pos_sim':local_pos_sim,
            }

    def contrastive_loss(self, similarities, gt_labels, neg_labels, label_confs=None, margin=1, conf=False):
        B = similarities.shape[0]
        # losses
        negative_loss = (neg_labels * torch.pow(similarities, self.conf.n))
        positive_loss = gt_labels * torch.pow(torch.clamp(margin - similarities, min=0.0), 2) * label_confs if conf \
                else gt_labels * torch.pow(torch.clamp(margin - similarities, min=0.0), 2)
        # similarities
        neg_sim = (neg_labels * similarities)
        pos_sim = (gt_labels * similarities)

        return negative_loss.view(B, -1).sum(-1)/(neg_labels.sum((-1, -2))+1e-4), \
                positive_loss.view(B, -1).sum(-1)/(gt_labels.sum((-1, -2))+1e-4), \
                neg_sim.view(B, -1).sum(-1)/(neg_labels.sum((-1, -2))+1e-4), \
                pos_sim.view(B, -1).sum(-1)/(gt_labels.sum((-1, -2))+1e-4)
