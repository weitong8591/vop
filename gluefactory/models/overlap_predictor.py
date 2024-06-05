import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange, repeat
import numpy as np
import torchvision
from ..utils.patch_helper import PatchCollect

from gluefactory.models.base_model import BaseModel

from ..geometry.depth import dense_warp_consistency, dense_patch_matching

from ..geometry.utils import is_inside

from torch.nn.functional import cosine_similarity

from gluefactory.models.utils.metrics import matcher_metrics

from sklearn.metrics import confusion_matrix

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
        "bce_loss": False,
        "add_score_head": False,
        "add_voting_head":True,
        "add_cls_tokens": False,
        "attentions": False,
        "n": 2,
        "dropout_prob": 0.
    }
    required_data_keys = []

    def _init(self, conf):
        n = conf.n_layers
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

        if conf.add_score_head or conf.add_cls_tokens:
            self.score_head = torchvision.ops.MLP(
                conf.input_dim, [conf.descriptor_dim, 1]
            )

        self.self_attn = nn.ModuleList(
            [
                Transformer(conf.descriptor_dim, conf.num_heads, conf.flash)
                for _ in range(n)
            ]
        )

        self.cross_attn = nn.ModuleList(
            [
                Transformer(conf.descriptor_dim, conf.num_heads, conf.flash)
                for _ in range(n)
            ]
        )

        self.logits = nn.Linear(conf.descriptor_dim, 1)

        self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.patch_helper = PatchCollect(patch_size=14, resize_shape = 224)

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
        if "image_size0" in data.keys():
            image_size0 = data.get("image_size0")
            image_size1 = data.get("image_size1")
        else:
            image_size0 = data['view0']['image_size']
            image_size1 = data['view1']['image_size']
        kpts0 = normalize_keypoints(data["keypoints0"], size=image_size0) # B, N, 2 = 64, 256, 3
        kpts1 = normalize_keypoints(data["keypoints1"], size=image_size1)
        desc0, desc1 = data["descriptors0"], data["descriptors1"] # B, N, D = 64, 256, 1024, N is the number of local patches
        # score over patch-level descriptors
        if self.conf.add_score_head:
            keypoint_logits0, keypoint_logits1 = self.score_head(
                desc0
            ), self.score_head(desc1)  # B, N, 1

        if self.conf.add_cls_tokens:
            cls_token0 = data["global_descriptor0"]
            cls_token1 = data["global_descriptor1"]
            desc0 = torch.concat((desc0, cls_token0[:, None, :]), dim=1)
            desc1 = torch.concat((desc1, cls_token1[:, None, :]), dim=1) # B, 256+1, D
            
        # reduce the dim of the patch-level descriptors, B, N, D, eg, to 256
        desc0, desc1 = self.input_proj(desc0), self.input_proj(desc1)
        
        if self.conf.add_cls_tokens:
            desc0 = desc0 + torch.concat((torch.zeros_like(desc0[:, None, 1, :]), self.keypoint_encoder(kpts0)), dim=1)
            desc1 = desc1 + torch.concat((torch.zeros_like(desc1[:, None, 1, :]), self.keypoint_encoder(kpts1)), dim=1)
        else:
            desc0 = desc0 + self.keypoint_encoder(kpts0)
            desc1 = desc1 + self.keypoint_encoder(kpts1)
            
        # return the projected desc0 desc1 here to do the patch voting
        embeddings = [desc0, desc1]
        
        # attention layers
        for i in range(self.conf.n_layers):
            desc0, desc1 = self.self_attn[i](desc0), self.self_attn[i](desc1)
            desc0, desc1 = self.cross_attn[i](desc0, desc1), self.cross_attn[i](
                desc1, desc0
            )
        # import pdb; pdb.set_trace() 
        # attentions over each patch with the patches in image 2   
        if self.conf.attentions:
            attentions = [desc0, desc1]
        
        if self.conf.add_cls_tokens:
            logits0, logits1 = self.logits(desc0[:, 1:, ]), self.logits(desc1[:, 1:, ]) 
        else:
            logits0, logits1 = self.logits(desc0), self.logits(desc1) # B, N, 1 
        
        # normalization
        scores0 = torch.sigmoid(logits0).squeeze(-1)
        scores1 = torch.sigmoid(logits1).squeeze(-1)

        pred = {
            "logits0": logits0.squeeze(-1),
            "logits1": logits1.squeeze(-1),
            "matching_scores0": scores0,
            "matching_scores1": scores1,
            "overlap0": scores0.mean(-1),
            "overlap1": scores1.mean(-1),
        }

        if self.conf.add_score_head:
            pred["keypoint_logits0"] = keypoint_logits0.squeeze(-1)
            pred["keypoint_logits1"] = keypoint_logits1.squeeze(-1)
            pred["keypoint_scores0"] = torch.sigmoid(keypoint_logits0).squeeze(-1)
            pred["keypoint_scores1"] = torch.sigmoid(keypoint_logits1).squeeze(-1)

        if self.conf.add_voting_head:
            pred["desc0"] = embeddings[0]
            pred["desc1"] = embeddings[1]
        if self.conf.attentions:
            pred["attention0"] = attentions[0]
            pred["attention1"] = attentions[1]

        return pred

    def loss(self, pred, data):
        if "depth0" not in data.keys():
            depth0 = data['view0']['depth']
            depth1 = data['view1']['depth']
        else:
            depth0 = data["depth0"]
            depth1 = data["depth1"]
            # data["depth0"] = data['view0']['depth']
            # data["depth1"] = data['view1']['depth']
            # data["camera0"] = data['view0']['camera']
            # data["camera1"] = data['view1']['camera']
            # data["image_size0"] = data['view0']['image_size']
            # data["image_size1"] = data['view1']['image_size']

        pred["gt_valid0"] = is_inside(pred["keypoints0"], data["image_size0"]).float()
        pred["gt_valid1"] = is_inside(pred["keypoints1"], data["image_size1"]).float()

        losses = {}
        losses["total"] = 0
        # import pdb; pdb.set_trace()
        if self.conf.add_voting_head:
            contrastive_results = self.contrastive_loss_patches(pred["desc0"], pred["desc1"], data["gt_labels"].squeeze(), data["label_confs"].squeeze())
            
            if self.conf.add_cls_tokens:
                losses["global_neg"], losses["global_pos"] = contrastive_results["global_neg"], contrastive_results["global_pos"]
                pred["global_neg_sim"], pred["global_pos_sim"] = contrastive_results["global_neg_sim"], contrastive_results["global_pos_sim"]
                losses["total"] += losses["global_neg"]
                losses["total"] += losses["global_pos"]
                       
            losses["local_neg"], losses["local_pos"] = contrastive_results["local_neg"], contrastive_results["local_pos"]
            pred["local_neg_sim"], pred["local_pos_sim"] = contrastive_results["local_neg_sim"], contrastive_results["local_pos_sim"]
            losses["total"] += losses["local_neg"]
            losses["total"] += losses["local_pos"]

        if self.conf.bce_loss:
            l0 = self.criterion(pred["logits0"], data["gt_visible0"].squeeze().float())
            l1 = self.criterion(pred["logits1"], data["gt_visible1"].squeeze().float())
            losses["bce"] = masked_mean(l0, pred["gt_valid0"]) + masked_mean(
            l1, pred["gt_valid1"]
        )
            losses["total"] += losses["bce"]

        if self.conf.add_score_head:
            pred["has_depth0"] = F.max_pool2d((depth0 > 0).float(), 14).flatten(
                -2
            )
            pred["has_depth1"] = F.max_pool2d((depth1 > 0).float(), 14).flatten(
                -2
            )
            kpl0 = self.criterion(pred["keypoint_logits0"], pred["has_depth0"].float())
            kpl1 = self.criterion(pred["keypoint_logits1"], pred["has_depth1"].float())

            losses["keypoint_loss"] = masked_mean(
                kpl0, pred["gt_valid0"]
            ) + masked_mean(kpl1, pred["gt_valid1"])
            losses["total"] += losses["keypoint_loss"]
        if self.conf.attentions:
            attention_losses = self.contrastive_loss_attentions(pred["attention0"], pred["attention1"], data["gt_labels"].squeeze(), data["label_confs"].squeeze())
            if self.conf.add_cls_tokens:
                losses["attention_global_neg"], losses["attention_global_pos"] = attention_losses["global_neg"], attention_losses["global_pos"]
                pred["attention_global_neg_sim"], pred["attention_global_pos_sim"] = attention_losses["global_neg_sim"], attention_losses["global_pos_sim"]
                losses["total"] += losses["attention_global_neg"]
                losses["total"] += losses["attention_global_pos"]
                
            losses["attention_local_neg"] = attention_losses["local_neg"]
            losses["attention_local_pos"] = attention_losses["local_pos"]
            pred["attention_local_neg_sim"], pred["attention_local_pos_sim"] = attention_losses["local_neg_sim"], attention_losses["local_pos_sim"]

            losses["total"] += losses["attention_local_neg"]
            losses["total"] += losses["attention_local_pos"]

        # todo: loss(rank of reference images to each query, by the number of pred["gt_visible0"], /countings of the intersts of patches)
        # gt_interest_patches = pred["gt_visible0"].sum(-1)
        # min(pred["gt_visible0"].sum(-1), pred["gt_visible1"].sum(-1) )
        # min(pred["overlap0"].sum(-1), pred["overlap1"].sum(-1) )

        return losses, self.metrics(pred, data)

    def metrics(self, pred, data):
        
        if self.conf.add_voting_head:
            
            if self.conf.add_cls_tokens:
                if self.conf.attentions:
                    return {
                    'global_neg_sim': pred["global_neg_sim"],
                    'global_pos_sim': pred["global_pos_sim"],
                    'local_neg_sim': pred["local_neg_sim"],
                    'local_pos_sim': pred["local_pos_sim"],
                    "attention_global_neg_sim": pred["attention_global_neg_sim"],
                    "attention_global_pos_sim": pred["attention_global_pos_sim"],
                    "attention_local_neg_sim" : pred["attention_local_neg_sim"],
                    "attention_local_pos_sim": pred["attention_local_pos_sim"],
                    # 'tpr': pred["tpr"], 
                    # 'fpr': pred["fpr"],
                    # 'tpr_local': pred["tpr_local"],
                    # 'fpr_local': pred["fpr_local"]
                    }
                else:
                    return {
                    'global_neg_sim': pred["global_neg_sim"],
                    'global_pos_sim': pred["global_pos_sim"],
                    'local_neg_sim': pred["local_neg_sim"],
                    'local_pos_sim': pred["local_pos_sim"],
                    # 'tpr': pred["tpr"], 
                    # 'fpr': pred["fpr"],
                    # 'tpr_local': pred["tpr_local"],
                    # 'fpr_local': pred["fpr_local"]
                    }
            else:
                if self.conf.attentions:
                    return {
                    'local_neg_sim': pred["local_neg_sim"],
                    'local_pos_sim': pred["local_pos_sim"],
                    "attention_local_neg_sim" : pred["attention_local_neg_sim"],
                    "attention_local_pos_sim": pred["attention_local_pos_sim"]
                }
                else:
                    return {
                        'local_neg_sim': pred["local_neg_sim"],
                        'local_pos_sim': pred["local_pos_sim"]
                    }
        else:
            return {}
    
    def contrastive_loss_patches(self, embed0, embed1, gt_labels, label_confs, margin=1, cos=True, conf=False):
        
        """
            contrastive loss on N patches pairs on each image pair
            embed1, embed2 in a shape (B, N, D)
            gt_labels: the matched patches where at least one correspondence are matched, (B, N, N)
            label_confs: how many correspondences in the matched patches, confidence of the labels
            cls_token1, cls_token1: cls tokens in shape (B, D)
        """
        if self.conf.add_cls_tokens:
            # class difference, global, on B image pairs
            global_label_confs = gt_labels.view(gt_labels.shape[0], -1).sum(-1) # global negative image pairs will have global labels=0, indicates no overlaps at all in this pair. 
            global_labels = [global_label_confs > 0][0].to(torch.int16)
            global_cos_sim = cosine_similarity(F.normalize(embed0[:, 0, :], dim=1), F.normalize(embed1[:, 0, :], dim=1), dim=1)
            global_neg, global_pos, global_neg_sim, global_pos_sim = self.contrastive_loss(global_cos_sim, global_labels, global_label_confs, margin, conf, global_loss=True)
            # import pdb; pdb.set_trace()
            # global_neg = (1 - global_labels) * torch.pow(global_cos_sim, 2)
            # global_pos = global_labels * torch.pow(torch.clamp(margin - global_cos_sim, min=0.0), 2) * global_label_confs if conf \
            #     else (global_labels) * torch.pow(torch.clamp(margin - global_cos_sim, min=0.0), 2)
            # # pred_global_labels = (global_cos_sim > 0.5).to(torch.int16)
            # global_neg_sim = (((1-global_labels) * global_cos_sim).sum()/(1-global_labels).sum()).repeat(global_neg.shape)
            # global_pos_sim = ((global_labels * global_cos_sim).sum()/(global_labels).sum()).repeat(global_neg.shape)
        
        # patch difference, local, on B*256 patches, normalize the descriptors for each patch
        x0_normalized = F.normalize(embed0[:, 1:, ], dim=2) if self.conf.add_cls_tokens else torch.nn.functional.normalize(embed0, dim=2) # B, 256, 256
        # all(F.softmax(x0_normalized, dim=2).sum(2) == 1)
        x1_normalized = F.normalize(embed1[:, 1:, ], dim=2) if self.conf.add_cls_tokens else torch.nn.functional.normalize(embed1, dim=2)
        # distances between each patch in each image pair, (B, N, N), [-1, 1]
        local_cos_sim = cosine_similarity(x0_normalized.unsqueeze(2), x1_normalized.unsqueeze(1), dim=3) if cos==True else torch.nn.PairwiseDistance()(x0_normalized.unsqueeze(1), x1_normalized.unsqueeze(2))
        # negative  + positive
        local_neg, local_pos, local_neg_sim, local_pos_sim = self.contrastive_loss(local_cos_sim, gt_labels, label_confs, margin, conf)
        # local_neg = (1 - gt_labels) * torch.pow(local_cos_sim, self.conf.n)
        # # import pdb; pdb.set_trace()
        # local_pos = (gt_labels) * torch.pow(torch.clamp(margin - local_cos_sim, min=0.0), self.conf.n) * label_confs if conf \
        #     else (gt_labels) * torch.pow(torch.clamp(margin - local_cos_sim, min=0.0), self.conf.n)

        # local_neg_sim = ((1-gt_labels) * local_cos_sim).view(local_neg.shape[0], -1).sum(-1)/((1-gt_labels).sum(-1).sum(-1)+1e-4)
        # # positive local loss over N positive patches
        # local_pos_sim = (gt_labels * local_cos_sim).view(local_neg.shape[0], -1).sum(-1)/(gt_labels.sum(-1).sum(-1)+1e-4)
        # import pdb; pdb.set_trace()
        if self.conf.add_cls_tokens:
            return {
                "global_neg": global_neg,#(global_neg.sum()/(1-global_labels).sum()).repeat(global_neg.shape), 
                "global_pos": global_pos,#(global_pos.sum()/global_labels.sum()).repeat(global_neg.shape), 
                "local_neg": local_neg,#.view(local_neg.shape[0], -1).sum(-1)/((1-gt_labels).sum(-1).sum(-1)+1e-4), 
                "local_pos": local_pos,#.view(local_pos.shape[0], -1).sum(-1)/(gt_labels.sum(-1).sum(-1)+1e-4),   
                           
                'global_neg_sim':global_neg_sim, 
                'global_pos_sim':global_pos_sim,
                'local_neg_sim':local_neg_sim,
                'local_pos_sim':local_pos_sim
            }
        else:
            return {
                
                "local_neg": local_neg,#.view(local_neg.shape[0], -1).sum(-1)/((1-gt_labels).sum(-1).sum(-1)+1e-4), 
                "local_pos": local_pos,#.view(local_pos.shape[0], -1).sum(-1)/(gt_labels.sum(-1).sum(-1)+1e-4),   
                'local_neg_sim':local_neg_sim,
                'local_pos_sim':local_pos_sim,
                # 'tpr':tpr,
                # 'fpr':fpr,
                # 'tpr_local':tpr_local,
                # 'fpr_local':fpr_local
                }
        
    def contrastive_loss_attentions(self, attention0, attention1, gt_labels, label_confs, margin=1, cos=True, conf=False):
        # import pdb; pdb.set_trace()
        if self.conf.add_cls_tokens:
            global_label_confs = gt_labels.view(gt_labels.shape[0], -1).sum(-1) # global negative image pairs will have global labels=0, indicates no overlaps at all in this pair. 
            global_labels = [global_label_confs > 0][0].to(torch.int16)           
            global_attentions = ((F.normalize(attention0[:, 0, :], dim=1) + F.normalize(attention1[:, 0, :], dim=1))/2).sum(-1)
            # import pdb; pdb.set_trace()
            global_neg, global_pos, global_neg_sim, global_pos_sim = self.contrastive_loss(global_attentions, global_labels, global_label_confs, global_loss=True)
            local_attentions = (F.normalize(attention0[:, 1:, :], dim=2) + F.normalize(attention1[:, 1:, :], dim=2))/2
            local_neg, local_pos, local_neg_sim, local_pos_sim = self.contrastive_loss(local_attentions, gt_labels, label_confs)
        else:
            local_attentions = (F.normalize(attention0, dim=2) + F.normalize(attention1, dim=2))/2
            local_neg, local_pos, local_neg_sim, local_pos_sim = self.contrastive_loss(local_attentions, gt_labels, label_confs)
        if self.conf.add_cls_tokens:
            return {
                    "global_neg": global_neg,
                    "global_pos": global_pos,
                    "local_neg": local_neg,
                    "local_pos": local_pos,
                    'global_neg_sim':global_neg_sim, 
                    'global_pos_sim':global_pos_sim,
                    'local_neg_sim':local_neg_sim,
                    'local_pos_sim':local_pos_sim}
        else:
            return { 
                    "local_neg": local_neg,
                    "local_pos": local_pos,
                    'local_neg_sim':local_neg_sim,
                    'local_pos_sim':local_pos_sim}
            
        
    def contrastive_loss(self, similarities, gt_labels, label_confs=None, margin=1, conf=False, global_loss=False):
        B = similarities.shape[0]
        # losses
        negative_loss = ((1 - gt_labels) * torch.pow(similarities, self.conf.n))
        positive_loss = gt_labels * torch.pow(torch.clamp(margin - similarities, min=0.0), 2) * label_confs if conf \
                else (gt_labels) * torch.pow(torch.clamp(margin - similarities, min=0.0), 2)   
        # similarities       
        neg_sim = ((1-gt_labels) * similarities)
        pos_sim = (gt_labels * similarities)
        
        if global_loss:
            return (negative_loss.sum()/(1-gt_labels).sum()).repeat(B),\
                    (positive_loss.sum()/gt_labels.sum()).repeat(B),\
                    (neg_sim.view(B, -1).sum()/(1-gt_labels).sum()).repeat(B), \
                    (pos_sim.view(B, -1).sum()/(gt_labels).sum()).repeat(B)
        else:
            return negative_loss.view(B, -1).sum(-1)/((1-gt_labels).sum(-1).sum(-1)+1e-4), \
                    positive_loss.view(B, -1).sum(-1)/(gt_labels.sum(-1).sum(-1)+1e-4), \
                    neg_sim.view(B, -1).sum(-1)/((1-gt_labels).sum(-1).sum(-1)+1e-4), \
                    pos_sim.view(B, -1).sum(-1)/(gt_labels.sum(-1).sum(-1)+1e-4)                    

        
    def confusion_matrix(self, y_true, y_pred):

        # Compute confusion matrix for each example in the batch
        tp = torch.sum((y_true == 1) & (y_pred == 1), dim=1).float()
        tn = torch.sum((y_true == 0) & (y_pred == 0), dim=1).float()
        fp = torch.sum((y_true == 0) & (y_pred == 1), dim=1).float()
        fn = torch.sum((y_true == 1) & (y_pred == 0), dim=1).float()

        return tp, tn, fp, fn
#         pred["gt_visible0"].sum(-1)
# tensor([134., 182.,  36.,  56., 123.,   0.,   0., 114., 146.,   0., 126.,  66.,
#         123.,   0., 135.,  57., 151.,  30.,  53.,  63.,   0.,  34.,  93.,  63.,
#         116.,   0.,   0.,  29., 152.,  50.,   0., 168.,  56.,  43., 109.,  55.,
#          83.,  52., 129.,  99.,  40.,  63.,   0.,  53.,  48.,  65.,  39., 154.,
#          95.,  56., 121.,  17.,   0.,   0.,  95.,  27.,   9.,   0., 166., 107.,
#         106., 147.,  17., 100.], device='cuda:0')
#      pred['overlap0']
# tensor([0.6646, 0.4962, 0.6117, 0.5725, 0.6739, 0.6303, 0.6171, 0.6919, 0.6185,
#         0.6659, 0.5983, 0.6478, 0.6609, 0.6276, 0.5718, 0.6181, 0.7409, 0.7098,
#         0.6317, 0.5886, 0.6350, 0.5674, 0.6665, 0.6212, 0.6446, 0.6108, 0.5837,
#         0.6092, 0.5869, 0.6699, 0.6699, 0.6051, 0.6109, 0.6446, 0.6243, 0.6564,
#         0.6103, 0.5805, 0.6082, 0.6265, 0.6525, 0.6489, 0.5699, 0.6634, 0.6208,
#         0.6792, 0.5886, 0.6076, 0.6821, 0.6834, 0.6212, 0.5635, 0.6236, 0.6759,
#         0.6800, 0.6966, 0.5864, 0.5340, 0.6192, 0.6338, 0.6444, 0.7129, 0.6532,
#         0.6491], device='cuda:0', grad_fn=<MeanBackward1>)   