import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig, AutoModel
from src.model.multimodal_encoder.vit import ViT
import torch.distributed as dist
from src.utils.dist_utils import gather_features, gather_variable


class CLIPConfig(PretrainedConfig):
    model_type = "clip"

    def __init__(
        self,
        language_model_name_or_path: str = "",
        local_loss: bool = False,
        gather_loss: bool = True,
        channels: int = 1,
        img_size: tuple = (192, 192, 336),
        patch_size: tuple = (16, 16, 12),
        dim: int = 768,
        mlp_dim: int = 3072,
        depth: int = 12,
        heads: int = 6,
        dim_head: int = 64,
        dropout: float = 0.,
        emb_dropout: float = 0.,
        use_region_mask: bool = False,
        use_cls: bool = True,
        **kwargs,
    ):
        self.language_model_name_or_path = language_model_name_or_path
        self.channels = channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.dim = dim
        self.mlp_dim = mlp_dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.emb_dropout = emb_dropout
        self.local_loss = local_loss
        self.gather_loss = gather_loss
        self.use_region_mask = use_region_mask
        self.use_cls = use_cls
        super().__init__(**kwargs)




class CLIP(PreTrainedModel):
    config_class = CLIPConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.use_cls = getattr(config, 'use_cls', True)

        self.fusion_conv = None
        if config.channels == 2:
            self.fusion_conv = nn.Sequential(
                nn.Conv3d(2, 8, kernel_size=3, padding=1),
                nn.BatchNorm3d(16),
                nn.GELU(),
                nn.Conv3d(8, 1, kernel_size=3, padding=1),
                nn.BatchNorm3d(1),
                nn.GELU(),
            )

        self.vision_encoder = ViT(
            img_size=config.img_size,
            patch_size=config.patch_size,
            dim=config.dim,
            depth=config.depth,
            heads=config.heads,
            mlp_dim=config.mlp_dim,
            channels=1,
            dim_head=config.dim_head,
            dropout=config.dropout,
            emb_dropout=config.emb_dropout,
            use_cls=self.use_cls,
        )

        self.language_encoder = AutoModel.from_pretrained(config.language_model_name_or_path, trust_remote_code=True)

        self.mm_vision_proj = nn.Linear(config.dim, config.dim)
        self.mm_language_proj = nn.Linear(config.dim, config.dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.local_loss = config.local_loss
        self.gather_loss = config.gather_loss
        self.use_region_mask = getattr(config, 'use_region_mask', False)

        self.img_size = config.img_size
        self.patch_size = config.patch_size

    def _mask_to_patch_weights(self, mask):
        B = mask.shape[0]
        D, H, W = self.img_size
        pd, ph, pw = self.patch_size
        nd, nh, nw = D // pd, H // ph, W // pw

        mask_resized = F.interpolate(
            mask.unsqueeze(1).float(),
            size=(D, H, W),
            mode='trilinear',
            align_corners=False
        )
        mask_resized = mask_resized.view(B, 1, nd, pd, nh, ph, nw, pw)
        patch_weights = mask_resized.mean(dim=(3, 5, 7)).view(B, nd * nh * nw)
        return patch_weights

    def _fuse(self, image):
        if self.fusion_conv is not None:
            image = self.fusion_conv(image)
        return image

    def encode_image(self, image):
        image = self._fuse(image)
        image_feats = self.vision_encoder(image)
        if self.use_cls:
            image_feats = image_feats[:, 0]
        image_feats = self.mm_vision_proj(image_feats)
        image_feats = F.normalize(image_feats, dim=-1)
        return image_feats

    def encode_region(self, image, mask):
        image = self._fuse(image)
        image_feats = self.vision_encoder(image)
        patch_feats = image_feats[:, 1:]
        patch_weights = self._mask_to_patch_weights(mask)
        patch_weights = patch_weights / (patch_weights.sum(dim=-1, keepdim=True) + 1e-8)
        region_feats = (patch_feats * patch_weights.unsqueeze(-1)).sum(dim=1)
        region_feats = self.mm_vision_proj(region_feats)
        region_feats = F.normalize(region_feats, dim=-1)
        return region_feats

    def encode_text(self, input_id, attention_mask):
        text_feats = self.language_encoder(input_id, attention_mask=attention_mask)["last_hidden_state"]
        text_feats = self.mm_language_proj(text_feats)
        text_feats = F.normalize(text_feats, dim=-1)
        return text_feats

    def forward(self, images, input_ids, attention_mask, labels, masks=None, **kwargs):
        if masks is not None and self.use_region_mask:
            image_features = self.encode_region(images, masks)
        else:
            image_features = self.encode_image(images)
        text_features = self.encode_text(input_ids, attention_mask)[:, 0]

        if self.gather_loss:
            all_image_features, all_text_features = gather_features(image_features, text_features)
            logits_per_image = self.logit_scale.exp() * all_image_features @ all_text_features.T
            logits_per_text = logits_per_image.T
        else:
            logits_per_image = self.logit_scale.exp() * image_features @ text_features.T
            logits_per_text = logits_per_image.T

        n = logits_per_image.shape[0]
        targets = torch.arange(n, device=logits_per_image.device)
        loss_i2t = F.cross_entropy(logits_per_image, targets)
        loss_t2i = F.cross_entropy(logits_per_text, targets)
        loss = (loss_i2t + loss_t2i) / 2

        ret = {
            "loss": loss,
            "logits": logits_per_image,
        }

        return ret


class SIGLIP(PreTrainedModel):
    config_class = CLIPConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.use_cls = getattr(config, 'use_cls', True)
        self.vision_encoder = ViT(
            img_size=config.img_size,
            patch_size=config.patch_size,
            dim=config.dim,
            depth=config.depth,
            heads=config.heads,
            mlp_dim=config.mlp_dim,
            channels=config.channels,
            dim_head=config.dim_head,
            dropout=config.dropout,
            emb_dropout=config.emb_dropout,
            use_cls=self.use_cls,
        )

        # self.language_encoder = AutoModel.from_pretrained(config.language_model_name_or_path, attn_implementation="flash_attention_2", dtype=torch.bfloat16)
        self.language_encoder = AutoModel.from_pretrained(config.language_model_name_or_path, trust_remote_code=True)

        self.mm_vision_proj = nn.Linear(config.dim, config.dim)
        self.mm_language_proj = nn.Linear(config.dim, config.dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(100))
        self.logit_bias = nn.Parameter(torch.ones([]) * (-3.5))

        self.local_loss = config.local_loss
        self.gather_loss = config.gather_loss
        self.use_region_mask = getattr(config, 'use_region_mask', False)

        self.img_size = config.img_size
        self.patch_size = config.patch_size

    def _mask_to_patch_weights(self, mask):
        B = mask.shape[0]
        D, H, W = self.img_size
        pd, ph, pw = self.patch_size
        nd, nh, nw = D // pd, H // ph, W // pw

        mask_resized = F.interpolate(
            mask.unsqueeze(1).float(),
            size=(D, H, W),
            mode='trilinear',
            align_corners=False
        )
        mask_resized = mask_resized.view(B, 1, nd, pd, nh, ph, nw, pw)
        patch_weights = mask_resized.mean(dim=(3, 5, 7)).view(B, nd * nh * nw)
        return patch_weights

    def encode_image(self, image):
        image_feats = self.vision_encoder(image)
        if self.use_cls:
            image_feats = image_feats[:, 0]
        image_feats = self.mm_vision_proj(image_feats)
        image_feats = F.normalize(image_feats, dim=-1)
        return image_feats

    def encode_region(self, image, mask):
        image_feats = self.vision_encoder(image)
        patch_feats = image_feats[:, 1:]
        patch_weights = self._mask_to_patch_weights(mask)
        patch_weights = patch_weights / (patch_weights.sum(dim=-1, keepdim=True) + 1e-8)
        region_feats = (patch_feats * patch_weights.unsqueeze(-1)).sum(dim=1)
        region_feats = self.mm_vision_proj(region_feats)
        region_feats = F.normalize(region_feats, dim=-1)
        return region_feats

    def encode_text(self, input_id, attention_mask):
        text_feats = self.language_encoder(input_id, attention_mask=attention_mask)["last_hidden_state"]
        text_feats = self.mm_language_proj(text_feats)
        text_feats = F.normalize(text_feats, dim=-1)
        return text_feats

    def forward(self, images, input_ids, attention_mask, labels, masks=None, **kwargs):
        if masks is not None and self.use_region_mask:
            image_features = self.encode_region(images, masks)
        else:
            image_features = self.encode_image(images)
        text_features = self.encode_text(input_ids, attention_mask)[:, 0]

        if self.gather_loss:
            all_image_features, all_text_features = gather_features(image_features, text_features)
            logits = self.logit_scale * all_image_features @ all_text_features.T
        else:
            logits = self.logit_scale * image_features @ text_features.T

        logits += self.logit_bias

        n = logits.shape[0]
        labels_siglip = 2 * torch.eye(n, device=logits.device, dtype=logits.dtype) - 1
        loss = -F.logsigmoid(labels_siglip * logits).sum() / n

        ret = {
            "loss": loss,
            "logits": logits,
        }

        return ret


class PETCTCLIPConfig(PretrainedConfig):
    model_type = "petctclip"

    def __init__(
        self,
        language_model_name_or_path: str = "",
        img_size: tuple = (192, 192, 336),
        patch_size: tuple = (16, 16, 12),
        dim: int = 768,
        mlp_dim: int = 3072,
        depth: int = 12,
        heads: int = 12,
        dim_head: int = 64,
        dropout: float = 0.,
        emb_dropout: float = 0.,
        margin: float = 0.1,
        lambda_loc: float = 1.0,
        lambda_ex: float = 0.5,
        lambda_and: float = 0.3,
        lambda_ent: float = 0.01,
        lambda_pr: float = 0.1,
        pi_0: list = None,
        **kwargs,
    ):
        self.language_model_name_or_path = language_model_name_or_path
        self.img_size = img_size
        self.patch_size = patch_size
        self.dim = dim
        self.mlp_dim = mlp_dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.emb_dropout = emb_dropout
        self.margin = margin
        self.lambda_loc = lambda_loc
        self.lambda_ex = lambda_ex
        self.lambda_and = lambda_and
        self.lambda_ent = lambda_ent
        self.lambda_pr = lambda_pr
        self.pi_0 = pi_0 if pi_0 is not None else [0.25, 0.25, 0.50]
        super().__init__(**kwargs)


class PETCTCLIP(PreTrainedModel):
    config_class = PETCTCLIPConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        vit_kwargs = dict(
            img_size=config.img_size,
            patch_size=config.patch_size,
            dim=config.dim,
            depth=config.depth,
            heads=config.heads,
            mlp_dim=config.mlp_dim,
            channels=1,
            dim_head=config.dim_head,
            dropout=config.dropout,
            emb_dropout=config.emb_dropout,
            use_cls=False,
        )
        self.pet_encoder = ViT(**vit_kwargs)
        self.ct_encoder = ViT(**vit_kwargs)

        self.language_encoder = AutoModel.from_pretrained(
            config.language_model_name_or_path,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )

        self.pet_vision_proj = nn.Linear(config.dim, config.dim)
        self.ct_vision_proj = nn.Linear(config.dim, config.dim)
        self.text_proj = nn.Linear(config.dim, config.dim)

        self.fusion_proj = nn.Sequential(
            nn.Linear(2 * config.dim, 2 * config.dim),
            nn.GELU(),
            nn.LayerNorm(2 * config.dim),
            nn.Linear(2 * config.dim, config.dim),
            nn.GELU(),
            nn.LayerNorm(config.dim),
        )

        self.router = nn.Linear(config.dim, 3)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10.0))
        self.logit_bias = nn.Parameter(torch.ones([]) * (-5.0))

        self.img_size = config.img_size
        self.patch_size = config.patch_size

    def _mask_pool(self, F_patches, mask):
        B, N, C = F_patches.shape
        mask_down = F.avg_pool3d(
            mask.unsqueeze(1).float(),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        mask_flat = mask_down.view(B, -1)
        weights = mask_flat / (mask_flat.sum(dim=-1, keepdim=True) + 1e-8)
        return (F_patches * weights.unsqueeze(-1)).sum(dim=1)

    def encode_vision(self, pet, ct):
        F_pet = self.pet_encoder(pet)
        F_ct = self.ct_encoder(ct)
        return F_pet, F_ct

    def get_global_vision(self, F_pet, F_ct):
        v_pet = F.normalize(self.pet_vision_proj(F_pet.mean(dim=1)), dim=-1)
        v_ct = F.normalize(self.ct_vision_proj(F_ct.mean(dim=1)), dim=-1)
        v_fus = F.normalize(
            self.fusion_proj(torch.cat([v_pet, v_ct], dim=-1)), dim=-1
        )
        return v_pet, v_ct, v_fus

    def get_organ_vision(self, F_pet, F_ct, organ_masks, sent_patient_idx):
        F_pet_g = F_pet[sent_patient_idx]
        F_ct_g = F_ct[sent_patient_idx]
        v_pet_o = F.normalize(self.pet_vision_proj(self._mask_pool(F_pet_g, organ_masks)), dim=-1)
        v_ct_o = F.normalize(self.ct_vision_proj(self._mask_pool(F_ct_g, organ_masks)), dim=-1)
        v_fus_o = F.normalize(
            self.fusion_proj(torch.cat([v_pet_o, v_ct_o], dim=-1)), dim=-1
        )
        return v_pet_o, v_ct_o, v_fus_o

    def encode_text_global(self, input_ids, attention_mask):
        H = self.language_encoder(input_ids, attention_mask=attention_mask)["last_hidden_state"]
        mask = attention_mask.unsqueeze(-1).float()
        t_global = self.text_proj((H * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1))
        return F.normalize(t_global, dim=-1)

    def encode_text_with_router(self, input_ids, attention_mask):
        H = self.language_encoder(input_ids, attention_mask=attention_mask)["last_hidden_state"]
        H = self.text_proj(H)

        w = torch.softmax(self.router(H), dim=-1)
        mask = attention_mask.unsqueeze(-1).float()
        L_valid = mask.sum(dim=1, keepdim=True).clamp(min=1)

        den_pet = (w[:, :, 0:1] * mask).sum(dim=1).clamp(min=1e-6)
        den_ct = (w[:, :, 1:2] * mask).sum(dim=1).clamp(min=1e-6)
        den_shr = (w[:, :, 2:3] * mask).sum(dim=1).clamp(min=1e-6)
        t_pet = F.normalize((w[:, :, 0:1] * H * mask).sum(dim=1) / den_pet, dim=-1)
        t_ct = F.normalize((w[:, :, 1:2] * H * mask).sum(dim=1) / den_ct, dim=-1)
        t_shr = F.normalize((w[:, :, 2:3] * H * mask).sum(dim=1) / den_shr, dim=-1)

        alpha_pet = (w[:, :, 0:1] * mask).sum(dim=1).squeeze(-1) / L_valid.view(-1)
        alpha_ct = (w[:, :, 1:2] * mask).sum(dim=1).squeeze(-1) / L_valid.view(-1)
        alpha_shr = (w[:, :, 2:3] * mask).sum(dim=1).squeeze(-1) / L_valid.view(-1)

        return {
            "t_pet": t_pet, "t_ct": t_ct, "t_shr": t_shr,
            "alpha_pet": alpha_pet, "alpha_ct": alpha_ct, "alpha_shr": alpha_shr,
            "w": w,
        }

    def forward(self, pet, ct,
                global_input_ids, global_attention_mask,
                sent_input_ids=None, sent_attention_mask=None,
                sent_patient_idx=None, sent_organ_idx=None,
                organ_masks=None, organ_labels=None, **kwargs):
        from src.model.loss import petctclip_loss

        F_pet, F_ct = self.encode_vision(pet, ct)
        v_pet, v_ct, v_fus = self.get_global_vision(F_pet, F_ct)
        t_global = self.encode_text_global(global_input_ids, global_attention_mask)

        all_t_global, all_v_fus = gather_features(t_global, v_fus)

        features = {
            "v_fus": all_v_fus,
            "t_global": all_t_global,
            "logit_scale": self.logit_scale.exp(),
            "logit_bias": self.logit_bias,
        }

        if sent_input_ids is not None:
            text_out = self.encode_text_with_router(sent_input_ids, sent_attention_mask)

            if organ_masks is not None:
                v_pet_o, v_ct_o, v_fus_o = self.get_organ_vision(
                    F_pet, F_ct, organ_masks, sent_patient_idx
                )

                world_size = dist.get_world_size() if dist.is_initialized() else 1
                rank = dist.get_rank() if world_size > 1 else 0
                offset_idx = sent_patient_idx + rank * pet.shape[0]

                (all_t_pet, all_t_ct, all_t_shr,
                 all_v_pet_o, all_v_ct_o, all_v_fus_o,
                 all_alpha_pet, all_alpha_ct, all_alpha_shr,
                 all_patient_idx, all_organ_idx) = gather_variable(
                    text_out["t_pet"], text_out["t_ct"], text_out["t_shr"],
                    v_pet_o, v_ct_o, v_fus_o,
                    text_out["alpha_pet"], text_out["alpha_ct"], text_out["alpha_shr"],
                    offset_idx, sent_organ_idx,
                )

                if world_size > 1:
                    ol_list = [torch.zeros_like(organ_labels) for _ in range(world_size)]
                    dist.all_gather(ol_list, organ_labels.contiguous())
                    all_organ_labels = torch.cat(ol_list, dim=0)
                else:
                    all_organ_labels = organ_labels

                features.update({
                    "t_pet": all_t_pet, "t_ct": all_t_ct, "t_shr": all_t_shr,
                    "v_pet_o": all_v_pet_o, "v_ct_o": all_v_ct_o, "v_fus_o": all_v_fus_o,
                    "alpha_pet": all_alpha_pet, "alpha_ct": all_alpha_ct, "alpha_shr": all_alpha_shr,
                    "sent_patient_idx": all_patient_idx,
                    "sent_organ_idx": all_organ_idx,
                    "organ_labels": all_organ_labels,
                })

            features["w"] = text_out["w"]
            features["sent_attention_mask"] = sent_attention_mask

        loss, losses_dict = petctclip_loss(features, self.config)
        return {"loss": loss, "losses": losses_dict}