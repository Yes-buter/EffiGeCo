from __future__ import annotations

import importlib
import os
import sys
from typing import Any, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align
from torchvision.transforms import Resize

from utils.box_ops import boxes_with_scores
from .common import MLP, LayerNorm2d
from .DQE import DQE


def _load_state_dict_flexible(model: nn.Module, checkpoint_path: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint type: {type(ckpt)}")

    model.load_state_dict(state, strict=False)


def _try_build_efficientvit_sam(args: Any) -> nn.Module:
    """Best-effort builder for EfficientViT-SAM.

    This repo doesn't vendor EfficientViT-SAM code. This function tries common import paths.
    If none work, it raises a clear error explaining what interface is expected.
    """

    ckpt_path = getattr(args, "efficientvit_sam_ckpt", None)
    model_name_arg = getattr(args, "efficientvit_sam_type", None)
    model_name = model_name_arg or getattr(args, "efficientvit_sam_name", None)

    # Make vendored repo importable if present: GeCo/third_party/efficientvit
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "third_party", "efficientvit"))
    if os.path.isdir(repo_root) and repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Prefer the official repo checkpoint directory layout.
    if not ckpt_path:
        for base in [
            repo_root,
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            getattr(args, "model_path", None),
        ]:
            if not base or not os.path.isdir(base):
                continue

            candidate_dir = os.path.join(base, "assets", "checkpoints", "efficientvit_sam")
            if os.path.isdir(candidate_dir):
                for name in [
                    "efficientvit_sam_l0.pt",
                    "efficientvit_sam_l1.pt",
                    "efficientvit_sam_l2.pt",
                    "efficientvit_sam_xl0.pt",
                    "efficientvit_sam_xl1.pt",
                ]:
                    candidate = os.path.join(candidate_dir, name)
                    if os.path.exists(candidate):
                        ckpt_path = candidate
                        break
            if ckpt_path:
                break

    # If user didn't specify model name, infer it from checkpoint filename.
    if model_name is None and ckpt_path:
        base = os.path.basename(ckpt_path).lower()
        mapping = {
            "efficientvit_sam_l0.pt": "efficientvit-sam-l0",
            "efficientvit_sam_l1.pt": "efficientvit-sam-l1",
            "efficientvit_sam_l2.pt": "efficientvit-sam-l2",
            "efficientvit_sam_xl0.pt": "efficientvit-sam-xl0",
            "efficientvit_sam_xl1.pt": "efficientvit-sam-xl1",
        }
        model_name = mapping.get(base, None)

    # If user specified a model name but provided/auto-found a mismatched checkpoint, fail fast.
    if model_name_arg and ckpt_path:
        expected_suffix = model_name_arg.replace("efficientvit-sam-", "efficientvit_sam_").replace("-", "_") + ".pt"
        if os.path.basename(ckpt_path).lower() != expected_suffix.lower():
            raise ValueError(
                f"Checkpoint/model mismatch: --efficientvit_sam_type={model_name_arg} but found ckpt={ckpt_path}. "
                f"Please download the matching .pt or set --efficientvit_sam_type to match the ckpt filename."
            )

    candidates = [
        # Official API from mit-han-lab/efficientvit
        ("efficientvit.sam_model_zoo", "create_efficientvit_sam_model"),
        # Backward/alternate guesses
        ("efficientvit.sam_model_zoo", "create_efficientvit_sam"),
        ("efficientvit.sam", "build_efficientvit_sam"),
        ("efficientvit.sam", "build_sam"),
        ("efficientvit_sam", "build_efficientvit_sam"),
        ("efficientvit_sam", "build_sam"),
    ]

    last_error: Optional[Exception] = None

    for module_name, fn_name in candidates:
        try:
            mod = importlib.import_module(module_name)
            builder = getattr(mod, fn_name)
        except Exception as e:
            last_error = e
            continue

        # Call builder with progressively simpler signatures.
        for kwargs in [
            # Official signature: create_efficientvit_sam_model(name, pretrained=True, weight_url=None)
            {"name": model_name or "efficientvit-sam-xl1", "pretrained": True, "weight_url": ckpt_path},
            {"name": model_name or "efficientvit-sam-xl1", "pretrained": True},
            # Generic fallbacks
            {"checkpoint": ckpt_path, "model_type": model_name},
            {"checkpoint": ckpt_path},
            {"ckpt": ckpt_path, "model_type": model_name},
            {"ckpt": ckpt_path},
            {"model_type": model_name},
            {},
        ]:
            try:
                # Remove None values
                call_kwargs = {k: v for k, v in kwargs.items() if v is not None}
                model = builder(**call_kwargs)
                if not isinstance(model, nn.Module):
                    continue
                # If builder didn't load weights, try loading if we found a ckpt path.
                if ckpt_path and os.path.exists(ckpt_path):
                    try:
                        _load_state_dict_flexible(model, ckpt_path)
                    except Exception:
                        pass
                return model
            except TypeError as e:
                last_error = e
                continue
            except Exception as e:
                last_error = e
                continue

    msg = (
        "EfficientViT-SAM backend not available.\n"
        "Expected either: vendored repo at third_party/efficientvit OR a pip-installed `efficientvit` package.\n\n"
        "Official API should be available as: efficientvit.sam_model_zoo.create_efficientvit_sam_model\n\n"
        "Optional args supported (via argparse):\n"
        "- --efficientvit_sam_ckpt <path>\n"
        "- --efficientvit_sam_type <string>   (e.g. efficientvit-sam-xl1)\n\n"
        f"Last import/build error: {last_error}"
    )
    raise ImportError(msg)


class EfficientViTSAMBackbone(nn.Module):
    """Backbone wrapper that matches the existing `Backbone` interface.

    Returns:
      - image_embeddings: (B, 256, H/16, W/16)  (typically 64x64 for 1024 input)
      - hq_features: (B, 32, H/4, W/4)          (typically 256x256 for 1024 input)

    Note: HQ features are approximated from image_embeddings if the upstream model doesn't provide them.
    """

    def __init__(self, requires_grad: bool, image_size: int, args: Any):
        super().__init__()

        self.image_size = image_size
        self.model = _try_build_efficientvit_sam(args)

        # Some implementations expose .image_encoder; some are directly callable.
        self.image_encoder = getattr(self.model, "image_encoder", None)

        # EfficientViT-SAM expects its own normalization; keep constants here.
        self._evit_mean = torch.tensor([123.675 / 255, 116.28 / 255, 103.53 / 255]).view(1, 3, 1, 1)
        self._evit_std = torch.tensor([58.395 / 255, 57.12 / 255, 57.375 / 255]).view(1, 3, 1, 1)

        # GeCo pipeline normalizes with ImageNet stats; keep those for inversion.
        self._imnet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self._imnet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        for _, param in self.named_parameters():
            param.requires_grad_(requires_grad)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert GeCo-normalized 1024x1024 tensor back to [0,1], then apply EfficientViT-SAM preprocessing.
        imnet_mean = self._imnet_mean.to(device=x.device, dtype=x.dtype)
        imnet_std = self._imnet_std.to(device=x.device, dtype=x.dtype)
        x_01 = (x * imnet_std + imnet_mean).clamp(0, 1)

        # EfficientViT-SAM image encoder is trained with long-side 512 and SAM-style mean/std.
        x_512 = F.interpolate(x_01, size=(512, 512), mode="bilinear", align_corners=False)
        evit_mean = self._evit_mean.to(device=x.device, dtype=x.dtype)
        evit_std = self._evit_std.to(device=x.device, dtype=x.dtype)
        x_evit = (x_512 - evit_mean) / evit_std

        if self.image_encoder is not None:
            image_embeddings = self.image_encoder(x_evit)
        else:
            image_embeddings = self.model(x_evit)

        if isinstance(image_embeddings, (tuple, list)):
            image_embeddings = image_embeddings[0]
        if not isinstance(image_embeddings, torch.Tensor):
            raise TypeError(
                "EfficientViT-SAM image encoder must return a torch.Tensor image embedding; "
                f"got {type(image_embeddings)}"
            )

        # Normalize to (B, C, H, W)
        if image_embeddings.dim() == 3:
            # (B, HW, C) or (B, C, HW)
            raise ValueError(
                "Unexpected embedding shape (3D). Please adapt EfficientViT-SAM output to (B, C, H, W)."
            )
        if image_embeddings.dim() != 4:
            raise ValueError(f"Unexpected embedding shape: {tuple(image_embeddings.shape)}")

        # Ensure 256 channels.
        if image_embeddings.shape[1] != 256:
            proj = getattr(self, "_proj", None)
            if proj is None:
                self._proj = nn.Conv2d(image_embeddings.shape[1], 256, kernel_size=1, bias=False).to(
                    image_embeddings.device
                )
            image_embeddings = self._proj(image_embeddings)

        # Parameter-free HQ features (better than random weights): upsample and slice channels.
        hq_upsampled = F.interpolate(image_embeddings, scale_factor=4, mode="bilinear", align_corners=False)
        hq_features = hq_upsampled[:, :32]
        return image_embeddings, hq_features


class GeCo(nn.Module):

    def __init__(
        self,
        image_size: int,
        num_objects: int,
        emb_dim: int,
        num_heads: int,
        kernel_dim: int,
        train_backbone: bool,
        reduction: int,
        zero_shot: bool,
        model_path: str,
        args: Any,
        return_masks: bool = False,
    ):

        super().__init__()

        self.emb_dim = emb_dim
        self.num_objects = num_objects
        self.reduction = reduction
        self.kernel_dim = kernel_dim
        self.image_size = image_size
        self.zero_shot = zero_shot
        self.num_heads = num_heads
        self.num_classes = 1
        self.model_path = model_path

        # Swap backbone implementation.
        self.backbone = EfficientViTSAMBackbone(requires_grad=train_backbone, image_size=image_size, args=args)

        self.class_embed = nn.Sequential(nn.Linear(emb_dim, 1), nn.LeakyReLU())
        self.bbox_embed = MLP(emb_dim, emb_dim, 4, 3)
        self.return_masks = return_masks

        self.emb_dim = 256
        self.adapt_features = DQE(
            transformer_dim=self.emb_dim,
            num_prototype_attn_steps=3,
            num_image_attn_steps=2,
            zero_shot=zero_shot,
        )
        from .prompt_encoder import PromptEncoder_DQE

        self.prompt_encoder = PromptEncoder_DQE(
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        )

        from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer

        prompt_embed_dim = 256
        image_embedding_size = 64
        image_size_sam = 1024
        self.prompt_encoder_sam = PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size_sam, image_size_sam),
            mask_in_chans=16,
        )

        image_embedding_size = 96
        image_size_sam = 1536
        self.prompt_encoder_sam_ = PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size_sam, image_size_sam),
            mask_in_chans=16,
        )

        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        # SAM prompt encoder + mask decoder weights are used ONLY for the refinement stage.
        # If you remove the checkpoint file, we can still run by disabling refinement.
        self._sam_refine_enabled = not bool(getattr(args, "disable_sam_refine", False))

        sam_ckpt = getattr(args, "sam_refine_ckpt", None) or getattr(args, "sam_ckpt", None) or "sam_vit_h_4b8939.pth"
        if not os.path.isabs(sam_ckpt):
            # Try resolving relative to repo root (GeCo/) first.
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            candidate = os.path.join(repo_root, sam_ckpt)
            if os.path.exists(candidate):
                sam_ckpt = candidate

        if self._sam_refine_enabled:
            try:
                try:
                    checkpoint = torch.load(sam_ckpt, map_location="cpu", weights_only=True)
                except TypeError:
                    checkpoint = torch.load(sam_ckpt, map_location="cpu")

                state_dict = {k.replace("mask_decoder.", ""): v for k, v in checkpoint.items() if "mask_decoder" in k}
                self.mask_decoder.load_state_dict(state_dict)
                state_dict = {k.replace("prompt_encoder.", ""): v for k, v in checkpoint.items() if "prompt_encoder" in k}
                self.prompt_encoder_sam.load_state_dict(state_dict)
                self.prompt_encoder_sam_.load_state_dict(state_dict)
            except FileNotFoundError:
                self._sam_refine_enabled = False
                print(
                    f"[GeCo] SAM refine checkpoint not found: {sam_ckpt}. "
                    "Refinement disabled (boxes will be coarse)."
                )

        if self.zero_shot:
            self.exemplars = nn.Parameter(torch.randn(1, emb_dim))
        else:
            self.shape_or_objectness = nn.Sequential(
                nn.Linear(2, 64),
                nn.ReLU(),
                nn.Linear(64, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, 1**2 * emb_dim),
            )
        self.resize = Resize((512, 512))

    def refine_bounding_boxes(self, features, outputs, return_masks=False):

        if not getattr(self, "_sam_refine_enabled", True):
            # No refinement: return empty masks, use existing boxes, and use box_v as scores proxy.
            batch_masks = [[] for _ in range(len(outputs))]
            batch_iou = [outputs[i]["box_v"].detach() for i in range(len(outputs))]
            batch_bboxes = [outputs[i]["pred_boxes"][0].detach() for i in range(len(outputs))]
            return batch_masks, batch_iou, batch_bboxes

        batch_masks = []
        batch_iou = []
        batch_bboxes = []
        for i in range(len(outputs)):
            step = 50
            masks = []
            iou_predictions = []
            corrected_bboxes_ = []
            for box_i in range(step, len(outputs[i]["pred_boxes"][0]) + step, step):
                box = outputs[i]["pred_boxes"][0][(box_i - step) : box_i] * features.shape[-1] * 16
                if features.shape[-1] * 16 == 1024:
                    sparse_embeddings, dense_embeddings = self.prompt_encoder_sam(
                        points=None,
                        boxes=box,
                        masks=None,
                    )
                else:
                    sparse_embeddings, dense_embeddings = self.prompt_encoder_sam_(
                        points=None,
                        boxes=box,
                        masks=None,
                    )

                masks_, iou_predictions_ = self.mask_decoder(
                    image_embeddings=features[i].unsqueeze(0),
                    image_pe=self.prompt_encoder_sam.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                masks_ = F.interpolate(
                    masks_, (features.shape[-1] * 16, features.shape[-1] * 16), mode="bilinear", align_corners=False
                )
                masks_ = masks_ > 0
                if return_masks:
                    masks_ = masks_[..., :1024, :1024]
                    masks.append(masks_)
                iou_predictions.append(iou_predictions_)

                corrected_bboxes = torch.zeros((masks_.shape[0], 4), dtype=torch.float)
                masks_ = masks_[:, 0]
                for index, mask_i in enumerate(masks_):
                    y, x = torch.where(mask_i != 0)
                    if y.shape[0] > 0 and x.shape[0] > 0:
                        corrected_bboxes[index, 0] = torch.min(x)
                        corrected_bboxes[index, 1] = torch.min(y)
                        corrected_bboxes[index, 2] = torch.max(x)
                        corrected_bboxes[index, 3] = torch.max(y)
                corrected_bboxes_.append(corrected_bboxes)
            if len(corrected_bboxes_) > 0:
                if return_masks:
                    batch_masks.append(torch.cat(masks, dim=0)[:, 0])
                else:
                    batch_masks.append([])
                batch_bboxes.append(torch.cat(corrected_bboxes_))
                batch_iou.append(torch.cat(iou_predictions).permute(1, 0))
            else:
                batch_masks.append([])
                batch_bboxes.append(torch.tensor([]).to(features.device))
                batch_iou.append(torch.tensor([]).to(features.device))
        return batch_masks, batch_iou, batch_bboxes

    def forward(self, img, bboxes):
        num_objects = bboxes.size(1) if not self.zero_shot else self.num_objects

        src, src_hq = self.backbone(img)
        bs, c, h, w = src.size()

        if not self.zero_shot:
            prototype_embeddings = self.create_prototypes(src, bboxes)
        else:  # zero shot
            prototype_embeddings = self.exemplars.expand(bs, -1, -1)

        adapted_f = self.adapt_features(
            image_embeddings=src,
            image_pe=self.prompt_encoder.get_dense_pe(),
            prototype_embeddings=prototype_embeddings,
            hq_features=src_hq,
        )

        bs, c, w, h = adapted_f.shape
        adapted_f = adapted_f.view(bs, self.emb_dim, -1).permute(0, 2, 1)
        centerness = self.class_embed(adapted_f).view(bs, w, h, 1).permute(0, 3, 1, 2)
        outputs_coord = self.bbox_embed(adapted_f).sigmoid().view(bs, w, h, 4).permute(0, 3, 1, 2)
        outputs, ref_points = boxes_with_scores(centerness, outputs_coord, batch_thresh=0.001)
        masks, ious, corrected_bboxes = self.refine_bounding_boxes(src, outputs, return_masks=self.return_masks)

        for i in range(len(outputs)):
            outputs[i]["scores"] = ious[i]
            if getattr(self, "_sam_refine_enabled", True):
                outputs[i]["pred_boxes"] = corrected_bboxes[i].to(outputs[i]["pred_boxes"].device).unsqueeze(0) / img.shape[
                    -1
                ]

        return outputs, ref_points, centerness, outputs_coord, masks

    def forward_cross(self, support_img, support_bboxes, query_img):
        src_support, _ = self.backbone(support_img)

        if not self.zero_shot:
            prototype_embeddings = self.create_prototypes(src_support, support_bboxes)
        else:
            bs = query_img.size(0)
            prototype_embeddings = self.exemplars.expand(bs, -1, -1)

        src_query, src_hq_query = self.backbone(query_img)

        adapted_f = self.adapt_features(
            image_embeddings=src_query,
            image_pe=self.prompt_encoder.get_dense_pe(),
            prototype_embeddings=prototype_embeddings,
            hq_features=src_hq_query,
        )

        bs, c, w, h = adapted_f.shape
        adapted_f = adapted_f.view(bs, self.emb_dim, -1).permute(0, 2, 1)
        centerness = self.class_embed(adapted_f).view(bs, w, h, 1).permute(0, 3, 1, 2)
        outputs_coord = self.bbox_embed(adapted_f).sigmoid().view(bs, w, h, 4).permute(0, 3, 1, 2)
        outputs, ref_points = boxes_with_scores(centerness, outputs_coord, batch_thresh=0.001)

        masks, ious, corrected_bboxes = self.refine_bounding_boxes(src_query, outputs, return_masks=self.return_masks)

        for i in range(len(outputs)):
            outputs[i]["scores"] = ious[i]
            if getattr(self, "_sam_refine_enabled", True):
                outputs[i]["pred_boxes"] = (
                    corrected_bboxes[i].to(outputs[i]["pred_boxes"].device).unsqueeze(0) / query_img.shape[-1]
                )

        return outputs, ref_points, centerness, outputs_coord, masks

    def create_prototypes(self, src, bboxes):
        bs = src.size(0)
        self.num_objects = bboxes.size(1)

        bboxes_roi = torch.cat(
            [
                torch.arange(bs, requires_grad=False).to(bboxes.device).repeat_interleave(self.num_objects).reshape(-1, 1),
                bboxes.flatten(0, 1),
            ],
            dim=1,
        )
        self.kernel_dim = 1

        exemplars = (
            roi_align(src, boxes=bboxes_roi, output_size=self.kernel_dim, spatial_scale=1.0 / self.reduction, aligned=True)
            .permute(0, 2, 3, 1)
            .reshape(bs, self.num_objects * self.kernel_dim**2, self.emb_dim)
        )

        box_hw = torch.zeros(bboxes.size(0), bboxes.size(1), 2).to(bboxes.device)
        box_hw[:, :, 0] = bboxes[:, :, 2] - bboxes[:, :, 0]
        box_hw[:, :, 1] = bboxes[:, :, 3] - bboxes[:, :, 1]

        shape = self.shape_or_objectness(box_hw).reshape(bs, -1, self.emb_dim)
        prototype_embeddings = torch.cat([exemplars, shape], dim=1)
        return prototype_embeddings


def build_model(args):
    assert args.reduction in [4, 8, 16]

    return GeCo(
        image_size=args.image_size,
        num_objects=args.num_objects,
        zero_shot=args.zero_shot,
        emb_dim=args.emb_dim,
        num_heads=args.num_heads,
        kernel_dim=args.kernel_dim,
        train_backbone=args.backbone_lr > 0,
        reduction=args.reduction,
        model_path=args.model_path,
        args=args,
        return_masks=args.output_masks,
    )
