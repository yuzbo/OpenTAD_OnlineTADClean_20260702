import contextlib
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS


class _TinyFrameVisionEncoder(nn.Module):
    """Small deterministic frame encoder used only by runtime smoke tests."""

    def __init__(self, embed_dims):
        super().__init__()
        self.conv = nn.Conv2d(3, embed_dims, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, pixel_values):
        x = self.conv(pixel_values)
        x = self.pool(x).flatten(1)
        return x


class CausalMotionBranch(nn.Module):
    """Causal feature-difference branch for streaming-safe motion cues."""

    def __init__(self, channels, kernel_size=3, causal=True, init_scale=0.1):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.causal = bool(causal)
        self.left_pad = self.kernel_size - 1 if self.causal else self.kernel_size // 2
        self.dwconv = nn.Conv1d(
            self.channels,
            self.channels,
            kernel_size=self.kernel_size,
            groups=self.channels,
            padding=0 if self.causal else self.kernel_size // 2,
        )
        self.mix = nn.Conv1d(self.channels, self.channels, kernel_size=1)
        self.act = nn.GELU()
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)

    def forward(self, x, masks=None):
        if x.shape[-1] <= 1:
            delta = x
        else:
            prev = F.pad(x[..., :-1], (1, 0))
            delta = x - prev

        if self.causal:
            delta = F.pad(delta, (self.left_pad, 0))
        motion = self.dwconv(delta)
        motion = self.mix(self.act(motion))
        out = x + self.scale.to(dtype=x.dtype) * motion
        if masks is not None:
            out = out * masks.unsqueeze(1).to(dtype=out.dtype)
        return out


@MODELS.register_module()
class OnlineSigLIPFrameEncoder(nn.Module):
    """Strictly online per-frame SigLIP/SigLIP2 vision encoder.

    This module encodes each sampled raw frame independently, so future frames
    cannot affect past feature tokens. A causal motion branch can be enabled for
    P2 experiments without changing the raw-frame dataset contract.
    """

    input_format = "raw_frames"
    online = True
    strict_online = True

    def __init__(
        self,
        model_name="google/siglip2-base-patch16-224",
        backend="transformers",
        embed_dims=None,
        image_size=None,
        image_mean=None,
        image_std=None,
        pooling="mean",
        frame_stride=1,
        frame_chunk_size=32,
        freeze_vision_encoder=True,
        strict_online=True,
        local_files_only=False,
        trust_remote_code=False,
        revision=None,
        use_motion_branch=False,
        motion_branch=None,
        frame_selector=None,
        encode_policy="dense",
        assert_selected_only=False,
        return_token_times=False,
        output_layout="bct",
    ):
        super().__init__()
        self.model_name = model_name
        self.backend = backend
        self.pooling = pooling
        self.frame_stride = int(frame_stride)
        self.frame_chunk_size = int(frame_chunk_size)
        self.freeze_vision_encoder = bool(freeze_vision_encoder)
        self.strict_online = bool(strict_online)
        self.local_files_only = bool(local_files_only)
        self.trust_remote_code = bool(trust_remote_code)
        self.revision = revision
        self.use_motion_branch = bool(use_motion_branch)
        self.encode_policy = str(encode_policy)
        if self.encode_policy not in {"dense", "selected_only"}:
            raise ValueError("encode_policy must be 'dense' or 'selected_only'")
        self.assert_selected_only = bool(assert_selected_only)
        self.return_token_times = bool(return_token_times)
        self.output_layout = output_layout.lower()
        if self.output_layout not in {"bct", "btc"}:
            raise ValueError("output_layout must be either 'bct' or 'btc'")
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be positive")
        if self.frame_stride != 1:
            raise ValueError("frame_stride != 1 would desynchronize dataset snippet_stride metadata")
        if self.frame_chunk_size < 1:
            raise ValueError("frame_chunk_size must be positive")
        if pooling not in {"mean", "cls", "pooler"}:
            raise ValueError("pooling must be one of: mean, cls, pooler")

        self.processor = None
        if backend == "tiny":
            self.embed_dims = int(embed_dims or 8)
            self.image_size = int(image_size or 8)
            self.patch_size = 1
            self.vision_encoder = _TinyFrameVisionEncoder(self.embed_dims)
        elif backend == "transformers":
            self.processor, self.vision_encoder, inferred_dims, inferred_size, inferred_patch_size = self._build_transformers_encoder(
                model_name=model_name,
                local_files_only=self.local_files_only,
                trust_remote_code=self.trust_remote_code,
                revision=self.revision,
            )
            self.embed_dims = int(embed_dims or inferred_dims)
            self.image_size = int(image_size or inferred_size or 224)
            self.patch_size = int(inferred_patch_size or 1)
            if inferred_dims is not None and self.embed_dims != int(inferred_dims):
                raise ValueError(
                    f"embed_dims={self.embed_dims} does not match SigLIP hidden size {inferred_dims}. "
                    "Set the adapter embed_dims to the vision hidden size and project later."
                )
        else:
            raise ValueError("backend must be 'transformers' or 'tiny'")

        mean = image_mean
        std = image_std
        if mean is None and self.processor is not None:
            mean = getattr(self.processor, "image_mean", None)
        if std is None and self.processor is not None:
            std = getattr(self.processor, "image_std", None)
        mean = mean if mean is not None else (0.5, 0.5, 0.5)
        std = std if std is not None else (0.5, 0.5, 0.5)
        self.register_buffer("image_mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

        motion_branch = motion_branch or {}
        if self.use_motion_branch:
            self.motion_branch = CausalMotionBranch(
                channels=int(motion_branch.get("channels", self.embed_dims)),
                kernel_size=int(motion_branch.get("kernel_size", 3)),
                causal=bool(motion_branch.get("causal", True)),
                init_scale=float(motion_branch.get("init_scale", 0.1)),
            )
        else:
            self.motion_branch = None
        self.frame_selector = MODELS.build(frame_selector) if frame_selector is not None else None
        if self.encode_policy == "selected_only" and self.frame_selector is None:
            raise ValueError("encode_policy='selected_only' requires frame_selector")
        self.last_selector_stats = {}

        self._configure_trainability()

    def _build_transformers_encoder(self, model_name, local_files_only=False, trust_remote_code=False, revision=None):
        if model_name is None:
            raise ValueError("model_name is required when backend='transformers'")
        try:
            from transformers import AutoConfig, AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ImportError(
                "OnlineSigLIPFrameEncoder requires transformers for backend='transformers'. "
                "Install transformers or use backend='tiny' for local smoke tests."
            ) from exc

        load_kwargs = dict(local_files_only=local_files_only, trust_remote_code=trust_remote_code)
        if revision is not None:
            load_kwargs["revision"] = revision
        processor = AutoImageProcessor.from_pretrained(model_name, **load_kwargs)
        config = AutoConfig.from_pretrained(model_name, **load_kwargs)
        model_type = getattr(config, "model_type", None)
        if model_type == "siglip":
            from transformers import SiglipModel

            model = SiglipModel.from_pretrained(model_name, **load_kwargs)
        elif model_type == "siglip2":
            from transformers import Siglip2Model

            model = Siglip2Model.from_pretrained(model_name, **load_kwargs)
        else:
            model = AutoModel.from_pretrained(model_name, **load_kwargs)
        vision_encoder = model.vision_model if hasattr(model, "vision_model") else model

        vision_cfg = getattr(getattr(model, "config", None), "vision_config", None)
        inferred_dims = getattr(vision_cfg, "hidden_size", None)
        if inferred_dims is None:
            inferred_dims = getattr(getattr(vision_encoder, "config", None), "hidden_size", None)
        inferred_size = getattr(vision_cfg, "image_size", None)
        if inferred_size is None:
            inferred_size = getattr(getattr(vision_encoder, "config", None), "image_size", None)
        if inferred_size is None:
            size = getattr(processor, "size", None)
            if isinstance(size, dict):
                inferred_size = size.get("height") or size.get("shortest_edge")
            elif isinstance(size, int):
                inferred_size = size
        inferred_patch_size = getattr(vision_cfg, "patch_size", None)
        if inferred_patch_size is None:
            inferred_patch_size = getattr(getattr(vision_encoder, "config", None), "patch_size", None)
        return processor, vision_encoder, inferred_dims, inferred_size, inferred_patch_size

    def _configure_trainability(self):
        if not self.freeze_vision_encoder:
            return
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        if self.motion_branch is not None:
            for param in self.motion_branch.parameters():
                param.requires_grad = True

    def _prepare_frames(self, frames, masks=None):
        if frames.dim() == 6:
            if frames.shape[1] != 1:
                raise ValueError("OnlineSigLIPFrameEncoder expects num_clips=1 for strict online encoding")
            frames = frames[:, 0]
        if frames.dim() != 5:
            raise ValueError(f"Expected frames with shape [B,1,3,T,H,W] or [B,3,T,H,W], got {tuple(frames.shape)}")
        if frames.shape[1] != 3:
            raise ValueError(f"Expected RGB channel dimension at dim=1, got {tuple(frames.shape)}")

        frames = frames[:, :, :: self.frame_stride]
        if masks is not None:
            masks = masks[:, :: self.frame_stride].contiguous()
        frames = frames.permute(0, 2, 1, 3, 4).contiguous()
        return frames, masks

    def _preprocess_pixels(self, pixels):
        if pixels.shape[-2:] != (self.image_size, self.image_size):
            pixels = F.interpolate(
                pixels,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        pixels = pixels.to(dtype=torch.float32)
        mean = self.image_mean.to(device=pixels.device, dtype=pixels.dtype)
        std = self.image_std.to(device=pixels.device, dtype=pixels.dtype).clamp(min=1e-6)
        return (pixels - mean) / std

    def _vision_forward_params(self):
        try:
            return inspect.signature(self.vision_encoder.forward).parameters
        except (TypeError, ValueError):
            return {}

    def _call_processor(self, pixels):
        processor_pixels = pixels.detach().cpu()
        try:
            return self.processor(
                images=processor_pixels,
                return_tensors="pt",
                do_rescale=False,
                input_data_format="channels_first",
            )
        except TypeError:
            return self.processor(
                images=processor_pixels,
                return_tensors="pt",
                do_rescale=False,
            )

    def _build_vision_inputs(self, pixels):
        if self.backend != "transformers":
            return {"pixel_values": pixels}

        params = self._vision_forward_params()
        accepts_kwargs = any(param.kind == param.VAR_KEYWORD for param in params.values())
        processed = self._call_processor(pixels)
        processed = {
            key: value.to(device=pixels.device) if torch.is_tensor(value) else value
            for key, value in processed.items()
        }

        if "attention_mask" in params and "attention_mask" not in processed and "pixel_attention_mask" in processed:
            processed["attention_mask"] = processed["pixel_attention_mask"]
        if (
            "pixel_attention_mask" in params
            and "pixel_attention_mask" not in processed
            and "attention_mask" in processed
        ):
            processed["pixel_attention_mask"] = processed["attention_mask"]
        if "spatial_shapes" in params and "spatial_shapes" not in processed and processed["pixel_values"].dim() == 4:
            _, _, height, width = processed["pixel_values"].shape
            patch_size = max(int(self.patch_size), 1)
            processed["spatial_shapes"] = torch.tensor(
                [height // patch_size, width // patch_size],
                dtype=torch.long,
                device=pixels.device,
            ).repeat(processed["pixel_values"].shape[0], 1)

        if accepts_kwargs:
            return processed

        vision_inputs = {}
        for key, value in processed.items():
            if key in params:
                vision_inputs[key] = value
        if "pixel_values" not in vision_inputs and "pixel_values" in processed:
            vision_inputs["pixel_values"] = processed["pixel_values"]
        return vision_inputs

    def _masked_mean_pool(self, hidden, pixel_attention_mask=None):
        if pixel_attention_mask is None:
            return hidden.mean(dim=1)

        if pixel_attention_mask.dim() == 2:
            patch_mask = pixel_attention_mask.bool()
        else:
            patch_size = max(int(self.patch_size), 1)
            patch_mask = F.avg_pool2d(
                pixel_attention_mask.float().unsqueeze(1),
                kernel_size=patch_size,
                stride=patch_size,
            ).flatten(1)
            patch_mask = patch_mask > 0.5
        if patch_mask.shape[1] + 1 == hidden.shape[1]:
            patch_mask = F.pad(patch_mask, (1, 0), value=True)
        if patch_mask.shape[1] != hidden.shape[1]:
            return hidden.mean(dim=1)

        weights = patch_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

    def _extract_features(self, pixels):
        if self.backend == "tiny":
            return self.vision_encoder(pixels)

        vision_inputs = self._build_vision_inputs(pixels)
        outputs = self.vision_encoder(**vision_inputs)
        pooler = getattr(outputs, "pooler_output", None)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None and isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            hidden = outputs[0]

        if self.pooling == "pooler" and pooler is not None:
            return pooler
        if hidden is None:
            if pooler is not None:
                return pooler
            raise RuntimeError("SigLIP vision encoder did not return last_hidden_state or pooler_output")
        if self.pooling == "cls":
            return hidden[:, 0]
        attention_mask = vision_inputs.get("pixel_attention_mask", vision_inputs.get("attention_mask"))
        return self._masked_mean_pool(hidden, attention_mask)

    def _encode_pixels_chunked(self, pixels):
        outputs = []
        grad_context = torch.no_grad() if self.freeze_vision_encoder else contextlib.nullcontext()
        with grad_context:
            for start in range(0, pixels.shape[0], self.frame_chunk_size):
                chunk = pixels[start : start + self.frame_chunk_size]
                outputs.append(self._extract_features(chunk))
        return torch.cat(outputs, dim=0)

    def _write_selection_meta(self, metas, selected):
        if metas is None:
            return
        for batch_idx, meta in enumerate(metas):
            if not isinstance(meta, dict):
                continue
            mask = selected.selected_masks[batch_idx]
            positions = selected.selected_positions[batch_idx][mask].detach().cpu().tolist()
            meta["irregular_selected_positions"] = [int(pos) for pos in positions]
            meta["irregular_selected_valid_len"] = int(selected.dense_lengths[batch_idx].item())
            meta["irregular_native_axis"] = False
            meta["frame_policy"] = "adaptive_selected_frames"
            if self.return_token_times:
                fps = float(meta.get("fps", -1))
                snippet_stride = int(meta.get("snippet_stride", 1))
                window_start_frame = int(meta.get("window_start_frame", 0))
                offset_frames = int(meta.get("offset_frames", 0))
                if fps > 0:
                    meta["token_times_sec"] = [
                        (int(pos) * snippet_stride + window_start_frame + offset_frames) / fps
                        for pos in positions
                    ]

    def _select_frames_before_vision(self, frames, masks=None, metas=None):
        if self.frame_selector is None:
            return None
        selected = self.frame_selector.select(frames, masks=masks)
        self._write_selection_meta(metas, selected)
        total_dense = frames.shape[0] * frames.shape[1]
        expected = int(selected.selected_masks.sum().item())
        self.last_selector_stats = dict(
            dense_frames=int(total_dense),
            encoded_frames=int(selected.frames.shape[0]),
            selected_slots=expected,
            selected_only=bool(selected.frames.shape[0] < total_dense or self.encode_policy == "selected_only"),
        )
        if self.assert_selected_only:
            if expected > total_dense:
                raise RuntimeError("selected_only selector produced more slots than dense input")
            if selected.frames.shape[0] != expected:
                raise RuntimeError("selected_only selector produced frames that do not match selected mask slots")
        return selected

    def forward(self, frames, masks=None, metas=None):
        frames, masks = self._prepare_frames(frames, masks=masks)
        batch_size, seq_len, channels, height, width = frames.shape
        selected = None
        if self.encode_policy == "selected_only":
            selected = self._select_frames_before_vision(frames, masks=masks, metas=metas)
            if selected is None:
                raise RuntimeError("selected_only requires selected frames at runtime")
        if selected is None:
            pixels = frames.reshape(batch_size * seq_len, channels, height, width)
            output_len = seq_len
            output_masks = masks
            features = None
        else:
            pixels = selected.frames.reshape(selected.frames.shape[0], channels, height, width)
            if self.assert_selected_only:
                expected = int(selected.selected_masks.sum().item())
                if pixels.shape[0] != expected:
                    raise RuntimeError("selected_only invariant failed: pixels.shape[0] != expected")
            output_len = selected.selected_masks.shape[1]
            output_masks = selected.selected_masks
            features = frames.new_zeros((batch_size, output_len, self.embed_dims), dtype=torch.float32)

        if self.backend != "transformers":
            pixels = self._preprocess_pixels(pixels)
        encoded_features = self._encode_pixels_chunked(pixels)
        if features is None:
            features = encoded_features.reshape(batch_size, output_len, self.embed_dims)
        else:
            features = features.to(device=encoded_features.device, dtype=encoded_features.dtype)
            features[selected.batch_indices, selected.slot_indices] = encoded_features
        if self.output_layout == "bct":
            features = features.transpose(1, 2).contiguous()

        if self.motion_branch is not None:
            if self.output_layout != "bct":
                motion_input = features.transpose(1, 2).contiguous()
                motion_output = self.motion_branch(motion_input, masks=output_masks)
                features = motion_output.transpose(1, 2).contiguous()
            else:
                features = self.motion_branch(features, masks=output_masks)
        elif output_masks is not None:
            if self.output_layout == "bct":
                features = features * output_masks.unsqueeze(1).to(dtype=features.dtype)
            else:
                features = features * output_masks.unsqueeze(-1).to(dtype=features.dtype)

        return features.to(torch.float32), output_masks
