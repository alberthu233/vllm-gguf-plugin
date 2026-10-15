# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nemotron-H (hybrid Mamba-2 / attention / latent MoE) GGUF adapter.

Covers llama.cpp's ``nemotron_h`` and ``nemotron_h_moe`` GGUF architectures,
including heterogeneous "puzzle" checkpoints such as Nemotron 3.5 Bolt, whose
per-layer expert counts come from the HF config's ``block_configs``.

Tensor transforms mirror llama.cpp ``conversion/nemotron.py``:
``ssm_a`` stores ``-exp(A_log)``, ``ssm_d``/``ssm_a`` are ``[n_head, 1]``,
``ssm_norm`` is reshaped to ``[n_group, d_inner / n_group]`` and ``ssm_conv1d``
is squeezed. q/k rows carry the Llama rotary permutation, which leaves NoPE
attention scores unchanged, so they are loaded as stored.

NVFP4 tensors (used for the Mamba in/out projections) and their per-tensor
``.scale`` sidecars are dequantized to the model dtype at load time, because the
GGUF kernels have no NVFP4 path.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

import gguf
import numpy as np
import torch
from vllm.logger import init_logger

from ..gguf_files import GGUFModelFiles
from ..weight_utils import get_gguf_tensor_names, split_stacked_experts
from .base import BaseGGUFWeightsAdapter, GGUFWeight

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig

logger = init_logger(__name__)

NEMOTRON_H_ARCHITECTURES = {
    "nemotron_h": "NemotronHForCausalLM",
    "nemotron_h_puzzle": "NemotronHPuzzleForCausalLM",
    # Omni (vision) wrapper; the backbone loads under ``language_model.``
    "nemotron_h_omni": "NemotronH_Omni_Reasoning_V3",
}
_OMNI_MODEL_TYPES = ("nemotron_h_omni",)

# Per-block GGUF suffix -> vLLM/HF suffix (exact match; ``ssm_d`` is a prefix of
# ``ssm_dt.bias`` so substring mapping is unsafe here).
_LAYER_SUFFIXES: dict[str, str] = {
    "attn_norm.weight": "norm.weight",
    # Mamba-2 mixer
    "ssm_in.weight": "mixer.in_proj.weight",
    "ssm_in.scale": "mixer.in_proj.gguf_scale",
    "ssm_out.weight": "mixer.out_proj.weight",
    "ssm_out.scale": "mixer.out_proj.gguf_scale",
    "ssm_conv1d.weight": "mixer.conv1d.weight",
    "ssm_conv1d.bias": "mixer.conv1d.bias",
    "ssm_dt.bias": "mixer.dt_bias",
    "ssm_a": "mixer.A_log",
    "ssm_d": "mixer.D",
    "ssm_norm.weight": "mixer.norm.weight",
    # Attention mixer
    "attn_q.weight": "mixer.q_proj.weight",
    "attn_k.weight": "mixer.k_proj.weight",
    "attn_v.weight": "mixer.v_proj.weight",
    "attn_output.weight": "mixer.o_proj.weight",
    # Dense MLP mixer (non-MoE Nemotron-H)
    "ffn_up.weight": "mixer.up_proj.weight",
    "ffn_down.weight": "mixer.down_proj.weight",
    # Latent MoE mixer
    "ffn_gate_inp.weight": "mixer.gate.weight",
    "exp_probs_b.bias": "mixer.gate.e_score_correction_bias",
    "ffn_latent_down.weight": "mixer.fc1_latent_proj.weight",
    "ffn_latent_up.weight": "mixer.fc2_latent_proj.weight",
    "ffn_up_shexp.weight": "mixer.shared_experts.up_proj.weight",
    "ffn_down_shexp.weight": "mixer.shared_experts.down_proj.weight",
    "ffn_up_exps.weight": "mixer.experts.0.up_proj.weight",
    "ffn_down_exps.weight": "mixer.experts.0.down_proj.weight",
}

_GLOBAL_NAMES: dict[str, str] = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm_f.weight",
    "output.weight": "lm_head.weight",
}

_BLOCK_RE = re.compile(r"blk\.(\d+)\.(.+)")

# mmproj (llama.cpp ``clip`` / ``nemotron_omni`` projector) -> HF Omni names.
_RADIO = "vision_model.radio_model.model."
_VISION_GLOBAL_NAMES: dict[str, str] = {
    "mm.model.mlp.0.weight": "mlp1.0.weight",
    "mm.model.mlp.1.weight": "mlp1.1.weight",
    "mm.model.mlp.3.weight": "mlp1.3.weight",
    "v.class_embd": _RADIO + "patch_generator.cls_token.token",
    "v.patch_embd.weight": _RADIO + "patch_generator.embedder.weight",
    "v.patch_embd.weight.1": _RADIO + "patch_generator.video_embedder.weight",
    "v.position_embd.weight": _RADIO + "patch_generator.pos_embed",
}
_VISION_BLOCK_SUFFIXES: dict[str, str] = {
    # q/k/v are split in GGUF; tagged here and re-fused in transform_weights.
    "attn_q.weight": "attn.qkv.weight" + "#q",
    "attn_k.weight": "attn.qkv.weight" + "#k",
    "attn_v.weight": "attn.qkv.weight" + "#v",
    "attn_q.bias": "attn.qkv.bias" + "#q",
    "attn_k.bias": "attn.qkv.bias" + "#k",
    "attn_v.bias": "attn.qkv.bias" + "#v",
    "attn_out.weight": "attn.proj.weight",
    "attn_out.bias": "attn.proj.bias",
    "ffn_up.weight": "mlp.fc1.weight",
    "ffn_up.bias": "mlp.fc1.bias",
    "ffn_down.weight": "mlp.fc2.weight",
    "ffn_down.bias": "mlp.fc2.bias",
    "ln1.weight": "norm1.weight",
    "ln1.bias": "norm1.bias",
    "ln2.weight": "norm2.weight",
    "ln2.bias": "norm2.bias",
}
_VISION_BLOCK_RE = re.compile(r"v\.blk\.(\d+)\.(.+)")
_QKV_ORDER = ("q", "k", "v")
_SCALE_SUFFIX = ".gguf_scale"
_NVFP4 = int(gguf.GGMLQuantizationType.NVFP4)


def _dequantize_nvfp4(raw: torch.Tensor) -> torch.Tensor:
    data = raw.numpy() if isinstance(raw, torch.Tensor) else np.asarray(raw)
    out = gguf.quants.dequantize(data, gguf.GGMLQuantizationType.NVFP4)
    return torch.from_numpy(np.ascontiguousarray(out))


class NemotronHGGUFAdapter(BaseGGUFWeightsAdapter):
    """Nemotron-H / Nemotron-H MoE (incl. puzzle) GGUF adapter."""

    # Dequantized NVFP4 projections are loaded as plain weights.
    extra_unquantized_modules = ("mixer.in_proj", "mixer.out_proj")

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type in NEMOTRON_H_ARCHITECTURES

    @classmethod
    def architecture(cls, config) -> str | None:
        return NEMOTRON_H_ARCHITECTURES.get(config.model_type)

    def patch_hf_config(
        self,
        files: GGUFModelFiles,
        hf_config: PretrainedConfig,
    ) -> PretrainedConfig:
        del files
        hf_config.architectures = [NEMOTRON_H_ARCHITECTURES[hf_config.model_type]]
        return hf_config

    def build_name_map(
        self,
        files: GGUFModelFiles,
        model_config: ModelConfig,
    ) -> dict[str, str]:
        is_omni = model_config.hf_config.model_type in _OMNI_MODEL_TYPES
        llm_prefix = "language_model." if is_omni else ""
        name_map: dict[str, str] = {}
        unmapped: list[str] = []
        for name in sorted(get_gguf_tensor_names(files.backbone)):
            if name in _GLOBAL_NAMES:
                name_map[name] = llm_prefix + _GLOBAL_NAMES[name]
                continue
            match = _BLOCK_RE.fullmatch(name)
            suffix = _LAYER_SUFFIXES.get(match.group(2)) if match else None
            if suffix is None:
                unmapped.append(name)
                continue
            name_map[name] = f"{llm_prefix}model.layers.{match.group(1)}.{suffix}"
        if is_omni and files.mm_proj is not None:
            for name in sorted(get_gguf_tensor_names([files.mm_proj])):
                if name in _VISION_GLOBAL_NAMES:
                    name_map[name] = _VISION_GLOBAL_NAMES[name]
                    continue
                match = _VISION_BLOCK_RE.fullmatch(name)
                suffix = _VISION_BLOCK_SUFFIXES.get(match.group(2)) if match else None
                if suffix is None:
                    unmapped.append(name)
                    continue
                name_map[name] = f"{_RADIO}blocks.{match.group(1)}.{suffix}"
        elif is_omni:
            logger.warning("Nemotron Omni config but no mmproj GGUF; vision disabled")
        if unmapped:
            logger.warning(
                "No HF name for %d Nemotron-H GGUF tensor(s), skipping: %s",
                len(unmapped),
                unmapped[:20],
            )
        return name_map

    def transform_weights(
        self,
        weights: Iterable[GGUFWeight],
        model_config: ModelConfig,
    ) -> Iterable[GGUFWeight]:
        dtype = model_config.dtype

        def transformed() -> Iterable[GGUFWeight]:
            qkv_parts: dict[str, dict[str, torch.Tensor]] = {}
            nvfp4_bases: set[str] = set()
            # base -> dequantized weight or scale, whichever arrived first
            pending_weight: dict[str, torch.Tensor] = {}
            pending_scale: dict[str, torch.Tensor] = {}
            for name, weight in weights:
                if name.endswith(".weight_type"):
                    base = name.removesuffix(".weight_type")
                    if int(weight.item()) == _NVFP4:
                        nvfp4_bases.add(base)
                        continue
                    yield name, weight
                    continue

                if "#" in name:
                    fused, part = name.split("#")
                    parts = qkv_parts.setdefault(fused, {})
                    parts[part] = weight
                    if len(parts) == len(_QKV_ORDER):
                        del qkv_parts[fused]
                        yield fused, torch.cat([parts[p] for p in _QKV_ORDER], dim=0)
                    continue
                if name.endswith("patch_generator.pos_embed"):
                    yield name, weight.unsqueeze(0)
                    continue
                if name.endswith(("patch_generator.embedder.weight",
                                  "patch_generator.video_embedder.weight")):
                    yield name, weight.reshape(weight.shape[0], -1)
                    continue

                if name.endswith(_SCALE_SUFFIX):
                    base = name.removesuffix(_SCALE_SUFFIX)
                    scale = weight.reshape(-1)[0].float()
                    if base in pending_weight:
                        w = pending_weight.pop(base)
                        yield f"{base}.weight", (w * scale).to(dtype)
                    else:
                        pending_scale[base] = scale
                    continue

                base = name.removesuffix(".weight")
                if base in nvfp4_bases:
                    w = _dequantize_nvfp4(weight)
                    if base in pending_scale:
                        scale = pending_scale.pop(base)
                        yield name, (w * scale).to(dtype)
                    else:
                        pending_weight[base] = w
                    continue

                if name.endswith(".A_log"):
                    # GGUF stores A = -exp(A_log) as [n_head, 1]
                    yield name, torch.log(-weight.float().reshape(-1))
                    continue
                if name.endswith((".D", ".dt_bias", "mixer.norm.weight")):
                    yield name, weight.reshape(-1)
                    continue
                if name.endswith("mixer.conv1d.weight") and weight.dim() == 2:
                    yield name, weight.unsqueeze(1)
                    continue
                yield name, weight

            if qkv_parts:
                logger.warning("Incomplete vision q/k/v groups: %s", list(qkv_parts))
            for base, w in pending_weight.items():
                logger.warning("NVFP4 tensor %s had no .scale; using 1.0", base)
                yield f"{base}.weight", w.to(dtype)

        yield from split_stacked_experts(transformed())
