# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from transformers import PretrainedConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from vllm.transformers_utils.config import HFConfigParser
from vllm.transformers_utils.config_parser_base import ConfigParserBase

from .gguf_utils import (
    check_gguf_file,
    is_gguf,
    is_remote_gguf,
    maybe_patch_hf_config_from_gguf,
    split_remote_gguf,
)
from .weights_adapter import get_adapter_architecture


_DRAFT_ARCH_MARKERS = ("DraftModel", "MTP", "Eagle", "DFlash", "DSpark", "Medusa")


def _is_draft_architecture(architecture: str) -> bool:
    return any(marker in architecture for marker in _DRAFT_ARCH_MARKERS)


class GGUFConfigParser(ConfigParserBase):
    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs,
    ) -> tuple[dict, PretrainedConfig]:
        original_model = model
        resolved_model = self._resolve_config_source(model)
        config_dict, config = HFConfigParser().parse(
            resolved_model,
            trust_remote_code=trust_remote_code,
            revision=revision,
            code_revision=code_revision,
            **kwargs,
        )

        # Speculative drafts (e.g. a DFlash/LiLiCorr safetensors draft beside a GGUF
        # target) inherit the target's GGUF config format but are not GGUF models;
        # keep their declared architecture.
        if any(_is_draft_architecture(a) for a in config_dict.get("architectures") or []):
            return config_dict, config

        if config.model_type == "qwen3_moe" and "norm_topk_prob" not in config_dict:
            config_dict["norm_topk_prob"] = True
            config.update({"norm_topk_prob": True})

        architecture = get_adapter_architecture(config)
        if (
            architecture is None
            and config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        ):
            architecture = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]
        if architecture is None:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")

        config_dict["architectures"] = [architecture]
        config.update({"architectures": [architecture]})

        if is_gguf(original_model):
            config = maybe_patch_hf_config_from_gguf(str(original_model), config)

        return config_dict, config

    @staticmethod
    def _resolve_config_source(model: str | Path) -> str | Path:
        if check_gguf_file(model):
            return Path(model).parent
        if is_remote_gguf(model):
            repo_id, _ = split_remote_gguf(model)
            return repo_id
        return model
