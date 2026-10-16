# Nemotron-H / Nemotron 3.5 Bolt GGUF support (fork notes)

This fork of [vllm-project/vllm-gguf-plugin](https://github.com/vllm-project/vllm-gguf-plugin) adds support for llama.cpp's Nemotron-H GGUFs (`nemotron_h`, `nemotron_h_moe`), including the vision (Omni) projector. It was built to run NVIDIA's [Nemotron 3.5 Bolt 74B-A9B 24GB GGUF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF) (2-bit, text + image) in vLLM on GPUs older than Blackwell. NVIDIA's native vLLM checkpoint needs `b12x` kernels on SM12x; this route was tested on an NVIDIA A40 (Ampere, sm_86).

It is kept as a working reference. It is not an upstream release and is not affiliated with NVIDIA or the vLLM project.

## Branches

| Branch | Contents | Use it for |
|---|---|---|
| `nemotron-h-support` (default) | Our commits on upstream `main` (`e2b8ad5`) | Reading the changes; a clean base for an upstream PR |
| `bolt-pr141` | The same changes merged onto upstream [PR #141](https://github.com/vllm-project/vllm-gguf-plugin/pull/141) (pinned llama.cpp kernels: MMQ/MMVQ for IQ types) | Serving: about 3x the multi-user throughput |
| `main` | Upstream, unchanged | Syncing with upstream |

Tag `nemotron-bolt-2026-10` marks the tip of `nemotron-h-support`, and `nemotron-bolt-2026-10-pr141` the tip of `bolt-pr141`.

## What the changes add

- **`weights_adapter/nemotron_h.py`**, a new adapter for `nemotron_h`, `nemotron_h_puzzle` and `nemotron_h_omni` configs:
  - maps llama.cpp tensor names to vLLM's `NemotronH*` and `NemotronH_Nano_VL_V2` (Omni) models; per-layer "puzzle" expert counts come from the HF config;
  - undoes llama.cpp's conversion transforms: `ssm_a = -exp(A_log)`, reshaped `ssm_d` and `ssm_norm`, squeezed `ssm_conv1d`. The Llama q/k row permutation is harmless for this model's attention (no rotary embedding), so it is loaded as stored;
  - dequantizes the NVFP4 Mamba `ssm_in`/`ssm_out` tensors to the model dtype and applies their per-tensor `.scale`;
  - maps the mmproj file (C-RADIO vision encoder and `mlp1` projector), re-fusing q/k/v and reshaping the patch embeddings and position embeddings back.
- **Non-gated MoE** (relu2 experts) in the GGUF MoE method: `w13_num_shards` and `MoEActivation.is_gated`.
- **Vec MoE chunking:** the vec MoE path processes large batches in chunks, because its kernels launch `grid.z = tokens * top_k`, capped at 65535.
- **`ReplicatedLinear`:** layers without `weight_loader_v2` get a GGUF-aware loader fallback.
- **Newer vLLM:** the `qwen3_5` adapter falls back to the Transformers config classes.
- **Speculative drafts:** the GGUF config parser keeps draft architectures (DFlash, Eagle, MTP and others), so a safetensors drafter such as NVIDIA's LiLiCorr can run beside a GGUF target.

## Running Nemotron 3.5 Bolt

Tested with vLLM `0.32.0rc1.dev100+g77154ca61` (cu129 wheel), torch 2.13.0+cu129, one A40 48 GB and NVIDIA driver 535. With that driver, use `torchcodec==0.16.0+cu129`.

1. Install vLLM and this plugin (`bolt-pr141` branch; init its `third_party/llama.cpp` submodule), as in the main README. Build for your GPU, e.g. `TORCH_CUDA_ARCH_LIST=8.6`.
2. Download:
   - `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF`: the GGUF and the `mmproj` GGUF;
   - `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB`: config, chat template, remote code and reasoning parser only (the weights are not needed);
   - `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr`: the DFlash drafter.
3. Build an HF config directory without the checkpoint's quantization config:

   ```python
   import json, shutil, pathlib
   src = pathlib.Path("NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB")
   dst = pathlib.Path("bolt-hf-config"); dst.mkdir(exist_ok=True)
   for f in src.iterdir():
       if f.is_file() and not f.name.endswith(".safetensors") and f.name not in (
               "model.safetensors.index.json", "hf_quant_config.json"):
           shutil.copy(f, dst / f.name)
   cfg = json.load(open(src / "config.json"))
   cfg.pop("quantization_config", None)
   cfg.get("llm_config", {}).pop("quantization_config", None)
   json.dump(cfg, open(dst / "config.json", "w"), indent=1)
   ```

4. Serve. Keep the `mmproj` GGUF next to the main GGUF; it is detected automatically.

   ```bash
   export VLLM_USE_V2_MODEL_RUNNER=1
   vllm serve /models/bolt-gguf/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf \
     --hf-config-path ./bolt-hf-config --tokenizer ./bolt-hf-config --trust-remote-code \
     --served-model-name nemotron-bolt --max-model-len 262144 --max-num-seqs 8 \
     --gpu-memory-utilization 0.92 --enable-prefix-caching \
     --limit-mm-per-prompt '{"image":2,"video":0}' \
     --enable-auto-tool-choice --tool-call-parser qwen3_coder \
     --reasoning-parser-plugin ./bolt-hf-config/ultra_v3_reasoning_parser.py --reasoning-parser ultra_v3 \
     --speculative-config '{"method":"dflash","model":"/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr","num_speculative_tokens":7,"rejection_sample_method":"standard","draft_load_config":{"load_format":"auto"}}'
   ```

   - `draft_load_config` keeps the drafter off the GGUF loader.
   - `--max-num-seqs 8` is needed on 48 GB: at 32 sequences the speculative-decoding CUDA graphs take about 9.8 GiB and leave no KV cache.
   - Without the drafter, drop `--speculative-config` and use `--max-num-seqs 32`.

The weights take about 28 GiB, plus about 1 GiB for the drafter. Text, thinking on/off, tool calls and image input were checked.

## Results on one A40 (October 2026)

Speed: output tok/s from `vllm bench serve`, random 1K-token prompts, 256 output tokens. Random prompts lower drafter acceptance; on real prompts a single stream reached about 111 tok/s.

| | 1 user | 8 users | 32 users | TTFT at 128K |
|---|---|---|---|---|
| Plugin `main` kernels | 28 | 57 | - | - |
| PR #141 kernels (`bolt-pr141`) | 38 | 164 | 297 | 34 s |
| PR #141 + LiLiCorr DFlash, 7 tokens | 72 | 198 | (max 8 seqs) | 36 s |
| llama.cpp, same GGUF, 1 slot, drafter | 58 | 69 (4 slots) | - | 72 s |

DFlash acceptance was 48-59%, with about 86-88% at the first draft position.

Accuracy, with the same harness and settings for every model (mostly temperature 0). In tau2 the model plays agent, user and judge.

| | Bolt 24GB GGUF | Qwen3.8-27B INT4 | MiniCPM-V 4.7 35B-A3B GPTQ |
|---|---|---|---|
| tau2 airline / retail / telecom | 0.700 / 0.825 / 0.711 | 0.780 / 0.851 / 0.886 | 0.840 / 0.886 / 0.965 |
| BIRD dev (execution accuracy) | 54.9% | 63.4% | 57.3% |
| OCRBench | 820 | 866 | 888 |
| BFCL simple_python / live_multiple | 93.3 / 78.7 | 92.3 / 77.4 | 95.0 / 82.5 |

The ranking matches the independent [Artificial Analysis Intelligence Index](https://artificialanalysis.ai/models/nemotron-3-5-bolt-74b-a9b): Bolt 21.4, Qwen3.8-27B 33.7. Bolt uses 9B active parameters per token, against 27B for dense Qwen3.8.

## Limitations

- **Slow prefill:** IQ2_XXS has no MMQ path in the plugin's own kernels; PR #141 adds one.
- **Video and audio:** not wired up. Video tensors are mapped, but video was not tested.
- **Multimodal drafting:** the drafter receives text-only inputs for image prompts, so acceptance is lower there.
- **Tensor parallelism:** only `tensor_parallel_size=1` was tested.
