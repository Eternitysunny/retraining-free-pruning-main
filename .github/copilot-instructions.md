**Overview**

This repository implements a fast post-training pruning framework for Transformer models (BERT-family, RoBERTa, etc.). The pipeline is organized around: collecting gradient-based importance signals, searching for head/neuron masks under a MAC or latency constraint, rearranging and rescaling masks, and saving final masks (the model weights themselves are not re-trained).

**Quick Commands**

- **Install deps:** `pip install -r requirements.txt` (tested on Python 3.7.10; GPU with 16+ GB recommended)
- **Run a pruning job (example):**
  ```bash
  python main.py --model_name bert-base-uncased \
                 --task_name qqp \
                 --ckpt_dir <HF-or-local-ckpt-dir> \
                 --constraint 0.5
  ```
- **Generate latency LUTs (needed for `--metric latency`):**
  ```bash
  python generate_lut.py --model_name bert-base-uncased --bs 8 --seq_len 128 --output_dir ./luts/bert-8-128
  # outputs: mha_lut.pt, ffn_lut.pt
  ```

**Key Files / Components (quick map)**

- `main.py` — the top-level pruning experiment driver (parses args, loads model & tokenizer, builds sample dataloader, runs the full pipeline and saves `head_mask.pt` / `neuron_mask.pt`).
- `generate_lut.py` — microbenchmarks multi-head-attention (MHA) and FFN operators to build latency lookup tables (`mha_lut.pt`, `ffn_lut.pt`).
- `prune/`:
  - `fisher.py` — collects mask gradients (`collect_mask_grads`) and computes fisher importance.
  - `search.py` — two search strategies: `search_mac` and `search_latency` (uses MAC or LUT-based latency model).
  - `rearrange.py` — greedy rearrangement to reduce interference (`rearrange_mask`).
  - `rescale.py` — least-squares rescaling that adjusts mask scales via `lsmr_cupy_solver`.
- `utils/arch.py` — helpers to access model backbone/layers, hook inputs/outputs, apply masks (`register_mask`, `apply_neuron_mask`, `MaskNeurons`, `hijack_input`, `collect_layer_inputs`). Important: masks are applied via forward pre-hooks.
- `utils/linalg.py` — solvers using `cupy`/`lsmr` for least-squares; `lsmr_cupy_solver` requires GPU-compatible `cupy` build matching CUDA.
- `efficiency/`:
  - `mac.py` — compute MAC counts per head/neuron.
  - `latency.py` — `estimate_latency` and helpers to fit latency functions from LUTs.
- `dataset/` and `evaluate/` — data loaders and evaluation helpers for GLUE / SQuAD tasks.

**Project-specific patterns & conventions**

- Outputs path: experiments write to `outputs/<model_name>/<task_name>/<metric>/<constraint>/seed_<seed>/` (see `main.py` for exact layout). Save artifacts there (saved masks and `log.txt`).
- GPU selection: `main.py` sets `CUDA_VISIBLE_DEVICES` from the `--gpu` arg. Do not assume multi-GPU unless explicitly configured.
- Small-sample search: the code deliberately samples a small subset (`--num_samples`, default 2048) and uses small `sample_batch_size` (in current repo set to `2` in `main.py`) to mimic the paper's low-sample regime — keep that when reproducing results.
- Hook-based masking: masking is implemented with forward pre-hooks (see `utils.arch.register_mask` and `apply_neuron_mask`). Many routines expect to temporarily hijack module inputs (`hijack_input`) and to restore modified `encoder.layers` during `collect_layer_inputs`.
- Rescaling uses GPU cupy lsmr solver: `utils/linalg.py::lsmr_cupy_solver` returns `(solution, success_flag)`; failure is handled by early break in `rescale.py`.

**Integration points / external deps that frequently cause issues**

- HuggingFace Transformers are required (`AutoModel*`, `AutoTokenizer`). Model loading may accept a local HF-style checkpoint dir (needs `config.json` and `pytorch_model.bin`) or a model name string.
- `cupy` is required for `lsmr_cupy_solver`. On machines without a matching `cupy` wheel for the installed CUDA, `rescale_mask` will fail; consider replacing with a CPU fallback for debugging (there is `closed_form_solver` in `utils/linalg.py` but it is not used by default).
- For latency search you must supply precomputed LUTs (`--mha_lut` and `--ffn_lut`) produced by `generate_lut.py` and loaded with `torch.load`.

**Concrete examples an AI coding agent can use**

- To reproduce the paper's MAC-based pruning for QQP:
  - Ensure a HF checkpoint (or `--model_name` recognized by Transformers) and run the `main.py` example above.
- To debug rescaling locally (small run):
  - Run `main.py` with `--num_samples 64 --seed 0 --constraint 0.5` and inspect `log.txt` in the `outputs` folder.
- To generate LUTs (GPU required):
  - `python generate_lut.py --model_name bert-base-uncased --bs 8 --seq_len 128 --output_dir ./luts/bert-8-128` then pass `--mha_lut ./luts/bert-8-128/mha_lut.pt --ffn_lut ./luts/bert-8-128/ffn_lut.pt` to `main.py`.

**Common pitfalls to watch for**

- Missing or incompatible `cupy` causes `rescale_mask` to error. If debugging on CPU-only machine, skip rescale or implement a CPU fallback.
- Passing a checkpoint dir that lacks expected files (`config.json`, `pytorch_model.bin` / `model.safetensors`) will break model loading.
- `collect_layer_inputs` temporarily mutates the model (`encoder.layers`) — do not run parallel calls that assume the model is untouched.
- `run_glue.py` is an HF example script included for dataset/model training and is not the pruning pipeline; use it only when you intend to fine-tune models.

**When to ask for clarification**

- If you need the exact environment used to reproduce a specific experiment (CUDA, cupy wheel, exact HF/transformers version), ask for logs or the `requirements.txt` to pin versions.
- If you want a CPU-only fallback for rescaling or automated test harnesses, say so and I can add a small helper and a smoke test.

If any section is unclear or you want more concrete reproduction instructions (e.g., exact `requirements.txt` pins, or a small CI-friendly smoke test), tell me which part to expand.
