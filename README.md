# NVIDIA Nemotron Reasoning Challenge — Progress Prize

Fine-tuning [NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) on rule-discovery reasoning problems using **deterministic symbolic solvers** to generate verified chain-of-thought training traces. Achieved **LB 0.85**, winning a Progress Prize in the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge).

- [Kaggle write-up](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915)
- [Kaggle notebook](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/code)


## The Core Idea

Each competition problem presents a few input→output examples and a query. The model must infer the hidden rule and produce the correct output inside `\boxed{}`.

The standard approach — prompting a strong LLM to generate reasoning traces and filtering by correctness — produces traces where the intermediate steps are inconsistent across problems, written differently each time, and sometimes subtly wrong even when the final answer is right. Training on a mix of these styles teaches the model to sound plausible rather than to follow a procedure.

Instead: for each problem category, a **deterministic Python solver** mirrors the exact reasoning process step by step, emitting natural-language text as it runs. The solver *is* the trace generator. Every training example is verified (`extract_answer(trace) == ground_truth`) before inclusion. No noisy or hallucinated steps enter training.


## Problem Categories

| Category | Rule type | Solver strategy |
|---|---|---|
| `gravity` | d = k·t² | Compute k per example, take median, multiply through |
| `unit_conversion` | Linear factor | Long-division factor extraction, median, long-multiplication application |
| `numeral` | Arabic → Roman | Greedy subtraction walk through standard Roman value table |
| `cipher` | Substitution cipher | Build bijective letter→letter map word-by-word using *Alice in Wonderland* word list |
| `bit_manipulation` | Per-bit 8-bit operation | Column-matching across Identity / NOT / AND / OR / XOR / shifts with stride extrapolation |
| `equation_numeric` | Arithmetic on parsed expressions | Try common + rare operations (div, mod, digit-level ops) with reversal variants |
| `cryptarithm_deduce` | Symbol→digit mapping + math | 7-family DSL cascade: concat → set ops → position ops → unary ops → char map → bit ops → arithmetic |
| `cryptarithm_guess` | Same, harder search space | Same cascade with global joint digit-mapping solver across operators |


## Pipeline

```
uv run python3 reasoning.py      # 1. Generate verified CoT traces → reasoning/
uv run python3 augmentation.py   # 2. Generate synthetic augmented problems → augmentations/
uv run python3 corpus.py         # 3. Tokenize and assemble training corpus → corpus/ + corpus.jsonl
uv run python3 train_sft.py      # 4. SFT training (Tinker/Modal backend)
uv run modal run upload_adapter.py   # 5. Push LoRA adapter to Kaggle
```

To train on Kaggle T4 GPUs directly instead of step 4–5:

```
# Upload corpus to Kaggle, then run train_kaggle.py as a notebook
uv run python3 upload_to_kaggle_local.py
```


## Training Setup

| Setting | Value |
|---|---|
| Base model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |
| Method | LoRA (rank 32) on attention + MLP + unembedding |
| Optimizer | AdamW β₁=0.9 β₂=0.95 |
| Learning rate | 2×10⁻⁴ → 0 (step-linear decay) |
| Batch size | 64 examples |
| Epochs | 1 |
| Max sequence length | 8192 tokens |
| Loss | Cross-entropy on completion tokens only |

**Token masking**: loss is computed only on the completion (reasoning text + `\boxed{answer}`) — never on the prompt. This matches the inference regime exactly.

**Stratified batching**: examples are grouped by category and distributed round-robin across batches, so each gradient step sees roughly equal representation from all nine categories regardless of their sizes.


## Reasoners

Each category has its own solver in `reasoners/`. The solver produces a natural-language trace as a byproduct of running — there is no separate trace-writing step.

### `gravity` / `unit_conversion`
Compute the constant (k or conversion factor) for each example pair, sort the values, pick the median, and multiply through for the query. Uses long-division and long-multiplication helpers in `store_types.py` to show digit-level arithmetic in the trace.

### `cipher`
Builds a bijective substitution mapping word-by-word. For each cipher word, generates candidates from a word list (derived from *Alice in Wonderland*) whose letter pattern matches and that are consistent with the partial mapping. The trace shows each word, the known mappings at that point, and the candidate chosen.

### `bit_manipulation`
Treats the problem as learning a per-output-bit rule from input bit columns. For each of 8 output bits, considers all families in priority order: Identity, NOT, Constant, AND, OR, XOR, AND-NOT, OR-NOT, XOR-NOT. Finds the longest contiguous run of bits where a single family applies with sequentially-shifting operand indices from both the left and right ends, then extrapolates the stride pattern into remaining slots. A "perfect match" criterion picks the single family that covers all pending bits simultaneously.

### `cryptarithm`
Input format is always `A1A2 OP B1B2 → output`. The solver tries seven rule families in cascade order (cheapest first):

1. **Concat** — 40 rearrangement/subset patterns of [A1, A2, B1, B2]
2. **Set ops** — union, intersection, difference, symmetric difference of character sets
3. **Position ops** — fixed-index subsets: even, odd, outer, inner, first3, last3, no_a1, ...
4. **Unary ops** — reverse, sort, deduplicate applied to the left or right string
5. **Char map** — learn a bijective symbol→symbol substitution from examples (28 key patterns)
6. **Bit ops** — XOR / AND / OR / XNOR / add-mod / sub-mod / shifts on ASCII codes of input chars, across 25 pairing schemes; plus ternary ops (majority, choice, XOR-3, median)
7. **Arithmetic** — symbol→digit mapping + math operation (add, sub, mul, div, mod, abs_diff, gcd, max, min) with reversal variants

When multiple operators share a global digit mapping, a joint solver finds a single consistent assignment across all operators simultaneously, reusing candidate caches from the per-operator cascade.

### `equation_numeric`
Parses each input as `A OP B`, detects the operator character, and tries arithmetic operations in two tiers:

- **Common**: concatenation, add, subtract, multiply, absolute difference
- **Rare**: divide, modulo, multiply±1, add/sub±1, and for 2-digit inputs: digit-level ops (digit absolute diff, add/sub mod 10, cross-multiply, determinant, digit product sum/diff)

Handles problems where negative results are encoded with the operator character as a prefix or suffix (e.g., `5-` instead of `-5`), strips the encoding for matching, and re-applies it in the trace.


## What Didn't Work

**LLM distillation (Claude)**: correct answers but inconsistent trace styles — sometimes working backward, sometimes skipping steps. Training on a mix of styles produced a model that couldn't commit to a single strategy. More critically, intermediate steps were sometimes subtly wrong even when the boxed answer was right.

**Z3 SMT solver**: finds the digit assignment but not the reasoning trace. Reconstructing a natural-language chain-of-thought from Z3's internal proof state is not straightforward, and the resulting traces didn't match how the model should reason at test time.

**AC-3 constraint propagation**: designed for problems with rich initial constraints. In the competition format (4 symbols, 5-char input, 2–5 examples), the initial domains are nearly unconstrained and AC-3 degenerates to backtracking anyway — without solving the trace generation problem.

The key insight across all three: the trace and the solver have to be the same thing. Every step in the trace must correspond to an actual operation in the solver, and the solver must emit the trace as a byproduct of running. This is what the DSL cascade implements.


## Data Augmentation

Five augmenters in `augmenters/` generate synthetic training examples for string-manipulation patterns:

| Augmenter | Rule type |
|---|---|
| `spelling` | Character-level string manipulation |
| `concatenation` | String joining rules |
| `splitting` | Substring extraction |
| `matching` | Pattern matching |
| `lstrip` | Prefix stripping |

These are structurally similar to competition problems but distinct enough to prevent overfitting to the narrow training distribution.


## Analysis Dashboard

A static site with five views, served locally with `./serve.sh`:

| Tab | What it shows |
|---|---|
| **Base** | Problems colored by base-model performance. Click for prompt, per-run extracted answers, and token-level logprob trace. |
| **Synthetic** | Problems colored by solver status (rule found / hypothesis / unknown). Click for reasoning text and investigation notes. |
| **Corpus** | Training corpus entries with masked/unmasked token counts. Open a row to see the token-level trace with masking highlighted. |
| **Training** | Per-problem loss and min logprob across training steps. Compare epochs against base model. |
| **Metrics** | Index of training runs with per-step charts: loss by category, gradient norm, learning rate, step time. |

Run locally:

```sh
./serve.sh
# → http://localhost:33304/
```


## Repository Structure

```
reasoners/          # Per-category solvers (also emit the training trace)
  store_types.py    # Shared Problem/Example types + long arithmetic helpers
  cryptarithm.py    # Fast-path concat + per-op arithmetic solver
  cryptarithm_dsl.py  # Full 7-family DSL cascade + joint arithmetic solver
  bit_manipulation.py
  cipher.py
  equation_numeric.py
  gravity.py / numeral.py / unit_conversion.py

augmenters/         # Synthetic data generators

reasoning.py        # Stage 1: run solvers, write reasoning/ traces
augmentation.py     # Stage 2: run augmenters, write augmentations/
corpus.py           # Stage 3: tokenize + assemble training corpus
train_sft.py        # Stage 4: SFT training loop (Tinker/Modal backend)
train_kaggle.py     # Stage 4 alt: QLoRA on Kaggle T4 GPUs (self-contained notebook)
train_common.py     # Shared data-loading utilities
loss_config.py      # Loss function implementations (CE, IS, PPO, CISPO, DRO)
lr_schedule.py      # Learning rate schedules
trainer/client.py   # ServiceClient wrapping Tinker + Modal backends

upload_adapter.py           # Push LoRA adapter via Modal → Kaggle model registry
upload_to_kaggle_local.py   # Same, direct upload without Modal
generate_csv.py             # Export generation runs to CSV for offline analysis
delete-tinker-checkpoint.sh # Clean up old Tinker checkpoints, keep latest

investigators/      # One-off analysis scripts used during problem investigation

*.html / serve.sh   # Static analysis dashboard
```


## Setup

```sh
# Install dependencies
uv sync

# Secrets (copy and fill in)
cp .env.example .env
```

`.env` needs:
```
TINKER_API_KEY=...
KAGGLE_API_KEY=...
```

The tokenizer is downloaded automatically from HuggingFace on first run of `corpus.py` (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`).
