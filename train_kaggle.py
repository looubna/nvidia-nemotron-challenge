# %% [markdown]
# # Nemotron-30B QLoRA Training on Kaggle
#
# Requirements (add to notebook):
# - Model: metric/nemotron-3-nano-30b-a3b-bf16/transformers/default
# - Dataset: (your upload) nemotron-corpus  → contains corpus.jsonl + corpus/ dir
# - GPU: T4 x2 (recommended) or P100 16GB (tight)

# %% [code]
import subprocess, sys

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "peft>=0.10.0", "bitsandbytes>=0.43.0", "transformers>=4.40.0",
    "accelerate>=0.29.0", "safetensors",
], check=True)

# %% [code]
import json, os, glob, zipfile, shutil
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH   = "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1"
CORPUS_INDEX = "/kaggle/input/nemotron-corpus/corpus.jsonl"
CORPUS_DIR   = "/kaggle/input/nemotron-corpus/corpus"
OUTPUT_DIR   = Path("/kaggle/working/adapter")

LORA_RANK       = 32
LORA_ALPHA      = 64
MAX_SEQ_LENGTH  = 1024     # truncate to fit in 16 GB VRAM
MICRO_BATCH     = 1
GRAD_ACCUM      = 64       # effective batch 64 (matches Tinker config)
LEARNING_RATE   = 2e-4
NUM_EPOCHS      = 1

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "down_proj", "out_proj",
    "in_proj",   # Mamba layers
    "lm_head",
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load corpus entries

# %% [code]
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

corpus_entries = [e for e in load_jsonl(CORPUS_INDEX) if e.get("included", True)]
print(f"Loaded {len(corpus_entries)} corpus entries")


class CorpusDataset(Dataset):
    def __init__(self, entries, corpus_dir, max_len=MAX_SEQ_LENGTH):
        self.entries    = entries
        self.corpus_dir = Path(corpus_dir)
        self.max_len    = max_len

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry    = self.entries[idx]
        seg_path = self.corpus_dir / entry["problem_id"] / entry["segment"]
        segments = load_jsonl(seg_path)

        tokens, mask = [], []
        for seg in segments:
            t = seg["tokens"]
            m = 1 if seg["type"] == "unmasked" else 0
            tokens.extend(t)
            mask.extend([m] * len(t))

        # Truncate
        tokens = tokens[: self.max_len]
        mask   = mask[: self.max_len]

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels    = torch.tensor(tokens[1:],  dtype=torch.long)
        loss_mask = torch.tensor(mask[1:],    dtype=torch.float)

        # Zero out masked positions
        labels = labels.masked_fill(loss_mask == 0, -100)
        return {"input_ids": input_ids, "labels": labels}


def collate_fn(batch):
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels    = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_ids"].shape[0]
        input_ids[i, :n] = b["input_ids"]
        labels[i, :n]    = b["labels"]
    return {"input_ids": input_ids, "labels": labels}


dataset = CorpusDataset(corpus_entries, CORPUS_DIR)
loader  = DataLoader(
    dataset,
    batch_size=MICRO_BATCH,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True,
)
print(f"Dataset: {len(dataset)} examples | {len(loader)} micro-batches per epoch")

# %% [markdown]
# ## Load model (4-bit QLoRA)

# %% [code]
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("Loading model in 4-bit …")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
model.config.use_cache = False
print(f"Model loaded  gpu_mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# %% [markdown]
# ## Apply LoRA

# %% [code]
# Only target modules that actually exist in the model
all_names = {name.split(".")[-1] for name, _ in model.named_modules()}
active_targets = [t for t in TARGET_MODULES if t in all_names]
print(f"LoRA target_modules: {active_targets}")

lora_cfg = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=active_targets,
    lora_dropout=0.0,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()
model.gradient_checkpointing_enable()

# %% [markdown]
# ## Training loop

# %% [code]
optimizer = AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.0,
)

device = next(model.parameters()).device
total_micro_steps = len(loader) * NUM_EPOCHS
total_steps = (total_micro_steps + GRAD_ACCUM - 1) // GRAD_ACCUM

print(f"Training: {total_steps} optimizer steps  ({total_micro_steps} micro-batches)")

model.train()
step, micro_step, running_loss = 0, 0, 0.0

for epoch in range(NUM_EPOCHS):
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss / GRAD_ACCUM
        loss.backward()
        running_loss += loss.item()
        micro_step   += 1

        if micro_step % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step % 10 == 0:
                mem = torch.cuda.memory_allocated() / 1e9
                print(f"step {step}/{total_steps}  loss={running_loss:.4f}  gpu={mem:.1f}GB")
                running_loss = 0.0

# flush any remaining gradient accumulation
if micro_step % GRAD_ACCUM != 0:
    optimizer.step()
    optimizer.zero_grad()

print("Training complete.")

# %% [markdown]
# ## Save adapter

# %% [code]
model.save_pretrained(str(OUTPUT_DIR))
print("Adapter files saved:")
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {f.name}  ({f.stat().st_size/1e6:.1f} MB)")

# %% [markdown]
# ## Fix adapter_config.json target_modules to match competition format

# %% [code]
cfg_path = OUTPUT_DIR / "adapter_config.json"
with open(cfg_path) as f:
    adapter_cfg = json.load(f)

# Ensure target_modules matches what vLLM / scoring expects
adapter_cfg["target_modules"] = active_targets
with open(cfg_path, "w") as f:
    json.dump(adapter_cfg, f, indent=2)

print("adapter_config.json:")
print(json.dumps(adapter_cfg, indent=2))

# %% [markdown]
# ## Create submission.zip

# %% [code]
zip_path = Path("/kaggle/working/submission.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in OUTPUT_DIR.iterdir():
        zf.write(f, arcname=f.name)

print(f"\nsubmission.zip  ({zip_path.stat().st_size/1e6:.1f} MB)")
print("Done. Submit /kaggle/working/submission.zip to the competition.")
