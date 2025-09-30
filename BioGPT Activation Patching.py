!pip -q install transformers accelerate torch datasets pandas matplotlib seaborn sacremoses


from __future__ import annotations
import math, random, time, io, os
from dataclasses import dataclass
from typing import Dict, Optional, List, Set
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from google.colab import files
uploaded = files.upload()
#------------------------------
# Config (easy to tweak)
#------------------------------
MODEL_ID = "microsoft/BioGPT-Large-PubMedQA"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
SEED = 42
MAX_LEN = 512
DEBUG = True #Set to False if you do not want it to print progress at each step (currently it's used mostly to make sure the code is actually running)
set_seed(SEED)

#------------------------------
# Utils
#------------------------------
def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_n = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b_n = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return a_n @ b_n.T

@dataclass
class Alignment:
    mapping: Dict[int, int]

@dataclass
class DonorBank:
    attn: Dict[int, torch.Tensor]
    mlp: Dict[int, torch.Tensor]

def compute_alignment_by_embeddings(tokenizer, model, simple_text: str, complex_text: str):
    tok_simple = tokenizer(simple_text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(model.device)
    tok_complex = tokenizer(complex_text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(model.device)
    embed = model.get_input_embeddings()
    with torch.no_grad():
        emb_simple = embed(tok_simple.input_ids)[0]
        emb_complex = embed(tok_complex.input_ids)[0]
        sims = cosine_sim(emb_simple, emb_complex)  
        best = sims.argmax(dim=-1).tolist()
    mapping = {i: j for i, j in enumerate(best)}
    return Alignment(mapping=mapping), {"tok_simple": tok_simple, "tok_complex": tok_complex, "sims": sims}

def _try_parse_layer_idx(name: str) -> Optional[int]:
    for tok in reversed(name.split(".")):
        if tok.isdigit():
            return int(tok)
    return None

def plot_attention_heatmap(matrix, title):
    plt.figure(figsize=(8,6))
    sns.heatmap(matrix, cmap="viridis", cbar=True)
    plt.title(title)
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.tight_layout()
    plt.show()

def plot_mlp_barplot(values, title):
    plt.figure(figsize=(8,4))
    plt.bar(range(len(values)), values)
    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel("Impact score (ΔPPL)")
    plt.tight_layout()
    plt.show()

#------------------------------
# Hooks: Capture
#------------------------------
class ActivationCatcher:
    """
    Captures post-attention output (reshaped per-head) and post-MLP output (per-layer).
    """
    def __init__(self, num_layers: int, num_heads: int):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.attn: Dict[int, torch.Tensor] = {}
        self.mlp: Dict[int, torch.Tensor] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def _make_attn_hook(self, layer_idx: int):
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            if not torch.is_tensor(hs):
                return out
            b, t, d = hs.shape
            if d % self.num_heads != 0:
                return out
            head_dim = d // self.num_heads
            hs_heads = hs.view(b, t, self.num_heads, head_dim) 
            self.attn[layer_idx] = hs_heads.detach()[0].to("cpu")
            return out
        return hook

    def _make_mlp_hook(self, layer_idx: int):
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            if not torch.is_tensor(hs):
                return out
            self.mlp[layer_idx] = hs.detach()[0].to("cpu")
            return out
        return hook

    def register(self, model):
        for name, module in model.named_modules():
            cls = module.__class__.__name__.lower()
            if "biogpt" in cls and cls.endswith("attention"):
                idx = _try_parse_layer_idx(name)
                if idx is not None:
                    self.handles.append(module.register_forward_hook(self._make_attn_hook(idx)))
            if "biogpt" in cls and cls.endswith("mlp"):
                idx = _try_parse_layer_idx(name)
                if idx is not None:
                    self.handles.append(module.register_forward_hook(self._make_mlp_hook(idx)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def to_bank(self) -> DonorBank:
        return DonorBank(attn=self.attn.copy(), mlp=self.mlp.copy())

#------------------------------
# Hooks: Patch
#------------------------------
class ActivationPatcher:
    #Patches selected layers/heads using a donor bank w/ an alignment map.
    def __init__(self, bank: DonorBank, alignment: Alignment, num_heads: int):
        self.bank = bank
        self.align = alignment.mapping
        self.num_heads = num_heads
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def _patch_tensor_heads(self, current_bthd: torch.Tensor, donor_thd: torch.Tensor, head_mask=None):
        #current_bthd: [B, T, Hn, Hd]
        #donor_thd:   [T_donor, Hn, Hd]
        patched = current_bthd.clone()
        T_cur = current_bthd.shape[1]
        for j in range(T_cur):
            if j in self.align:
                k = self.align[j]
                if 0 <= k < donor_thd.shape[0]:
                    for h in range(self.num_heads):
                        if head_mask is None or head_mask[h]:
                            patched[0, j, h, :] = donor_thd[k, h, :].to(patched.device, dtype=patched.dtype)
        return patched

    def _patch_tensor_seq(self, current_bth: torch.Tensor, donor_th: torch.Tensor):
        #current_bth: [B, T, H]
        #donor_th:   [T_donor, H]
        patched = current_bth.clone()
        T_cur = current_bth.shape[1]
        for j in range(T_cur):
            if j in self.align:
                k = self.align[j]
                if 0 <= k < donor_th.shape[0]:
                    patched[0, j, :] = donor_th[k, :].to(patched.device, dtype=patched.dtype)
        return patched

    def _make_attn_hook(self, idx: int, head_mask=None):
        def hook(module, inp, out):
            donor = self.bank.attn.get(idx)
            if donor is None:
                return out
            hs = out[0] if isinstance(out, tuple) else out  # [B, T, H]
            if not torch.is_tensor(hs):
                return out
            b, t, d = hs.shape
            if d % self.num_heads != 0:
                return out
            hd = d // self.num_heads
            hs_heads = hs.view(b, t, self.num_heads, hd)
            hs_p = self._patch_tensor_heads(hs_heads, donor, head_mask)
            hs_p = hs_p.view(b, t, d)
            return (hs_p, *out[1:]) if isinstance(out, tuple) else hs_p
        return hook

    def _make_mlp_hook(self, idx: int):
        def hook(module, inp, out):
            donor = self.bank.mlp.get(idx)
            if donor is None:
                return out
            hs = out[0] if isinstance(out, tuple) else out
            if not torch.is_tensor(hs):
                return out
            hs_p = self._patch_tensor_seq(hs, donor)
            return (hs_p, *out[1:]) if isinstance(out, tuple) else hs_p
        return hook

    def register(self, model, layers_fraction: float = 1.0, rng=None, head_mask=None, only_layers: Optional[Set[int]] = None):
        if rng is None:
            rng = random.Random(SEED)
        all_layers = sorted(set(self.bank.attn.keys()) | set(self.bank.mlp.keys()))
        if not all_layers:
            return
        if only_layers is not None:
            chosen = set(l for l in all_layers if l in only_layers)
        else:
            k = max(1, int(len(all_layers) * layers_fraction))
            chosen = set(rng.sample(all_layers, k))

        for name, module in model.named_modules():
            cls = module.__class__.__name__.lower()
            if "biogpt" in cls and cls.endswith("attention"):
                idx = _try_parse_layer_idx(name)
                if idx in chosen:
                    self.handles.append(module.register_forward_hook(self._make_attn_hook(idx, head_mask)))
            if "biogpt" in cls and cls.endswith("mlp"):
                idx = _try_parse_layer_idx(name)
                if idx in chosen:
                    self.handles.append(module.register_forward_hook(self._make_mlp_hook(idx)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

#------------------------------
# Perplexity helpers
#------------------------------
def build_prompt_and_labels(tokenizer, prompt: str, target: str):
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    enc_prompt = tokenizer(prompt, add_special_tokens=False).input_ids
    enc_target = tokenizer(target, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    max_prompt_len = MAX_LEN - len(enc_target)
    if max_prompt_len < 0:
        enc_target = enc_target[-MAX_LEN:]
        enc_prompt = []
    else:
        enc_prompt = enc_prompt[:max_prompt_len]

    input_ids = enc_prompt + enc_target
    labels = [-100] * len(enc_prompt) + enc_target

    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "attention_mask": torch.ones(1, len(input_ids), dtype=torch.long),
    }


def perplexity_from_loss(loss: torch.Tensor) -> float:

    return float(torch.exp(loss.detach()).item())
#------------------------------
# Analyses
#------------------------------

def record_results(results, row_idx, patch_type, strategy, layer, head, delta):
    results.append({
        "row": row_idx,
        "patch_type": patch_type,
        "strategy": strategy,
        "layer": layer,
        "head": head,
        "delta_ppl": delta
    })

def per_block_analysis(tokenizer, model, bank, simple_text, complex_text, alignment, num_heads):
    data = build_prompt_and_labels(tokenizer, simple_text, complex_text)
    data = {k: v.to(model.device) for k, v in data.items()}
    with torch.no_grad():
        base = model(**data)
    base_ppl = perplexity_from_loss(base.loss)

    num_layers = model.config.num_hidden_layers
    block_scores = []

    for layer in range(num_layers):
        patcher = ActivationPatcher(bank, alignment, num_heads)
        patcher.register(model, layers_fraction=1.0, only_layers={layer})
        with torch.no_grad():
            out = model(**data)
        patcher.remove()
        delta = perplexity_from_loss(out.loss) - base_ppl
        block_scores.append(delta)
        if DEBUG:
            print(f"[per-block] layer={layer:02d} ΔPPL={delta:.4f}")
    return block_scores


def per_block_random_analysis(tokenizer, model, bank, simple_text, complex_text, alignment, num_heads, seed=42):
    rng = random.Random(seed)
    data = build_prompt_and_labels(tokenizer, simple_text, complex_text)
    data = {k: v.to(model.device) for k, v in data.items()}
    with torch.no_grad():
        base = model(**data)
    base_ppl = perplexity_from_loss(base.loss)

    keys = list(alignment.mapping.keys())
    values = list(alignment.mapping.values())
    rng.shuffle(values)
    rand_mapping = dict(zip(keys, values))
    rand_alignment = Alignment(mapping=rand_mapping)

    num_layers = model.config.num_hidden_layers
    block_scores = []

    for layer in range(num_layers):
        patcher = ActivationPatcher(bank, rand_alignment, num_heads)
        patcher.register(model, layers_fraction=1.0, only_layers={layer})
        with torch.no_grad():
            out = model(**data)
        patcher.remove()
        delta = perplexity_from_loss(out.loss) - base_ppl
        block_scores.append(delta)
        if DEBUG:
            print(f"[random per-block] layer={layer:02d} ΔPPL={delta:.4f}")
    return block_scores


def per_head_random_analysis(tokenizer, model, bank, simple_text, complex_text, alignment, num_heads, seed=42):
    rng = random.Random(seed)
    data = build_prompt_and_labels(tokenizer, simple_text, complex_text)
    data = {k: v.to(model.device) for k, v in data.items()}
    print("Labels:", data["labels"].cpu().tolist())
    print("Valid tokens:", (data["labels"] != -100).sum().item())
    with torch.no_grad():
        base = model(**data)
    base_ppl = perplexity_from_loss(base.loss)

    num_layers = model.config.num_hidden_layers
    attn_matrix = torch.zeros(num_layers, num_heads)
    keys = list(alignment.mapping.keys())
    values = list(alignment.mapping.values())
    rng.shuffle(values)
    rand_mapping = dict(zip(keys, values))
    rand_alignment = Alignment(mapping=rand_mapping)

    for layer in range(num_layers):
        for head in range(num_heads):
            head_mask = [False] * num_heads
            head_mask[head] = True
            patcher = ActivationPatcher(bank, rand_alignment, num_heads)
            patcher.register(model, layers_fraction=1.0, head_mask=head_mask, only_layers={layer})
            with torch.no_grad():
                out = model(**data)
            patcher.remove()
            delta = perplexity_from_loss(out.loss) - base_ppl
            attn_matrix[layer, head] = delta
            if DEBUG:
                print(f"[random per-head] layer={layer:02d} head={head:02d} ΔPPL={delta:.4f}")
    return attn_matrix.cpu().numpy()

def per_head_analysis(tokenizer, model, bank, simple_text, complex_text, alignment, num_heads):
    data = build_prompt_and_labels(tokenizer, simple_text, complex_text)
    data = {k: v.to(model.device) for k, v in data.items()}
    print("Labels:", data["labels"].cpu().tolist())
    print("Valid tokens:", (data["labels"] != -100).sum().item())
    with torch.no_grad():
        base = model(**data)
    base_ppl = perplexity_from_loss(base.loss)

    num_layers = model.config.num_hidden_layers
    attn_matrix = torch.zeros(num_layers, num_heads)

    for layer in range(num_layers):
        for head in range(num_heads):
            head_mask = [False] * num_heads
            head_mask[head] = True
            patcher = ActivationPatcher(bank, alignment, num_heads)
            patcher.register(model, layers_fraction=1.0, head_mask=head_mask, only_layers={layer})
            with torch.no_grad():
                out = model(**data)
            patcher.remove()
            delta = perplexity_from_loss(out.loss) - base_ppl
            attn_matrix[layer, head] = delta
            if DEBUG:
                print(f"[per-head] layer={layer:02d} head={head:02d} ΔPPL={delta:.4f}")
    return attn_matrix.cpu().numpy()

def per_layer_mlp_analysis(tokenizer, model, bank, simple_text, complex_text, alignment):
    data = build_prompt_and_labels(tokenizer, simple_text, complex_text)
    data = {k: v.to(model.device) for k, v in data.items()}
    print("Labels:", data["labels"].cpu().tolist())
    print("Valid tokens:", (data["labels"] != -100).sum().item())
    with torch.no_grad():
        base = model(**data)
    base_ppl = perplexity_from_loss(base.loss)

    num_layers = model.config.num_hidden_layers
    mlp_scores: List[float] = []

    for layer in range(num_layers):
        patcher = ActivationPatcher(bank, alignment, model.config.num_attention_heads)
        patcher.register(model, layers_fraction=1.0, only_layers={layer})
        with torch.no_grad():
            out = model(**data)
        patcher.remove()
        delta = perplexity_from_loss(out.loss) - base_ppl
        mlp_scores.append(delta)
        if DEBUG:
            print(f"[per-mlp] layer={layer:02d} ΔPPL={delta:.4f}")
    return mlp_scores
def per_layer_random_mlp_analysis(tokenizer, model, bank, simple_text, complex_text, alignment, seed=42):
    rng = random.Random(seed)
    data = build_prompt_and_labels(tokenizer, simple_text, complex_text)
    data = {k: v.to(model.device) for k, v in data.items()}
    print("Labels:", data["labels"].cpu().tolist())
    print("Valid tokens:", (data["labels"] != -100).sum().item())
    with torch.no_grad():
        base = model(**data)
    base_ppl = perplexity_from_loss(base.loss)

    num_layers = model.config.num_hidden_layers
    mlp_scores = []
    keys = list(alignment.mapping.keys())
    values = list(alignment.mapping.values())
    rng.shuffle(values)
    rand_mapping = dict(zip(keys, values))
    rand_alignment = Alignment(mapping=rand_mapping)

    for layer in range(num_layers):
        patcher = ActivationPatcher(bank, rand_alignment, model.config.num_attention_heads)
        patcher.register(model, layers_fraction=1.0, only_layers={layer})
        with torch.no_grad():
            out = model(**data)
        patcher.remove()
        delta = perplexity_from_loss(out.loss) - base_ppl
        mlp_scores.append(delta)
        if DEBUG:
            print(f"[random per-mlp] layer={layer:02d} ΔPPL={delta:.4f}")
    return mlp_scores
#------------------------------
# One pair end-to-end
#------------------------------
def run_single_pair(tokenizer, model, complex_text, simple_text, rng=None):
    t0 = time.time()
    alignment, toks = compute_alignment_by_embeddings(tokenizer, model, simple_text, complex_text)
    if DEBUG: print(f"Alignment computed in {time.time()-t0:.2f}s  (|map|={len(alignment.mapping)})")

    catcher = ActivationCatcher(num_layers=model.config.num_hidden_layers,
                                num_heads=model.config.num_attention_heads)
    catcher.register(model)
    with torch.no_grad():
        _ = model(**tokenizer(complex_text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(model.device))
    catcher.remove()
    bank = catcher.to_bank()
    if DEBUG:
        print(f"Captured layers: attn={sorted(bank.attn.keys())}, mlp={sorted(bank.mlp.keys())}")

    data = build_prompt_and_labels(tokenizer, simple_text, complex_text)
    data = {k: v.to(model.device) for k, v in data.items()}
    print("Labels:", data["labels"].cpu().tolist())
    print("Valid tokens:", (data["labels"] != -100).sum().item())
    with torch.no_grad():
        baseline = model(**data)
    baseline_ppl = perplexity_from_loss(baseline.loss)

    patcher = ActivationPatcher(bank, alignment, model.config.num_attention_heads)
    patcher.register(model, layers_fraction=(1/ max(1, model.config.num_hidden_layers)), rng=rng)
    with torch.no_grad():
        patched = model(**data)
    patcher.remove()
    patched_ppl = perplexity_from_loss(patched.loss)

    impact = patched_ppl - baseline_ppl  # positive => harm, negative => helps
    if DEBUG:
        print(f"baseline_ppl={baseline_ppl:.4f} | patched_ppl={patched_ppl:.4f} | impact={impact:+.4f}")

    return {
        "baseline_ppl": baseline_ppl,
        "patched_ppl": patched_ppl,
        "impact_score": impact,
        "alignment": alignment.mapping,
        "sims": toks["sims"].detach().cpu().numpy(),
    }, bank, alignment

#------------------------------
# Main
#------------------------------
#------------------------------
# Main
#------------------------------
import ipywidgets as widgets
from IPython.display import display
button = widgets.Button(description="Click me")
button.on_click(print("BUTTON WAS CLICKED\n"))
display(button)
def main(MAX_ROWS=None):
    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "</s>"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        attn_implementation="eager",
        device_map=None,
    ).to(DEVICE)
    model.eval()
    if os.path.exists("test.csv"):
        df = pd.read_csv("test.csv")
    else:
        df = pd.DataFrame([{
            "input_text": "The patient presents with dyspnea and chest pain suggestive of pulmonary embolism.",
            "target_text": "Patient has shortness of breath and chest pain; possible blood clot in the lung."
        }])

    if MAX_ROWS is not None:
        df = df.iloc[:MAX_ROWS] 

    results = []
    num_heads = model.config.num_attention_heads

    last_bank, last_alignment, last_simple, last_complex = None, None, None, None
    for idx, row in df.iterrows():
        button.click()
        complex_text, simple_text = str(row["input_text"]), str(row["target_text"])
        print(f"\n=== Row {idx} ===")
        res, bank, alignment = run_single_pair(tokenizer, model, complex_text, simple_text)

        num_heads = model.config.num_attention_heads
        num_layers = model.config.num_hidden_layers

        attn_scores = per_head_analysis(tokenizer, model, bank, simple_text, complex_text, alignment, num_heads)
        for l in range(num_layers):
            for h in range(num_heads):

                record_results(results, idx, "attn_head", "sufficiency", l, h, attn_scores[l, h])

        rand_attn_scores = per_head_random_analysis(tokenizer, model, bank, simple_text, complex_text, alignment, num_heads)
        for l in range(num_layers):
            for h in range(num_heads):
                record_results(results, idx, "attn_head", "random", l, h, rand_attn_scores[l, h])

        mlp_scores = per_layer_mlp_analysis(tokenizer, model, bank, simple_text, complex_text, alignment)
        for l, delta in enumerate(mlp_scores):
            record_results(results, idx, "mlp", "sufficiency", l, None, delta)

        rand_mlp_scores = per_layer_random_mlp_analysis(tokenizer, model, bank, simple_text, complex_text, alignment)
        for l, delta in enumerate(rand_mlp_scores):
            record_results(results, idx, "mlp", "random", l, None, delta)

        block_scores = per_block_analysis(tokenizer, model, bank, simple_text, complex_text, alignment, num_heads)
        for l, delta in enumerate(block_scores):
            record_results(results, idx, "attn_block", "sufficiency", l, None, delta)

        rand_block_scores = per_block_random_analysis(tokenizer, model, bank, simple_text, complex_text, alignment, num_heads)
        for l, delta in enumerate(rand_block_scores):
            record_results(results, idx, "attn_block", "random", l, None, delta)

        last_bank, last_alignment = bank, alignment
        last_simple, last_complex = simple_text, complex_text

    #----------------------------------
    # Plots only for the last pair
    #----------------------------------
    if last_bank is not None and last_alignment is not None:
        rand_attn_scores = per_head_random_analysis(tokenizer, model, last_bank, last_simple, last_complex, last_alignment, num_heads)
        attn_scores = per_head_analysis(tokenizer, model, last_bank, last_simple, last_complex, last_alignment, num_heads)
        attn_block_scores = per_block_analysis(tokenizer, model, last_bank, last_simple, last_complex, last_alignment, num_heads)
        rand_attn_block_scores = per_block_random_analysis(tokenizer, model, last_bank, last_simple, last_complex, last_alignment, num_heads)

        mlp_scores = per_layer_mlp_analysis(tokenizer, model, last_bank, last_simple, last_complex, last_alignment)
        rand_mlp_scores = per_layer_random_mlp_analysis(tokenizer, model, last_bank, last_simple, last_complex, last_alignment)

    outdf = pd.DataFrame(results)
    outdf.to_csv("BioGPT_activation_patching_full_results.csv", index=False)
    print("\nSaved detailed results to BioGPT_activation_patching_full_results.csv")
    print(outdf.head())


if __name__ == "__main__":
    main(MAX_ROWS=148) #There are 148 rows in test.csv so this takes in the whole CSV. But for testing purposes, this can be set lower
