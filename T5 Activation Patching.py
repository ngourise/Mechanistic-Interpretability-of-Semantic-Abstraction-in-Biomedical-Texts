'''We originally had one code file to run activation patching for all of the models but were having trouble debugging,
so we made a file for each model type and ran them separately. This may explain why you see code for other model types
as you go through this file.'''


# ------------------ Colab Setup ------------------
!pip install torch transformers pandas matplotlib
from google.colab import files
import pandas as pd
import re, os
from torch.nn import ModuleList

DEBUG_T5 = False          #set False after we’re done
DEBUG_LAYERS_TO_PEEK = 2 #how many encoder blocks to print/trace when debugging
print("Please upload your val.csv file (with columns: 'input_text', 'target_text')")
uploaded = files.upload()

if not uploaded:
    raise FileNotFoundError("No file uploaded.")
DATA_CSV = list(uploaded.keys())[0]
# ------------------ Activation Patching Script ------------------
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import os, math, re, warnings
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,)

# --------- USER CONFIG ---------
MODEL_LIST = [
    # "dmis-lab/biobert-base-cased-v1.1",   # encoder-only
    # "allenai/scibert_scivocab_uncased",   # encoder-only
      "hossboll/clinical-t5"               # encoder–decoder
    # "microsoft/BioGPT-Large-PubMedQA"    # decoder-only (BioGPT)
]
MAX_PAIRS = None           # set to None for all pairs
GENERATE_TEXT = False    # optional text generation
MAX_NEW_TOKENS = 64
NUM_BEAMS = 4
OUT_DIR = "patching_outputs"
SEED = 42
# -------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def masked_mean(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    m = attn_mask.unsqueeze(-1).float()
    num = (last_hidden * m).sum(dim=1)
    den = m.sum(dim=1).clamp_min(1e-6)
    return num / den

def detect_arch(model: PreTrainedModel) -> str:
    cfg = getattr(model, "config", None)
    if getattr(cfg, "is_encoder_decoder", False):
        return "encdec"
    mt = str(getattr(cfg, "model_type", "")).lower()
    name = model.__class__.__name__.lower()
    if any(k in mt for k in ["gpt2", "gpt", "biogpt", "llama", "mistral", "qwen"]) or "gpt" in name:
        return "decoder"
    return "encoder"

def load_model_and_tokenizer(model_name: str) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, str]:
    lname = model_name.lower()
    device = get_device()

    if any(k in lname for k in ["t5", "bart", "mbart", "m2m"]):
        model_cls = AutoModelForSeq2SeqLM
    elif any(k in lname for k in ["gpt", "llama", "mistral", "qwen", "biogpt"]):
        model_cls = AutoModelForCausalLM
    else:
        model_cls = AutoModelForMaskedLM

    try:
        tok = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    model = model_cls.from_pretrained(model_name, output_hidden_states=True)
    arch = detect_arch(model)

    # BioGPT & friends: ensure pad/eos exist
    if arch in ["decoder", "encdec"]:
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        if getattr(model.config, "pad_token_id", None) is None and tok.pad_token_id is not None:
            model.config.pad_token_id = tok.pad_token_id
        if getattr(model.config, "eos_token_id", None) is None and tok.eos_token_id is not None:
            model.config.eos_token_id = tok.eos_token_id
        if getattr(model.config, "bos_token_id", None) is None and tok.bos_token_id is not None:
            model.config.bos_token_id = tok.bos_token_id

    model = model.to(device).eval()
    return model, tok, arch

def get_blocks(model: PreTrainedModel, arch: str, which: Optional[str] = None) -> List[nn.Module]:
    m = model
    if arch == "encdec":
        if which == "encoder":
            if hasattr(m, "model") and hasattr(m.model, "encoder") and hasattr(m.model.encoder, "layers"):
                return list(m.model.encoder.layers)  # BART-like
            if hasattr(m, "encoder") and hasattr(m.encoder, "block"):
                return list(m.encoder.block)         # T5-like
        else:
            if hasattr(m, "model") and hasattr(m.model, "decoder") and hasattr(m.model.decoder, "layers"):
                return list(m.model.decoder.layers)
            if hasattr(m, "decoder") and hasattr(m.decoder, "block"):
                return list(m.decoder.block)
        return []
    if arch == "decoder":
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):  # GPT-2
            return list(m.transformer.h)
        if hasattr(m, "model") and hasattr(m.model, "layers"):         # LLaMA-ish
            return list(m.model.layers)
        return []
    # encoder-only
    if hasattr(m, "bert") and hasattr(m.bert, "encoder") and hasattr(m.bert.encoder, "layer"):
        return list(m.bert.encoder.layer)          # BERT
    if hasattr(m, "roberta") and hasattr(m.roberta, "encoder") and hasattr(m.roberta.encoder, "layer"):
        return list(m.roberta.encoder.layer)       # RoBERTa
    if hasattr(m, "encoder") and hasattr(m.encoder, "layer"):
        return list(m.encoder.layer)               # DeBERTa-ish
    return []


def block_parts(block: nn.Module, arch: str) -> Dict[str, Optional[nn.Module]]:
    """
    Choose modules to hook. For T5, hook the wrapper layers so we get
    residual-applied hidden_states tensors in hook outputs.
    """
    out = {"attn": None, "mlp": None}
    name = block.__class__.__name__.lower()

    # ----- T5 encoder/decoder block -----
    # encoder: block.layer = [T5LayerSelfAttention, T5LayerFF]
    # decoder: block.layer = [T5LayerSelfAttention, T5LayerCrossAttention, T5LayerFF]
    # We hook the wrapper modules (layer[0] and layer[-1]) because they return tensor outputs.
    if hasattr(block, "layer") and isinstance(block.layer, (list, tuple, ModuleList)) and "t5" in name:
        try:
            layers = list(block.layer)
            if len(layers) >= 1 and isinstance(layers[0], nn.Module):
                out["attn"] = layers[0]      # wrapper for self-attn
            if len(layers) >= 1 and isinstance(layers[-1], nn.Module):
                out["mlp"]  = layers[-1]     # wrapper for FFN
            return out
        except Exception:
            # fall through to generic heuristics below if anything odd happens
            pass

    # ----- BART / mBART -----
    if hasattr(block, "self_attn"):
        out["attn"] = block.self_attn
    if hasattr(block, "fc2"):
        out["mlp"] = block.fc2

    # ----- GPT-2 style -----
    if hasattr(block, "attn"):
        out["attn"] = block.attn
    if hasattr(block, "mlp"):
        out["mlp"] = block.mlp

    # ----- LLaMA / Mistral style -----
    if hasattr(block, "self_attn"):
        out["attn"] = block.self_attn
    if hasattr(block, "mlp"):
        out["mlp"] = block.mlp

    # ----- BERT / RoBERTa -----
    if hasattr(block, "attention") and hasattr(block.attention, "self"):
        out["attn"] = block.attention.self
    if hasattr(block, "output") and hasattr(block.output, "dense"):
        out["mlp"] = block.output.dense

    return out



def num_heads(attn_module: nn.Module) -> Optional[int]:

    inner = getattr(attn_module, "SelfAttention", None)
    cand = inner if isinstance(inner, nn.Module) else attn_module

    for attr in ["num_attention_heads", "num_heads", "n_head", "n_heads"]:
        if hasattr(cand, attr):
            try:
                return int(getattr(cand, attr))
            except Exception:
                pass

    cfg = getattr(cand, "config", None)
    for attr in ["num_attention_heads", "num_heads", "n_head", "n_heads"]:
        if cfg is not None and hasattr(cfg, attr):
            try:
                return int(getattr(cfg, attr))
            except Exception:
                pass

    return None



def align_tokens_by_input_embeddings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    donor_text: str,
    recipient_text: str,
) -> List[int]:
    device = get_device()
    emb: nn.Embedding = model.get_input_embeddings()  # type: ignore
    if emb is None:
        raise RuntimeError("Model has no input embedding layer")

    donor = tokenizer(donor_text, return_tensors="pt", truncation=True, padding=False, max_length=512).to(device)
    recip = tokenizer(recipient_text, return_tensors="pt", truncation=True, padding=False, max_length=512).to(device)

    with torch.no_grad():
        dE = F.normalize(emb(donor["input_ids"]), dim=-1)  # [1,Sd,H]
        rE = F.normalize(emb(recip["input_ids"]), dim=-1)  # [1,Sr,H]
        S = rE @ dE.transpose(1, 2)                        # [1,Sr,Sd]

    special_ids = set(tokenizer.all_special_ids or [])
    d_ids = donor["input_ids"][0].tolist()
    r_ids = recip["input_ids"][0].tolist()
    align = []
    for j, tid in enumerate(r_ids):
        if tid in special_ids:
            candidates = [i for i, did in enumerate(d_ids) if did == tid]
            align.append(candidates[0] if candidates else int(S[0, j].argmax().item()))
        else:
            align.append(int(S[0, j].argmax().item()))
    return align

@dataclass
class DonorBank:
    tensors: Dict[Tuple[str, int, Optional[int]], torch.Tensor] = field(default_factory=dict)
    def put(self, kind: str, layer: int, tensor: torch.Tensor, head: Optional[int] = None):
        self.tensors[(kind, int(layer), None if head is None else int(head))] = tensor.detach()
    def get(self, kind: str, layer: int, head: Optional[int] = None) -> Optional[torch.Tensor]:
        return self.tensors.get((kind, int(layer), None if head is None else int(head)))

@dataclass
class PatchSpec:
    entries: List[Tuple[int, str, Optional[int]]]  # (layer_idx, 'attn'|'mlp', head_idx|None)

def _split_hook_out(out):
    if out is None:
        return None, (), False
    if isinstance(out, (list, tuple)):
        if len(out) == 0:
            return None, tuple(), True
        return out[0], tuple(out[1:]), True
    return out, tuple(), False

def _dbg_peek_block(block, li):
    cls = block.__class__.__name__
    has_layer = hasattr(block, "layer")
    layer_type = type(block.layer).__name__ if has_layer else None
    print(f"[PEEK] Enc L{li}: {cls}, has layer={has_layer}, layer_type={layer_type}")
    if has_layer:
        try:
            L = list(block.layer)
            names = [c.__class__.__name__ for c in L]
            print(f"       layer len={len(L)}: {names}")
            # common T5 internals if present
            try:
                sa_inner = getattr(L[0], "SelfAttention", None)
                ff_inner = getattr(L[-1], "DenseReluDense", None)
                print(f"       inner: SelfAttention={type(sa_inner).__name__}, DenseReluDense={type(ff_inner).__name__}")
            except Exception as e:
                print(f"       [PEEK] inner inspect error: {e}")
        except Exception as e:
            print(f"[PEEK] error listing .layer: {e}")

def _dbg_hook(name):
    def f(_m, _in, out):
        y, _, _ = _split_hook_out(out)
        kind = "tensor" if torch.is_tensor(y) else type(y).__name__
        shape = tuple(y.shape) if torch.is_tensor(y) else None
        print(f"[DBG] {name}: y_kind={kind}, y_shape={shape}")
        return out
    return f

def collect_donor_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    arch: str,
) -> DonorBank:
    bank = DonorBank()
    blocks = get_blocks(model, arch, which="encoder" if arch == "encdec" else None)
    if DEBUG_T5 and arch == "encdec":
        for li, block in enumerate(blocks[:DEBUG_LAYERS_TO_PEEK]):
          _dbg_peek_block(block, li)

    hooks = []

    def make_attn_hook(li: int, attn_mod: nn.Module):
        nh = num_heads(attn_mod)
        def hook(_m, _in, out):
            y, _, _ = _split_hook_out(out)
            if y is None or not torch.is_tensor(y):
                if DEBUG_T5:
                    print(f"[DONOR] L{li} attn: y is {type(y).__name__}, skipping capture")
                return
            bank.put("attn", li, y)
            if nh is not None and y.dim() == 3:
                B, T, H = y.shape
                if H % nh == 0:
                    d = H // nh
                    y_heads = y.view(B, T, nh, d).contiguous()
                    for h in range(nh):
                        bank.put("attn", li, y_heads[:, :, h, :], head=h)
        return hook

    def make_mlp_hook(li: int):
        def hook(_m, _in, out):
            y, _, _ = _split_hook_out(out)
            if y is None or not torch.is_tensor(y):
                if DEBUG_T5:
                    print(f"[DONOR] L{li} mlp: y is {type(y).__name__}, skipping capture")
                return
            bank.put("mlp", li, y)
        return hook

    debug_hooks = []
    for li, block in enumerate(blocks):
        parts = block_parts(block, arch)

        if parts["attn"] is not None:
            hooks.append(parts["attn"].register_forward_hook(make_attn_hook(li, parts["attn"])))
            if DEBUG_T5 and li < DEBUG_LAYERS_TO_PEEK:
                debug_hooks.append(parts["attn"].register_forward_hook(_dbg_hook(f"donor.enc.attn L{li}")))

        if parts["mlp"] is not None:
            hooks.append(parts["mlp"].register_forward_hook(make_mlp_hook(li)))
            if DEBUG_T5 and li < DEBUG_LAYERS_TO_PEEK:
                debug_hooks.append(parts["mlp"].register_forward_hook(_dbg_hook(f"donor.enc.mlp  L{li}")))


    # Forward once on COMPLEX (donor)
    ins = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(get_device())
    with torch.no_grad():
        if arch == "encdec":
            enc = model.get_encoder() if hasattr(model, "get_encoder") else model.encoder
            _ = enc(**ins, output_hidden_states=True, return_dict=True)
        else:
            _ = model(**ins, output_hidden_states=True, return_dict=True)

    for h in hooks: h.remove()
    for h in debug_hooks:
      h.remove()
    return bank

def _map_by_alignment(donor_act: torch.Tensor, alignment: List[int], recipient_len: int) -> torch.Tensor:
    mapped = donor_act[:, alignment, ...]
    if mapped.size(1) == recipient_len:
        return mapped
    out = donor_act.new_zeros((1, recipient_len) + donor_act.shape[2:])
    keep = min(mapped.size(1), recipient_len)
    out[:, :keep] = mapped[:, :keep]
    return out

def apply_patches_and_run(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    arch: str,
    simple_text: str,                  # recipient (simple)
    donor_bank: DonorBank,             # captured on complex
    patch_spec: PatchSpec,
    donor_text: Optional[str] = None,  # needed for alignment
    target_text: Optional[str] = None, # for perplexity in encdec/decoder
    generate: bool = False,
    max_new_tokens: int = 64,
    num_beams: int = 4,
) -> Dict[str, Any]:
    device = get_device()

    align = None
    if donor_text is not None:
        align = align_tokens_by_input_embeddings(model, tokenizer, donor_text, simple_text)

    blocks = get_blocks(model, arch, which="encoder" if arch == "encdec" else None)
    if DEBUG_T5 and arch == "encdec":
        for li, block in enumerate(blocks[:DEBUG_LAYERS_TO_PEEK]):
            _dbg_peek_block(block, li)

    want = set((li, k, None if h is None else int(h)) for (li, k, h) in patch_spec.entries)
    hooks = []


    def _split_hook_out(out):
        if out is None:
            return None, (), False
        if isinstance(out, (list, tuple)):
            if len(out) == 0:
                return None, tuple(), True
            return out[0], tuple(out[1:]), True
        return out, tuple(), False

    def _rebuild_hook_out(y, rest, was_tuple):
        return (y,) + rest if was_tuple else y

    def _dbg_hook(name):
        def f(_m, _in, out):
            y, _, _ = _split_hook_out(out)
            kind = "tensor" if torch.is_tensor(y) else type(y).__name__
            shape = tuple(y.shape) if torch.is_tensor(y) else None
            print(f"[DBG] {name}: y_kind={kind}, y_shape={shape}")
            return out
        return f


    def make_attn_patch(li: int, attn_mod: nn.Module):
        nh = num_heads(attn_mod)  # may be None for T5 wrapper; then per-head patching is skipped

        def hook(_m, _in, out):
            try:
                y, rest, was_tuple = _split_hook_out(out)
                if y is None or not torch.is_tensor(y):
                    return out  # don't touch non-tensor outputs

                B, T, H = y.shape

                # whole-attention replacement
                if (li, "attn", None) in want:
                    donor = donor_bank.get("attn", li, None)
                    if donor is not None:
                        donor = donor.to(y.device)
                        if align is not None:
                            donor = _map_by_alignment(donor, align, T)
                        if donor.shape == y.shape:
                            y = donor

                # per-head replacement (only if we know head count and shapes match)
                if nh is not None and H % nh == 0:
                    d = H // nh
                    y_heads = y.view(B, T, nh, d).clone()
                    changed = False
                    for h in range(nh):
                        if (li, "attn", h) in want:
                            donor_h = donor_bank.get("attn", li, h)
                            if donor_h is not None:
                                donor_h = donor_h.to(y.device)
                                if align is not None:
                                    donor_h = _map_by_alignment(donor_h, align, T)
                                if donor_h.shape == y_heads[:, :, h, :].shape:
                                    y_heads[:, :, h, :] = donor_h
                                    changed = True
                    if changed:
                        y = y_heads.view(B, T, H)

                return _rebuild_hook_out(y, rest, was_tuple)
            except Exception:
                return out  # fail-open on any surprise
        return hook



    def make_mlp_patch(li: int):
        def hook(_m, _in, out):
            try:
                y, rest, was_tuple = _split_hook_out(out)
                if y is None or not torch.is_tensor(y):
                    return out

                if (li, "mlp", None) in want:
                    donor = donor_bank.get("mlp", li, None)
                    if donor is not None:
                        donor = donor.to(y.device)
                        if align is not None:
                            donor = _map_by_alignment(donor, align, y.size(1))
                        if donor.shape == y.shape:
                            y = donor

                return _rebuild_hook_out(y, rest, was_tuple)
            except Exception:
                return out  # fail-open
        return hook


    # DEBUG: register patch hooks + attach printers for first few layers
    hooks = []
    debug_hooks = []  # <— NEW

    for li, block in enumerate(blocks):
        parts = block_parts(block, arch)

        # log what block_parts picked for this block
        if DEBUG_T5 and arch == "encdec" and li < DEBUG_LAYERS_TO_PEEK:
            attn_t = type(parts["attn"]).__name__ if parts["attn"] is not None else None
            mlp_t  = type(parts["mlp"] ).__name__ if parts["mlp"]  is not None else None
            print(f"[DBG] enc L{li}: block_parts -> attn={attn_t}, mlp={mlp_t}")
            if parts["attn"] is None:
                print(f"[DBG] enc L{li}: parts['attn'] is None")
            if parts["mlp"] is None:
                print(f"[DBG] enc L{li}: parts['mlp'] is None")

        if parts["attn"] is not None:
            hooks.append(parts["attn"].register_forward_hook(make_attn_patch(li, parts["attn"])))
            if DEBUG_T5 and arch == "encdec" and li < DEBUG_LAYERS_TO_PEEK:
                debug_hooks.append(parts["attn"].register_forward_hook(_dbg_hook(f"patch.enc.attn L{li}")))

        if parts["mlp"] is not None:
            hooks.append(parts["mlp"].register_forward_hook(make_mlp_patch(li)))
            if DEBUG_T5 and arch == "encdec" and li < DEBUG_LAYERS_TO_PEEK:
                debug_hooks.append(parts["mlp"].register_forward_hook(_dbg_hook(f"patch.enc.mlp  L{li}")))










    out: Dict[str, Any] = {}
    ins = tokenizer(simple_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

    if arch == "encdec":
      if generate:
          with torch.no_grad():
              gen_ids = model.generate(**ins, max_new_tokens=max_new_tokens, num_beams=num_beams)
          out["generated"] = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]

      if target_text is not None:
          tgt = tokenizer(target_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
          labels = tgt["input_ids"]
          # HuggingFace handles shifting automatically when you pass labels
          with torch.no_grad():
              loss = model(**ins, labels=labels).loss
          out["loss"] = float(loss.item())
          out["perplexity"] = float(math.exp(min(20.0, loss.item())))

      with torch.no_grad():
          enc = model.get_encoder() if hasattr(model, "get_encoder") else model.encoder
          enc_out = enc(**ins, output_hidden_states=True, return_dict=True)
      out["pooled"] = masked_mean(enc_out.hidden_states[-1], ins["attention_mask"]).squeeze(0).detach().cpu()


    elif arch == "decoder":
        if generate:
            with torch.no_grad():
                gen_ids = model.generate(**ins, max_new_tokens=max_new_tokens, num_beams=num_beams)
            out["generated"] = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]

        if target_text is not None:
            tgt = tokenizer(target_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
            with torch.no_grad():
                loss = model(input_ids=tgt["input_ids"], labels=tgt["input_ids"]).loss
            out["loss"] = float(loss.item())
            out["perplexity"] = float(math.exp(min(20.0, loss.item())))

        with torch.no_grad():
            hs = model(**ins, output_hidden_states=True, return_dict=True).hidden_states
        out["pooled"] = hs[-1][:, -1, :].squeeze(0).detach().cpu()

    else:
        with torch.no_grad():
            enc_out = model(**ins, output_hidden_states=True, return_dict=True)
        out["pooled"] = masked_mean(enc_out.hidden_states[-1], ins["attention_mask"]).squeeze(0).detach().cpu()

    for h in hooks: h.remove()
    for h in debug_hooks: h.remove()   # <— NEW
    return out


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a, b, dim=0).item()

def encoder_target_rep(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, complex_text: str) -> torch.Tensor:
    device = get_device()
    ins = tokenizer(complex_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        out = model(**ins, output_hidden_states=True, return_dict=True)
    return masked_mean(out.hidden_states[-1], ins["attention_mask"]).squeeze(0).detach().cpu()

def baseline_perplexity(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, arch: str, simple_text: str, complex_text: str
) -> float:
    device = get_device()
    if arch == "encdec":
        ins = tokenizer(simple_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        tgt = tokenizer(complex_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        labels = tgt["input_ids"]
        decoder_start = (
            model.config.decoder_start_token_id
            if getattr(model.config, "decoder_start_token_id", None) is not None
            else tokenizer.eos_token_id
        )
        decoder_input_ids = labels.new_zeros(labels.shape)
        decoder_input_ids[:, 1:] = labels[:, :-1]
        decoder_input_ids[:, 0] = decoder_start
        decoder_input_ids.masked_fill_(decoder_input_ids == -100, tokenizer.pad_token_id or 0)
        with torch.no_grad():
            loss = model(**ins, labels=labels, decoder_input_ids=decoder_input_ids).loss
        return float(math.exp(min(20.0, loss.item())))
    else:
        tgt = tokenizer(complex_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            loss = model(input_ids=tgt["input_ids"], labels=tgt["input_ids"]).loss
        return float(math.exp(min(20.0, loss.item())))

# -------- Sweep utilities --------
def get_necessity_patch_spec(model, arch, exclude_li, exclude_kind, exclude_head=None) -> PatchSpec:
    entries = []
    blocks = get_blocks(model, arch, which="encoder" if arch == "encdec" else None)
    for li, block in enumerate(blocks):
        parts = block_parts(block, arch)
        if parts["attn"] is not None:
            nh = num_heads(parts["attn"])
            if nh:
                for h in range(nh):
                    if not (li == exclude_li and exclude_kind == "attn" and h == exclude_head):
                        entries.append((li, "attn", h))
            if not (li == exclude_li and exclude_kind == "attn" and exclude_head is None):
                entries.append((li, "attn", None))
        if parts["mlp"] is not None and not (li == exclude_li and exclude_kind == "mlp"):
            entries.append((li, "mlp", None))
    return PatchSpec(entries)

def random_patch_spec(model, arch, num_patches=5) -> PatchSpec:
    entries = []
    blocks = get_blocks(model, arch, which="encoder" if arch == "encdec" else None)
    candidates = []
    for li, block in enumerate(blocks):
        parts = block_parts(block, arch)
        if parts["attn"] is not None:
            nh = num_heads(parts["attn"])
            if nh:
                for h in range(nh):
                    candidates.append((li, "attn", h))
            candidates.append((li, "attn", None))
        if parts["mlp"] is not None:
            candidates.append((li, "mlp", None))
    if len(candidates) == 0:
        return PatchSpec([])
    return PatchSpec(random.sample(candidates, min(num_patches, len(candidates))))

def run_component_sweep(
    model_name: str,
    complex_text: str,   # donor (complex)
    simple_text: str,    # recipient (simple)
    generate_text: bool = False,
    max_new_tokens: int = 64,
    num_beams: int = 4
) -> pd.DataFrame:

    model, tok, arch = load_model_and_tokenizer(model_name)

    # Donor bank capture (encoder only stack for enc-dec)
    donor = collect_donor_activations(model, tok, complex_text, arch)

    # Baselines
    if arch == "encoder":
        target_rep = encoder_target_rep(model, tok, complex_text)
        with torch.no_grad():
            ins = tok(simple_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(next(model.parameters()).device)
            enc_out = model(**ins, output_hidden_states=True, return_dict=True)
        pooled_before = masked_mean(enc_out.hidden_states[-1], ins["attention_mask"]).squeeze(0).detach().cpu()
        ppl_before = None
    else:
        ppl_before = baseline_perplexity(model, tok, arch, simple_text, complex_text)
        pooled_before = None
        target_rep = None

    records = []
    blocks = get_blocks(model, arch, which="encoder" if arch == "encdec" else None)

    for li, block in enumerate(blocks):
        parts = block_parts(block, arch)

        # ---- heads ----
        if parts["attn"] is not None:
            nh = num_heads(parts["attn"])
            if nh:
                for h in range(nh):
                    for mode, spec in [
                        ("sufficiency", PatchSpec([(li, "attn", h)])),
                        ("necessity", get_necessity_patch_spec(model, arch, li, "attn", h)),
                        ("random", random_patch_spec(model, arch, num_patches=5)),
                    ]:
                        after = apply_patches_and_run(
                            model, tok, arch,
                            simple_text=simple_text,
                            donor_bank=donor,
                            patch_spec=spec,
                            donor_text=complex_text,
                            target_text=complex_text if arch in ["decoder","encdec"] else None,
                            generate=generate_text,
                            max_new_tokens=max_new_tokens,
                            num_beams=num_beams
                        )
                        rec = {"model": model_name, "layer": li, "component": "attn_head", "index": h, "mode": mode}
                        if arch == "encoder":
                            rep_after = after["pooled"]
                            rec.update({
                                "to_target_cosine": cosine_sim(rep_after, target_rep),
                                "delta_to_target": cosine_sim(rep_after, target_rep) - cosine_sim(pooled_before, target_rep)
                            })
                        else:
                            ppl_after = after.get("perplexity", np.nan)
                            rec.update({
                                "baseline_ppl": ppl_before,
                                "patched_ppl": ppl_after,
                                "delta_ppl": (ppl_before - ppl_after) if (ppl_after == ppl_after) else np.nan,
                            })
                        records.append(rec)

        # ---- whole attention ----
        if parts["attn"] is not None:
            for mode, spec in [
                ("sufficiency", PatchSpec([(li, "attn", None)])),
                ("necessity", get_necessity_patch_spec(model, arch, li, "attn", None)),
                ("random", random_patch_spec(model, arch, num_patches=5)),
            ]:
                after = apply_patches_and_run(
                    model, tok, arch,
                    simple_text=simple_text,
                    donor_bank=donor,
                    patch_spec=spec,
                    donor_text=complex_text,
                    target_text=complex_text if arch in ["decoder","encdec"] else None,
                    generate=generate_text,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams
                )
                rec = {"model": model_name, "layer": li, "component": "attn_block", "index": None, "mode": mode}
                if arch == "encoder":
                    rep_after = after["pooled"]
                    rec.update({
                        "to_target_cosine": cosine_sim(rep_after, target_rep),
                        "delta_to_target": cosine_sim(rep_after, target_rep) - cosine_sim(pooled_before, target_rep)
                    })
                else:
                    ppl_after = after.get("perplexity", np.nan)
                    rec.update({
                        "baseline_ppl": ppl_before,
                        "patched_ppl": ppl_after,
                        "delta_ppl": (ppl_before - ppl_after) if (ppl_after == ppl_after) else np.nan,
                    })
                records.append(rec)

        # ---- MLP ----
        if parts["mlp"] is not None:
            for mode, spec in [
                ("sufficiency", PatchSpec([(li, "mlp", None)])),
                ("necessity", get_necessity_patch_spec(model, arch, li, "mlp")),
                ("random", random_patch_spec(model, arch, num_patches=5)),
            ]:
                after = apply_patches_and_run(
                    model, tok, arch,
                    simple_text=simple_text,
                    donor_bank=donor,
                    patch_spec=spec,
                    donor_text=complex_text,
                    target_text=complex_text if arch in ["decoder","encdec"] else None,
                    generate=generate_text,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams
                )
                rec = {"model": model_name, "layer": li, "component": "mlp", "index": None, "mode": mode}
                if arch == "encoder":
                    rep_after = after["pooled"]
                    rec.update({
                        "to_target_cosine": cosine_sim(rep_after, target_rep),
                        "delta_to_target": cosine_sim(rep_after, target_rep) - cosine_sim(pooled_before, target_rep)
                    })
                else:
                    ppl_after = after.get("perplexity", np.nan)
                    rec.update({
                        "baseline_ppl": ppl_before,
                        "patched_ppl": ppl_after,
                        "delta_ppl": (ppl_before - ppl_after) if (ppl_after == ppl_after) else np.nan,
                    })
                records.append(rec)




    df = pd.DataFrame.from_records(records)

    if df.empty:
        return df

    if arch == "encoder":
        # cosine similarity for encoders
        df["impact_score"] = df.get("to_target_cosine", np.nan)

    elif arch == "encdec":
        # T5 / BART style: patched perplexity minus baseline
        if {"baseline_ppl", "patched_ppl"}.issubset(df.columns):
            df["impact_score"] = df["patched_ppl"] - df["baseline_ppl"]
        elif "delta_ppl" in df.columns:
            # if delta_ppl = baseline - patched, invert it
            df["impact_score"] = -df["delta_ppl"]
        else:
            warnings.warn("No perplexity columns available for encdec; setting impact_score=NaN")
            df["impact_score"] = np.nan

    else:
        # decoder-only fallback
        if {"baseline_ppl", "patched_ppl"}.issubset(df.columns):
            df["impact_score"] = df["patched_ppl"] - df["baseline_ppl"]
        else:
            df["impact_score"] = np.nan

    return df


# -------- Plots --------
def plot_causal_heatmaps(df: pd.DataFrame, model_name: str, metric: str = "impact_score", save_path: Optional[str]=None):
    modes = [m for m in ["sufficiency", "necessity", "random"] if m in df["mode"].unique()]
    if not modes: modes = ["sufficiency"]
    n_modes = len(modes)
    fig, axes = plt.subplots(2, n_modes, figsize=(5.2 * n_modes, 9.0), squeeze=False)

    for j, mode in enumerate(modes):
        sub = df[df["mode"] == mode]
        # Attention heads heatmap
        attn = sub[sub["component"] == "attn_head"]
        axes[0, j].set_title(f"{model_name}\n{mode} - Attention ({metric})")
        if not attn.empty:
            n_layers = int(attn["layer"].max()) + 1
            n_heads = int(attn["index"].max()) + 1
            mat = np.full((n_layers, n_heads), np.nan, dtype=np.float32)
            for _, row in attn.iterrows():
                li, hi = int(row["layer"]), int(row["index"])
                val = row.get(metric, np.nan)
                mat[li, hi] = np.nan if pd.isna(val) else float(val)
            im = axes[0, j].imshow(mat, aspect="equal", interpolation="nearest")
            axes[0, j].set_xlabel("Head")
            axes[0, j].set_ylabel("Layer")
            axes[0, j].set_xlim(-0.5, n_heads - 0.5)
            axes[0, j].set_ylim(n_layers - 0.5, -0.5)
            cb = fig.colorbar(im, ax=axes[0, j], fraction=0.046, pad=0.04)
            cb.ax.set_ylabel(metric)
        else:
            axes[0, j].text(0.5, 0.5, "No attention-head records", ha="center", va="center")

        # MLP bar chart
        mlp = sub[sub["component"] == "mlp"].sort_values("layer")
        axes[1, j].set_title(f"{mode} - MLP ({metric})")
        axes[1, j].set_xlabel("Layer")
        axes[1, j].set_ylabel(metric)
        if not mlp.empty:
            axes[1, j].bar(mlp["layer"].astype(int), mlp[metric].astype(float), width=0.6)
            axes[1, j].set_xlim(mlp["layer"].min() - 0.5, mlp["layer"].max() + 0.5)
            y_vals = mlp[metric].dropna().astype(float).values
        if y_vals.size:
            ymax = np.nanmax(y_vals)
            pad = 0.1 * (ymax if ymax > 0 else 1e-3)
            axes[1, j].set_ylim(0, ymax + pad)
        else:
            axes[1, j].text(0.5, 0.5, "No MLP records", ha="center", va="center")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.show()

def plot_delta_ppl(df: pd.DataFrame, model_name: str, save_path: Optional[str]=None):
    sub = df[df["delta_ppl"].notna()]
    if sub.empty:
        return
    agg = sub.groupby(["component", "layer"], as_index=False)["delta_ppl"].mean()
    plt.figure(figsize=(10, 5))
    plt.title(f"{model_name} - Mean ΔPPL (baseline - patched) by Layer/Component")
    for comp in ["attn_head", "attn_block", "mlp"]:
        comp_df = agg[agg["component"] == comp]
        if not comp_df.empty:
            plt.plot(comp_df["layer"], comp_df["delta_ppl"], marker="o", label=comp)
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Layer")
    plt.ylabel("ΔPPL (higher is better)")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.show()

# -------- Run the whole experiment --------
def run_full_experiment(
    model_list: List[str],
    technical_sentences: List[str],   # complex
    informal_sentences: List[str],    # simple
    generate_text: bool,
    max_pairs: Optional[int],
    max_new_tokens: int,
    num_beams: int
):
    os.makedirs(OUT_DIR, exist_ok=True)
    N = len(technical_sentences) if max_pairs is None else min(len(technical_sentences), max_pairs)
    all_results = []
    for model_name in model_list:
        print(f"\n[INFO] Model: {model_name}")
        for i in range(N):
            complex_text = technical_sentences[i]
            simple_text  = informal_sentences[i]
            try:
                df_i = run_component_sweep(
                    model_name=model_name,
                    complex_text=complex_text,
                    simple_text=simple_text,
                    generate_text=generate_text,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams
                )
                df_i["pair_idx"] = i
                all_results.append(df_i)
            except Exception as e:
                print(f"[WARN] pair {i} on {model_name} failed: {e}")

        if not all_results:  # if every pair failed for first model
            continue

        results_df = pd.concat(all_results, ignore_index=True)
        # Save CSVs
        csv_all = os.path.join(OUT_DIR, f"{sanitize(model_name)}__testcsv_component_causal_results.csv")
        results_df.to_csv(csv_all, index=False)
        print(f"[SAVE] {csv_all}")

        # Top-k by impact
        by_model = results_df[results_df["model"] == model_name]
        topk = by_model.sort_values("impact_score", ascending=False).groupby(["component"]).head(20)
        csv_top = os.path.join(OUT_DIR, f"{sanitize(model_name)}__testcsv__topk.csv")
        topk.to_csv(csv_top, index=False)
        print(f"[SAVE] {csv_top}")

        # Plots
        try:
            plot_causal_heatmaps(by_model, model_name,
                                 save_path=os.path.join(OUT_DIR, f"{sanitize(model_name)}__testcsv__heatmaps.png"))
            if by_model["delta_ppl"].notna().any():
                plot_delta_ppl(by_model, model_name,
                               save_path=os.path.join(OUT_DIR, f"{sanitize(model_name)}__testcsv__delta_ppl.png"))
        except Exception as e:
            print(f"[WARN] plotting failed for {model_name}: {e}")

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()

def sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


# --------------------- main ---------------------
if __name__ == "__main__":
    # Load data
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"{DATA_CSV} not found. Please upload your CSV with columns 'input_text' and 'target_text'.")
    df = pd.read_csv(DATA_CSV)
    if "input_text" not in df.columns or "target_text" not in df.columns:
        raise ValueError("CSV must contain columns: 'input_text' (complex), 'target_text' (simple)")
    technical_sentences = df["input_text"].astype(str).tolist()
    informal_sentences  = df["target_text"].astype(str).tolist()

    # Sanity: limit pairs
    if MAX_PAIRS is not None:
        n = min(len(technical_sentences), MAX_PAIRS)
        technical_sentences = technical_sentences[:n]
        informal_sentences  = informal_sentences[:n]

    if not MODEL_LIST:
        print("[NOTE] MODEL_LIST is empty. Edit MODEL_LIST at the top to run your desired checkpoints.")
    else:
        results = run_full_experiment(
            model_list=MODEL_LIST,
            technical_sentences=technical_sentences,
            informal_sentences=informal_sentences,
            generate_text=GENERATE_TEXT,
            max_pairs=MAX_PAIRS,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS
        )
        if not results.empty:
            print("[DONE] Results shape:", results.shape)
            print(results.head(8))
        else:
            print("[DONE] No results (all pairs may have failed).")
