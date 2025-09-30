import os
import argparse
import math
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn.functional import kl_div, log_softmax, softmax

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoConfig,
)

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import seaborn as sns

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_to_device(inputs: dict, device: torch.device) -> dict:
    """Only move tensor-like objects to device; keep non-tensor items (e.g., metadata) as-is."""
    new = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            new[k] = v.to(device)
        else:
            new[k] = v
    return new


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).float()
    masked = tensor * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


def load_model_and_tokenizer(model_name: str, model_type: Optional[str] = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = get_device()

    if model_type is None:
        lname = model_name.lower()
        if "t5" in lname or "bart" in lname or "m2m" in lname:
            model_type = "seq2seq"
        elif "gpt" in lname or "gpt2" in lname or "gpt-j" in lname:
            model_type = "causal"
        else:
            model_type = "base"

    model_cls = AutoModel
    if model_type == "seq2seq":
        model_cls = AutoModelForSeq2SeqLM
    elif model_type == "causal":
        model_cls = AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
    model = model_cls.from_pretrained(model_name, config=config)
    model = model.eval().to(device)
    model = model.to(device)
    return model, tokenizer


def run_inference(model, tokenizer, sentence_or_batch, model_name: str):
    """Accepts a single string or a list of strings. Returns (hidden_states, attentions, attention_mask)
    hidden_states is a list of layer tensors (batch, seq_len, hidden)
    attentions is a tuple/list per layer when available.
    """
    single = isinstance(sentence_or_batch, str)
    if single:
        sentences = [sentence_or_batch]
    else:
        sentences = sentence_or_batch

    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = move_to_device(inputs, model.device)

    with torch.no_grad():
        lname = model_name.lower()
        try:
            if "t5" in lname and hasattr(model, "encoder"):
                encoder_outputs = model.encoder(**inputs)
                hidden_states = list(encoder_outputs.hidden_states) if getattr(encoder_outputs, "hidden_states", None) is not None else None
                attentions = getattr(encoder_outputs, "attentions", None)
            else:
                outputs = model(**inputs)
                hidden_states = getattr(outputs, "hidden_states", None)
                attentions = getattr(outputs, "attentions", None)

            # normalize hidden_states to a list
            if hidden_states is None:
                hidden_states = None
            else:
                hidden_states = [hs for hs in hidden_states]

            attn = attentions if attentions is not None else None
            return hidden_states, attn, inputs.get("attention_mask")

        except Exception as e:
            print(f"Inference failure for model {model_name}: {e}")
            return None, None, None

def align_reps_for_cka(X: torch.Tensor, Y: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    n1 = X.shape[0]
    n2 = Y.shape[0]
    if n1 == 0 or n2 == 0:
        return None, None
    if n1 == n2:
        return X, Y
    min_n = min(n1, n2)
    idx1 = [int(round(x)) for x in np.linspace(0, n1 - 1, min_n)]
    idx2 = [int(round(x)) for x in np.linspace(0, n2 - 1, min_n)]
    return X[idx1], Y[idx2]


def linear_CKA(X, Y, eps=1e-10):
    X = X.double()
    Y = Y.double()

    # centers the features -> required for CKA calculation
    X -= X.mean(dim=0, keepdim=True)
    Y -= Y.mean(dim=0, keepdim=True)

    # cross-covariance
    cross_term = X.T @ Y

    # HSIC terms
    hsic = torch.norm(cross_term, p='fro') ** 2
    hsic_xx = torch.norm(X.T @ X, p='fro') ** 2
    hsic_yy = torch.norm(Y.T @ Y, p='fro') ** 2

    #normalized term, adding eps to avoid division by 0 (common procedure)
    return float(hsic / (torch.sqrt(hsic_xx * hsic_yy) + eps))

def compute_layerwise_similarity(h1: List[torch.Tensor], h2: List[torch.Tensor], mask1: torch.Tensor, mask2: torch.Tensor):
    cos_sims, l2_dists, cka_scores = [], [], []
    if h1 is None or h2 is None:
        return [float('nan')] * 1, [float('nan')] * 1, [float('nan')] * 1

    for layer_idx in range(len(h1)):
        h1_layer = h1[layer_idx].squeeze(0)
        h2_layer = h2[layer_idx].squeeze(0)
        h1_masked = h1_layer[mask1.squeeze(0).bool()]
        h2_masked = h2_layer[mask2.squeeze(0).bool()]

        if h1_masked.size(0) == 0 or h2_masked.size(0) == 0:
            cos_sims.append(float('nan'))
            l2_dists.append(float('nan'))
            cka_scores.append(float('nan'))
            continue

        mean1 = h1_masked.mean(dim=0)
        mean2 = h2_masked.mean(dim=0)
        cos = F.cosine_similarity(mean1, mean2, dim=0).item()
        l2 = torch.norm(mean1 - mean2, p=2).item()
        rep1_al, rep2_al = align_reps_for_cka(h1_masked, h2_masked)
        cka = linear_CKA(rep1_al, rep2_al) if rep1_al is not None else float('nan')

        cos_sims.append(cos)
        l2_dists.append(l2)
        cka_scores.append(cka)

    return cos_sims, l2_dists, cka_scores


def compute_cohens_d(group1: torch.Tensor, group2: torch.Tensor) -> float:
    mean1, mean2 = group1.mean(), group2.mean()
    std1, std2 = group1.std(unbiased=True), group2.std(unbiased=True)
    pooled_std = torch.sqrt(((std1 ** 2 + std2 ** 2) / 2) + 1e-8)
    return ((mean1 - mean2) / pooled_std).item()


def compute_neuronwise_cohens_d(hidden_states_tech: List[torch.Tensor], hidden_states_plain: List[torch.Tensor], mask_tech: torch.Tensor, mask_plain: torch.Tensor):
    rankings = []
    for layer_idx in range(len(hidden_states_tech)):
        h1 = hidden_states_tech[layer_idx].squeeze(0)[mask_tech.squeeze(0).bool()]
        h2 = hidden_states_plain[layer_idx].squeeze(0)[mask_plain.squeeze(0).bool()]
        for neuron_idx in range(h1.shape[1]):
            group1 = h1[:, neuron_idx]
            group2 = h2[:, neuron_idx]
            d = compute_cohens_d(group1, group2)
            rankings.append((layer_idx, neuron_idx, d))
    rankings.sort(key=lambda x: abs(x[2]), reverse=True)
    return rankings


def compute_attention_kl(attn1, attn2):
    if attn1 is None or attn2 is None:
        return None
    try:
        a1 = attn1[-1].mean(1).squeeze(0)
        a2 = attn2[-1].mean(1).squeeze(0)
        log_p = log_softmax(a1, dim=-1)
        q = softmax(a2, dim=-1)
        return kl_div(log_p, q, reduction='batchmean').item()
    except Exception:
        return None


def plot_similarity_curves(all_results: dict):
    plt.figure(figsize=(10, 6))
    for model_name, sims in all_results.items():
        plt.plot(sims, label=model_name)
    plt.title("Layerwise Cosine Similarity Between Technical and Informal Sentences")
    plt.xlabel("Layer")
    plt.ylabel("Cosine Similarity")
    plt.ylim(0.0, 1.05)
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def visualize_trajectory(hidden_states: List[torch.Tensor], title: str = "Trajectory PCA Visualization"):
    if hidden_states is None or len(hidden_states) == 0:
        print(f"{title}: no hidden states available")
        return
    try:
        layer_reps = torch.stack([torch.mean(h, dim=1).squeeze(0) for h in hidden_states]).cpu().numpy()
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(layer_reps)
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=list(range(len(hidden_states))), s=100)
        for i, (x, y) in enumerate(reduced):
            plt.text(x + 0.01, y + 0.01, f"L{i}", fontsize=8)
        plt.colorbar(scatter, label="Layer")
        plt.title(title)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"Failed to visualize trajectory: {e}")


def analyze_attention(attentions, title: str = "Attention Heatmap"):
    if attentions is None or len(attentions) == 0:
        print(f"{title}: No attention data available.")
        return
    try:
        attn = attentions[-1]
        attn_sum = attn.sum(dim=1).squeeze(0).cpu().numpy()
        plt.figure(figsize=(8, 6))
        sns.heatmap(attn_sum)
        plt.title(title)
        plt.xlabel("Token Position")
        plt.ylabel("Token Position")
        plt.show()
    except Exception as e:
        print(f"{title}: Failed to plot attention heatmap — {e}")


def main(args):
    df = pd.read_csv(args.data)
    tech_sents = df["input_text"].tolist()
    info_sents = df["target_text"].tolist()
    assert len(tech_sents) == len(info_sents), "input and target must have same length"

    models = args.models

    results_rows = []
    all_cos_curves = {}

    for model_name in models:
        print(f"\nLoading model: {model_name}")
        try:
            model, tokenizer = load_model_and_tokenizer(model_name)
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue

        N = len(tech_sents)
        sample_k = min(args.sample_k, max(1, N))
        sample_indices = np.unique(np.round(np.linspace(0, N - 1, num=sample_k)).astype(int)).tolist()
        for start in tqdm(range(0, N, args.batch_size), desc=model_name):
            end = min(start + args.batch_size, N)
            tech_batch = tech_sents[start:end]
            info_batch = info_sents[start:end]

            tech_hidden, _, tech_mask = run_inference(model, tokenizer, tech_batch, model_name)
            info_hidden, _, info_mask = run_inference(model, tokenizer, info_batch, model_name)

            if tech_hidden is None or info_hidden is None:
                print(f"Skipping batch {start}:{end} for {model_name} due to missing hidden states")
                continue

            for i in range(len(tech_batch)):
                global_idx = start + i
                h1 = [layer[i].unsqueeze(0) for layer in tech_hidden]
                h2 = [layer[i].unsqueeze(0) for layer in info_hidden]
                m1 = tech_mask[i].unsqueeze(0)
                m2 = info_mask[i].unsqueeze(0)

                if global_idx in sample_indices:
                    visualize_trajectory(h1, title=f"{model_name} - Trajectory (Technical) idx={global_idx}")
                    visualize_trajectory(h2, title=f"{model_name} - Trajectory (Informal) idx={global_idx}")

                cos_sims, l2_dists, cka_scores = compute_layerwise_similarity(h1, h2, m1, m2)
                avg_cos = float(np.nanmean(cos_sims))
                avg_l2 = float(np.nanmean(l2_dists))
                avg_cka = float(np.nanmean(cka_scores))

                _, tech_attn, _ = run_inference(model, tokenizer, tech_batch[i], model_name)
                _, info_attn, _ = run_inference(model, tokenizer, info_batch[i], model_name)
                attn_kl = compute_attention_kl(tech_attn, info_attn)

                ranked_neurons = compute_neuronwise_cohens_d(h1, h2, m1, m2)
                cohen_d_score = float(np.mean([abs(d) for _, _, d in ranked_neurons])) if ranked_neurons else float('nan')
                top_neurons = ranked_neurons[:10]

                results_rows.append({
                    "model_name": model_name,
                    "index": global_idx,
                    "input_text": tech_batch[i],
                    "target_text": info_batch[i],
                    "avg_cosine_similarity": avg_cos,
                    "avg_l2_distance": avg_l2,
                    "avg_cka": avg_cka,
                    "attention_kl_divergence": attn_kl,
                    "cohen_d_neuronwise": cohen_d_score,
                    "layerwise_cosine": str(cos_sims),
                    "layerwise_l2": str(l2_dists),
                    "layerwise_cka": str(cka_scores),
                    "top_neurons": str(top_neurons),
                })

        try:
            all_cos_curves[model_name] = [r["avg_cosine_similarity"] for r in results_rows if r["model_name"] == model_name]
        except Exception:
            pass

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(args.output, index=False)
    print(f"Saved results to {args.output}")

    if len(results_df) > 0:
        model_summary = results_df.groupby("model_name").agg({
            "avg_cosine_similarity": "mean",
            "avg_l2_distance": "mean",
            "avg_cka": "mean",
            "attention_kl_divergence": "mean",
            "cohen_d_neuronwise": "mean",
        }).reset_index()
        model_summary.to_csv(args.summary_output, index=False)
        print(f"Saved model summary to {args.summary_output}")

        plt.figure(figsize=(8, 4))
        sns.barplot(data=model_summary, x="model_name", y="avg_cka")
        plt.xticks(rotation=45)
        plt.title("Average CKA Score per Model")
        plt.tight_layout()
        plt.show()

    if len(all_cos_curves) > 0:
        plot_similarity_curves(all_cos_curves)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="val.csv", help="Path to CSV with input_text and target_text columns")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--sample-k", type=int, default=5, help="How many example indices to sample for trajectory viz per model")
    parser.add_argument("--output", type=str, default="similarity_results_val.csv")
    parser.add_argument("--summary-output", type=str, default="model_summary.csv")
    parser.add_argument("--models", nargs="*", default=[
        "dmis-lab/biobert-base-cased-v1.1",
        "hossboll/clinical-t5",
        "allenai/scibert_scivocab_uncased",
        "microsoft/BioGPT-Large-PubMedQA"
    ])
    args = args, unknown = parser.parse_known_args()
    main(args)
