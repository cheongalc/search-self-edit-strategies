"""This script computes the intra-iteration similarity scores used to analyze 
the diversity among self-edit templates that are generated during 1 iteration.

There are two types of intra-iteration similarity scores:
1. Text similiarity, which measures the amount of overlap in the data creation
   instructions among the templates.
2. Hyperparameter similarity, which predefines a fixed range of sensible values
   for each hyperparameter, then measures how close the templates' hyperparameters
   are, using those ranges as a basis for comparison.

In the paper, these scores are reported in Figures 4 and 5, and in Appendix A.9.
The text version of how these similarity scores are computed is described in
Appendix A.10.
"""

from __future__ import annotations

import argparse
import dotenv
import json
import math
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple


def compute_similarity_scores(
    items: List[Dict[str, Any]],
    text_field: str = "data_creation_instruction",
    hp_field: str = "hyperparameters",
    embedding_model: str = "text-embedding-3-large",  # or "text-embedding-3-small"
    hp_domains: Optional[Dict[str, Dict[str, Any]]] = None,
    hp_weights: Optional[Dict[str, float]] = None,
    missing_hp_similarity: float = 0.0,  # set to None to ignore missing keys
) -> Tuple[float, float]:
    """Compute average text and hyperparameter similarity for a list of objects.

    Returns:
        A tuple of (text_score, hp_score), each in $[0, 1]$.

    Requires:
        - `OPENAI_API_KEY` needs to be available in the environment
    """
    if len(items) < 2:
        raise ValueError("Need at least 2 JSON objects to compute similarity.")

    # TEXT: OpenAI embeddings + cosine mapped to [0,1].
    from openai import OpenAI
    import numpy as np

    client = OpenAI()

    texts = []
    for obj in items:
        t = str(obj.get(text_field, ""))
        if not t.strip():
            t = "[EMPTY]"
        texts.append(t)

    resp = client.embeddings.create(
        model=embedding_model,
        input=texts,
        encoding_format="float",
    )
    embs = np.array([d.embedding for d in resp.data], dtype=np.float32)

    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    embs = embs / norms

    def sim_text(i: int, j: int) -> float:
        cos = float(embs[i] @ embs[j])      # [-1, 1]
        s = (1.0 + cos) / 2.0               # [0, 1]
        return max(0.0, min(1.0, s))

    N = len(items)
    text_pair_sims = [sim_text(i, j) for i, j in combinations(range(N), 2)]
    text_score = sum(text_pair_sims) / len(text_pair_sims)

    # HYPERPARAMS: fixed-domain similarity.
    if hp_domains is None:
        hp_domains = {
            "lora_rank": {"type": "log2", "min": 4, "max": 64},
            "lora_alpha": {"type": "log2", "min": 1, "max": 256},
            "lora_dropout": {"type": "linear", "min": 0.0, "max": 1.0},
            "learning_rate": {"type": "log10", "min": 1e-5, "max": 5e-3},
            "num_epochs": {"type": "linear", "min": 1, "max": 20},
            "gradient_accumulation_steps": {"type": "linear", "min": 1, "max": 8},
        }

    if hp_weights is None:
        hp_weights = {k: 1.0 for k in hp_domains.keys()}
    else:
        for k in hp_domains.keys():
            hp_weights.setdefault(k, 1.0)

    def _safe_log(v: float, base: float) -> float:
        if v <= 0:
            return float("nan")
        return math.log(v, base)

    def sim_from_domain(k: str, a: Any, b: Any) -> Optional[float]:
        dom = hp_domains[k]
        t = dom["type"]

        if a is None or b is None:
            return missing_hp_similarity  # None => ignore

        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return 1.0 if a == b else 0.0

        lo = float(dom["min"])
        hi = float(dom["max"])
        if hi <= lo:
            return 1.0 if a == b else 0.0

        if t == "linear":
            dist = abs(float(a) - float(b))
            denom = (hi - lo)
            s = 1.0 - (dist / denom)
            return max(0.0, min(1.0, s))

        if t == "log10":
            la = _safe_log(float(a), 10.0)
            lb = _safe_log(float(b), 10.0)
            llo = _safe_log(lo, 10.0)
            lhi = _safe_log(hi, 10.0)
        elif t == "log2":
            la = _safe_log(float(a), 2.0)
            lb = _safe_log(float(b), 2.0)
            llo = _safe_log(lo, 2.0)
            lhi = _safe_log(hi, 2.0)
        else:
            raise ValueError(f"Unknown domain type for {k}: {t}")

        if any(math.isnan(x) for x in (la, lb, llo, lhi)):
            return 0.0

        dist = abs(la - lb)
        denom = abs(lhi - llo)
        if denom == 0:
            return 1.0 if a == b else 0.0

        s = 1.0 - (dist / denom)
        return max(0.0, min(1.0, s))

    def sim_hparams(hp_a: Dict[str, Any], hp_b: Dict[str, Any]) -> float:
        num = 0.0
        den = 0.0
        for k, w in hp_weights.items():
            if k not in hp_domains:
                continue
            s = sim_from_domain(k, hp_a.get(k), hp_b.get(k))
            if s is None:
                continue
            num += float(w) * s
            den += float(w)
        return 0.0 if den == 0.0 else (num / den)

    hps = [obj.get(hp_field, {}) or {} for obj in items]
    hp_pair_sims = [sim_hparams(hps[i], hps[j]) for i, j in combinations(range(N), 2)]
    hp_score = sum(hp_pair_sims) / len(hp_pair_sims)

    return text_score, hp_score


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute text and hyperparameter overlap scores for batched JSON input."
        )
    )
    parser.add_argument(
        "--path",
        help=(
            "Path to a JSON file whose top-level object is a list of lists. Each "
            "inner list contains JSON objects with 'data_creation_instruction' and "
            "'hyperparameters' keys."
        ),
    )
    parser.add_argument(
        "--env_path",
        help="Path to the .env file containing OPENAI_API_KEY.",
        default=".env",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    with open(args.path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dotenv.load_dotenv(args.env_path)

    if not isinstance(data, list):
        raise ValueError("Top-level JSON object must be a list of lists.")

    for idx, group in enumerate(data):
        if not isinstance(group, list):
            raise ValueError(f"Item {idx} is not a list of JSON objects.")
        scores = compute_similarity_scores(group)
        print(scores)


if __name__ == "__main__":
    main()