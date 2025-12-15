# src/training/incremental_trainer.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Set, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from .fetch_data import fetch_training_rows_since
from .dataset import TimeseriesDataset
from .param_nn_model import HierarchicalParamsNN, INPUT_DIM, OUTPUT_DIM
from .vocab import Vocab, resize_embedding
from src.parametric.depreciation_function import price_from_params

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_PATH = MODELS_DIR / "hier_params_nn.pt"
VOCAB_PATH = MODELS_DIR / "gamme_vocab.json"


def _load_model_and_vocab(device: str = "cpu") -> tuple[HierarchicalParamsNN, Vocab]:
    vocab = Vocab.load(VOCAB_PATH)
    # if empty, initialize with a tiny size (we'll grow before training)
    num_gammes = max(1, vocab.size)

    model = HierarchicalParamsNN(num_gammes=num_gammes)
    if MODEL_PATH.exists():
        state = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state, strict=False)

    return model, vocab


def _ensure_vocab_and_resize_model(
    model: HierarchicalParamsNN,
    vocab: Vocab,
    rows: List[Dict[str, Any]],
) -> tuple[HierarchicalParamsNN, Vocab]:
    # Collect gamme_ids from data
    gamme_ids: List[str] = []
    for r in rows:
        try:
            gid = r["vehicles"]["gammes"].get("gamme_id")
        except Exception:
            gid = None
        if gid:
            gamme_ids.append(gid)

    added = vocab.add_many(gamme_ids)
    if added > 0:
        # Resize embedding table
        model.gamme_base = resize_embedding(model.gamme_base, vocab.size)

    return model, vocab


def train_incremental_on_new_rows(
    since_date: str,
    epochs: int = 3,
    batch_size: int = 64,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-3,
    supabase_weight: float = 1.0,
    marketcheck_weight: float = 0.25,
    device: str = "cpu",
) -> Dict[str, Any]:
    rows = fetch_training_rows_since(since_date, use_marketcheck=True)
    if not rows:
        return {"status": "no_new_data", "avg_loss": None, "gamme_ids": []}

    model, vocab = _load_model_and_vocab(device=device)
    model, vocab = _ensure_vocab_and_resize_model(model, vocab, rows)

    # Save vocab immediately so indices stay stable across runs
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    vocab.save(VOCAB_PATH)

    dataset = TimeseriesDataset(
        rows,
        supabase_weight=supabase_weight,
        marketcheck_weight=marketcheck_weight,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.to(device)
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Huber is usually better than MSE when your targets are noisy/outlier-prone
    loss_fn = torch.nn.SmoothL1Loss(reduction="none")

    total_loss = 0.0
    total_weight = 0.0
    gamme_ids_seen: Set[str] = set()

    for epoch in range(epochs):
        for batch in loader:
            x = batch["nn_input"].to(device)                  # [B, INPUT_DIM]
            V0 = batch["V0"].to(device)                       # [B]
            age_years = batch["age_years"].to(device)         # [B]
            mileage = batch["mileage"].to(device)             # [B]
            y = batch["target_value"].to(device).unsqueeze(1) # [B, 1]
            w = batch["sample_weight"].to(device).unsqueeze(1)# [B, 1]

            # Convert gamme_id strings -> indices
            gids = batch["gamme_id"]
            gamme_idx = torch.tensor([vocab.get(g) for g in gids], dtype=torch.long, device=device)
            # (safety) unknown ids shouldn't happen now, but if they do, clamp
            gamme_idx = torch.clamp(gamme_idx, min=0)

            params = model(x, gamme_idx)  # [B, 6]

            y_hat = price_from_params(
                V0=V0,
                age_years=age_years,
                mileage=mileage,
                params=params,
            ).unsqueeze(1)

            per_item = loss_fn(y_hat, y)    # [B, 1]
            loss = (per_item * w).sum() / (w.sum().clamp_min(1e-6))

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += float((per_item * w).sum().item())
            total_weight += float(w.sum().item())

            for g in gids:
                if g is not None:
                    gamme_ids_seen.add(g)

    avg_loss = total_loss / max(1e-6, total_weight)

    torch.save(model.state_dict(), MODEL_PATH)

    return {
        "status": "ok",
        "avg_loss": avg_loss,
        "gamme_ids": list(gamme_ids_seen),
        "num_rows": len(rows),
        "num_gammes": vocab.size,
    }
