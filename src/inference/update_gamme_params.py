# src/inference/update_gamme_params.py

from __future__ import annotations

from typing import Dict, Any
from pathlib import Path
import json
import torch

from src.data.supabase_client import get_supabase_client
from src.training.params_nn_model import HierarchicalParamsNN, OUTPUT_DIM

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_PATH = MODELS_DIR / "hier_params_nn.pt"
VOCAB_PATH = MODELS_DIR / "gamme_vocab.json"


def _load_vocab() -> Dict[str, int]:
    if not VOCAB_PATH.exists():
        raise FileNotFoundError(f"Missing vocab file: {VOCAB_PATH}")
    data = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    stoi = data.get("stoi", {}) or {}
    return {str(k): int(v) for k, v in stoi.items()}


def _load_model(num_gammes: int) -> HierarchicalParamsNN:
    model = HierarchicalParamsNN(num_gammes=num_gammes)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model weights: {MODEL_PATH}")
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def get_gamme_base_params(gamme_id: str) -> Dict[str, Any]:
    """
    Return the learned base depreciation params for a gamme_id
    from the hierarchical model: theta_gamme.
    """
    vocab = _load_vocab()
    if gamme_id not in vocab:
        return {"status": "unknown_gamme_id_in_vocab", "gamme_id": gamme_id}

    idx = vocab[gamme_id]
    model = _load_model(num_gammes=max(1, len(vocab)))

    with torch.no_grad():
        theta = model.gamme_base.weight[idx].detach().cpu().tolist()

    if len(theta) != OUTPUT_DIM:
        return {"status": "bad_theta_dim", "gamme_id": gamme_id, "theta_dim": len(theta)}

    delta, alpha, k_a, k_m, L, c = [float(x) for x in theta]

    return {
        "status": "ok",
        "gamme_id": gamme_id,
        "params": {
            "delta": delta,
            "alpha": alpha,
            "k_a": k_a,
            "k_m": k_m,
            "L": L,
            "c": c,
        },
    }


def update_gamme_params(gamme_id: str) -> Dict[str, Any]:
    """
    Write theta_gamme (base params) into Supabase gammes table.
    """
    client = get_supabase_client()

    res = get_gamme_base_params(gamme_id)
    if res.get("status") != "ok":
        return res

    params = res["params"]

    update_resp = (
        client.table("gammes")
        .update(
            {
                "delta": params["delta"],
                "alpha": params["alpha"],
                "k_a": params["k_a"],
                "k_m": params["k_m"],
                "L": params["L"],
                "c": params["c"],
            }
        )
        .eq("gamme_id", gamme_id)
        .execute()
    )

    return {
        "status": "ok",
        "gamme_id": gamme_id,
        "params": params,
        "supabase_update": update_resp.data,
    }
