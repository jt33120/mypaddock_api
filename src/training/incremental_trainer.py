# src/training/incremental_trainer.py

from pathlib import Path
from typing import Dict, Any, List, Set

import torch
from torch import nn
from torch.utils.data import DataLoader

from .fetch_data import fetch_training_rows_since
from .dataset import TimeseriesDataset
from src.parametric.depreciation_function import price_from_params

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_PATH = MODELS_DIR / "params_nn.pt"


# ------------------------------
# Inline ParamsNN here to avoid import issues
# ------------------------------

# MUST match encode_sample() length in encoding.py
INPUT_DIM = 23           # if you change encode_sample, update this
OUTPUT_DIM = 6           # delta, alpha, k_a, k_m, L, c


class ParamsNN(nn.Module):
    """
    Global neural net mapping full (gamme + vehicle) features -> 6 parametric coefficients.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, INPUT_DIM]
        returns: [B, OUTPUT_DIM]
        """
        return self.net(x)


# ------------------------------
# Load / train helpers
# ------------------------------

def _load_or_init_model() -> ParamsNN:
    """
    Load existing ParamsNN weights from disk if present, otherwise
    initialize a fresh model.
    """
    model = ParamsNN()
    if MODEL_PATH.exists():
        state = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state)
    return model


def train_incremental_on_new_rows(
    since_date: str,
    epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
) -> Dict[str, Any]:
    """
    Incremental training on rows with date >= since_date.
    Intended to be called every time new timeseries rows are added.

    Returns:
        {
          "status": "ok" | "no_new_data",
          "avg_loss": float or None,
          "gamme_ids": List[str]
        }
    """
    # 1) Fetch joined timeseries+vehicles+gammes rows from Supabase
    rows = fetch_training_rows_since(since_date)
    if not rows:
        return {"status": "no_new_data", "avg_loss": None, "gamme_ids": []}

    # 2) Build dataset/dataloader
    dataset = TimeseriesDataset(rows)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3) Load or init model
    model = _load_or_init_model()
    model.train()

    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    total_loss = 0.0
    total_samples = 0
    gamme_ids: Set[str] = set()

    # 4) Training loop
    for epoch in range(epochs):
        for batch in loader:
            x = batch["nn_input"]           # [B, INPUT_DIM]
            V0 = batch["V0"]                # [B]
            age_years = batch["age_years"]  # [B]
            mileage = batch["mileage"]      # [B]
            y = batch["target_value"].unsqueeze(1)  # [B, 1]

            # NN maps full features -> 6 params
            params = model(x)               # [B, 6]

            # Parametric depreciation model -> predicted price
            y_hat = price_from_params(
                V0=V0,
                age_years=age_years,
                mileage=mileage,
                params=params,
            ).unsqueeze(1)                  # [B, 1]

            loss = loss_fn(y_hat, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            batch_size_actual = y.shape[0]
            total_loss += loss.item() * batch_size_actual
            total_samples += batch_size_actual

            # Track gammes that appeared in this batch
            for gid in batch["gamme_id"]:
                if gid is not None:
                    gamme_ids.add(gid)

    # 5) Compute average loss over all samples in all epochs
    avg_loss = total_loss / max(1, total_samples)

    # 6) Save updated model
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), MODEL_PATH)

    return {
        "status": "ok",
        "avg_loss": avg_loss,
        "gamme_ids": list(gamme_ids),
    }
