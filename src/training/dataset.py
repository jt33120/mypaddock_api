# src/training/dataset.py

from typing import List, Dict, Any
from datetime import datetime

import torch
from torch.utils.data import Dataset

from .encoding import encode_sample


class TimeseriesDataset(Dataset):
    """
    Dataset using real timeseries rows joined with vehicles + gammes.
    Each item returns:
      - nn_input: full feature vector using *all* relevant columns
      - V0, age, mileage: for the parametric depreciation function
      - target_value: observed price
      - gamme_id: to know which gamme was touched
    """

    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]

        # Supabase structure after our SELECT:
        # row["vehicles"] -> vehicle dict
        # row["vehicles"]["gammes"] -> gamme dict
        v = row["vehicles"]
        g = v["gammes"]

        # Target price (ground truth)
        target_value = torch.tensor(float(row["value"]), dtype=torch.float32)

        # Date / age
        date = datetime.fromisoformat(row["date"]).date()
        year = float(g.get("year") or 0.0)
        age_years = torch.tensor(max(0.0, date.year - year), dtype=torch.float32)

        # Mileage
        mileage = torch.tensor(float(row.get("mileage") or 0.0), dtype=torch.float32)

        # V0 from gamme (fixed per gamme)
        V0 = torch.tensor(float(g.get("V0") or 0.0), dtype=torch.float32)

        # Full input vector for the NN, using *all* relevant fields
        nn_input = encode_sample(v, g)  # [INPUT_DIM]

        return {
            "nn_input": nn_input,              # [INPUT_DIM]  -> NN
            "V0": V0,                          # [ ]         -> depreciation func
            "age_years": age_years,            # [ ]
            "mileage": mileage,                # [ ]
            "target_value": target_value,      # [ ]
            "gamme_id": g.get("gamme_id"),
        }
