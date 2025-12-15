# src/training/dataset.py
from typing import List, Dict, Any
from datetime import datetime

import torch
from torch.utils.data import Dataset

from .encoding import encode_sample


class TimeseriesDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        supabase_weight: float = 1.0,
        marketcheck_weight: float = 0.25,
    ):
        self.rows = rows
        self.supabase_weight = float(supabase_weight)
        self.marketcheck_weight = float(marketcheck_weight)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        v = row["vehicles"]
        g = v["gammes"]

        target_value = torch.tensor(float(row["value"]), dtype=torch.float32)

        date = datetime.fromisoformat(row["date"]).date()
        year = float(g.get("year") or 0.0)
        age_years = torch.tensor(max(0.0, date.year - year), dtype=torch.float32)

        mileage = torch.tensor(float(row.get("mileage") or 0.0), dtype=torch.float32)
        V0 = torch.tensor(float(g.get("V0") or 0.0), dtype=torch.float32)

        nn_input = encode_sample(v, g)

        source = str(row.get("source") or "supabase")
        w = self.supabase_weight if source == "supabase" else self.marketcheck_weight
        sample_weight = torch.tensor(w, dtype=torch.float32)

        gamme_id = g.get("gamme_id")  # must exist for hierarchical base params

        return {
            "nn_input": nn_input,              # [INPUT_DIM]
            "V0": V0,                          # []
            "age_years": age_years,            # []
            "mileage": mileage,                # []
            "target_value": target_value,      # []
            "gamme_id": gamme_id,              # str
            "source": source,                  # str
            "sample_weight": sample_weight,    # []
        }
