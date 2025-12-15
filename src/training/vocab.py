# src/training/vocab.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import json
import torch
from torch import nn


@dataclass
class Vocab:
    """Simple persistent string->index vocabulary."""
    stoi: Dict[str, int]

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        if not path.exists():
            return cls(stoi={})
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(stoi={k: int(v) for k, v in data.get("stoi", {}).items()})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"stoi": self.stoi}, indent=2), encoding="utf-8")

    def add_many(self, keys: List[str]) -> int:
        """Add keys, return how many new items were added."""
        added = 0
        for k in keys:
            if k is None:
                continue
            if k not in self.stoi:
                self.stoi[k] = len(self.stoi)
                added += 1
        return added

    def get(self, key: Optional[str]) -> int:
        if key is None:
            return -1
        return self.stoi.get(key, -1)

    @property
    def size(self) -> int:
        return len(self.stoi)


def resize_embedding(emb: nn.Embedding, new_num_embeddings: int) -> nn.Embedding:
    """
    Resize an embedding layer while preserving existing weights.
    """
    old_num, dim = emb.weight.shape
    if new_num_embeddings <= old_num:
        return emb

    new_emb = nn.Embedding(new_num_embeddings, dim)
    # init new weights similar scale
    nn.init.normal_(new_emb.weight, mean=0.0, std=0.02)

    with torch.no_grad():
        new_emb.weight[:old_num].copy_(emb.weight)

    return new_emb
