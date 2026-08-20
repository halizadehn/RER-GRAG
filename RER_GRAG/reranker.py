from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _entity_text(entity: Any) -> str:
    """Return the textual representation used by the cross-encoder."""
    if isinstance(entity, dict):
        for key in ("description", "text", "summary", "content", "title", "name"):
            value = entity.get(key)
            if value:
                return str(value)
        return str(entity)

    for attr in ("description", "text", "summary", "content", "title", "name"):
        value = getattr(entity, attr, None)
        if value:
            return str(value)

    return str(entity)


class OutcomeSupervisedEntityReranker:
    """
    Cross-encoder entity reranker used by RER-GRAG and Tr-RER-GRAG.

    RER-GRAG uses the pretrained model only.
    Tr-RER-GRAG updates the same model online using bag-level answer correctness.
    """

    def __init__(
        self,
        *,
        base_model: str = "BAAI/bge-reranker-base",
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 256,
        pair_batch_size: int = 16,
        learning_rate: float = 1e-6,
        weight_decay: float = 0.01,
        fp16: bool = True,
        replay_buffer_size: int = 200,
        replay_probability: float = 0.3,
        replay_balance: float = 0.5,
        top_k_pool: int = 5,
        soft_topk_temperature: float = 0.7,
        gradient_clip: float = 1.0,
        seed: int = 13,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = int(max_length)
        self.pair_batch_size = int(pair_batch_size)
        self.fp16 = bool(fp16 and str(self.device).startswith("cuda"))

        self.replay_buffer_size = int(replay_buffer_size)
        self.replay_probability = float(replay_probability)
        self.replay_balance = float(replay_balance)
        self.top_k_pool = int(top_k_pool)
        self.soft_topk_temperature = float(soft_topk_temperature)
        self.gradient_clip = float(gradient_clip)

        self.rng = random.Random(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        source = model_dir if model_dir else base_model
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            source,
            num_labels=1,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

        self._pending_bag: Optional[Dict[str, Any]] = None
        self._replay_pos: List[Dict[str, Any]] = []
        self._replay_neg: List[Dict[str, Any]] = []
        self._training_enabled = False

    def set_training(self, enabled: bool) -> None:
        self._training_enabled = bool(enabled)
        if enabled:
            self.model.train()
        else:
            self.model.eval()

    def _encode(self, query: str, texts: List[str]) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            [query] * len(texts),
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def _forward_logits(self, query: str, texts: List[str]) -> torch.Tensor:
        pieces: List[torch.Tensor] = []
        for start in range(0, len(texts), self.pair_batch_size):
            chunk = texts[start : start + self.pair_batch_size]
            encoded = self._encode(query, chunk)
            if str(self.device).startswith("cuda"):
                with torch.amp.autocast("cuda", enabled=self.fp16):
                    logits = self.model(**encoded).logits.squeeze(-1)
            else:
                logits = self.model(**encoded).logits.squeeze(-1)
            pieces.append(logits)
        return torch.cat(pieces, dim=0)

    def _soft_topk_bag_logit(self, scores: torch.Tensor) -> torch.Tensor:
        k = min(self.top_k_pool, int(scores.numel()))
        top_values, _ = torch.topk(scores, k=k, largest=True)
        tau = max(self.soft_topk_temperature, 1e-6)
        weights = torch.softmax(top_values / tau, dim=0)
        return torch.sum(weights * top_values)

    @torch.no_grad()
    def rerank(self, query: str, entities: List[Any], k: int) -> List[Any]:
        if not entities:
            self._pending_bag = None
            return []

        texts = [_entity_text(entity) for entity in entities]

        was_training = self.model.training
        self.model.eval()
        scores = self._forward_logits(query, texts)
        order = torch.argsort(scores, descending=True).tolist()
        selected = [entities[i] for i in order[: int(k)]]

        self._pending_bag = {
            "query": str(query),
            "texts": texts,
        }

        if was_training and self._training_enabled:
            self.model.train()

        return selected

    def _push_replay(self, bag: Dict[str, Any]) -> None:
        buffer = self._replay_pos if int(bag["label"]) == 1 else self._replay_neg
        buffer.append(bag)
        if len(buffer) > self.replay_buffer_size:
            buffer.pop(0)

    def _sample_replay(self) -> Dict[str, Any]:
        have_pos = bool(self._replay_pos)
        have_neg = bool(self._replay_neg)

        if have_pos and have_neg:
            buffer = (
                self._replay_pos
                if self.rng.random() < self.replay_balance
                else self._replay_neg
            )
        elif have_pos:
            buffer = self._replay_pos
        elif have_neg:
            buffer = self._replay_neg
        else:
            raise RuntimeError("Replay buffer is empty.")

        return self.rng.choice(buffer)

    def update_last_bag(self, label: int) -> Optional[float]:
        """Apply one outcome-supervised MIL update using the last reranked pool."""
        if self._pending_bag is None:
            return None
        if not self._training_enabled:
            raise RuntimeError("Training is disabled. Call set_training(True) first.")

        current = dict(self._pending_bag)
        current["label"] = int(label)

        have_replay = bool(self._replay_pos or self._replay_neg)
        use_replay = have_replay and self.rng.random() < self.replay_probability
        bag = self._sample_replay() if use_replay else current

        query = bag["query"]
        texts = bag["texts"]
        target = torch.tensor(float(bag["label"]), device=self.device)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        scores = self._forward_logits(query, texts)
        bag_logit = self._soft_topk_bag_logit(scores)
        loss = self.loss_fn(bag_logit.view(()), target.view(()))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        self.optimizer.step()

        self._push_replay(current)
        self._pending_bag = None

        return float(loss.detach().cpu().item())

    def save(self, output_dir: str) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output)
        self.tokenizer.save_pretrained(output)

        metadata = {
            "max_length": self.max_length,
            "pair_batch_size": self.pair_batch_size,
            "replay_buffer_size": self.replay_buffer_size,
            "replay_probability": self.replay_probability,
            "replay_balance": self.replay_balance,
            "top_k_pool": self.top_k_pool,
            "soft_topk_temperature": self.soft_topk_temperature,
        }
        (output / "reranker_meta.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
