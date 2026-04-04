import json
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_entity_text_map(entities_parquet: str) -> Dict[str, str]:
    df = pd.read_parquet(entities_parquet)
    if "id" not in df.columns:
        raise ValueError("entities.parquet must contain column 'id'")
    df["id"] = df["id"].astype(str)

    text_col = None
    for c in ["description", "text", "summary", "content", "title"]:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        raise ValueError(
            f"entities.parquet must contain one of description/text/summary/content/title. Columns={list(df.columns)}"
        )

    df[text_col] = df[text_col].astype(str)
    return dict(zip(df["id"].tolist(), df[text_col].tolist()))



class OnlineCEBagReranker:
 
    def __init__(
        self,
        *,
        entities_parquet: str,
        base_model: str = "BAAI/bge-reranker-base",
        model_dir: Optional[str] = None,              
        teacher_model_name: Optional[str] = None,     
        device: Optional[str] = None,
        max_len: int = 256,
        pair_batch_size: int = 16,
        lr: float = 1e-6,                             
        weight_decay: float = 0.01,
        fp16: bool = True,
        grad_accum: int = 1,
        clip_grad: float = 1.0,
        replay_size: int = 200,
        replay_p: float = 0.3,
        replay_balance: float = 0.5,
        topk_pool: int = 5,
        pool_mode: str = "soft_topk",
        soft_topk_tau: float = 0.7,
        neg_update_min_prob: float = 0.0,
        pos_weight: Optional[float] = None,
        seed: int = 13,
        debug: bool = False,
        loss_log_path: str = "online_loss.csv",
        flush_every: int = 1,

        # ---- new stability knobs ----
        freeze_mode: str = "last_layer",             
        distill_weight: float = 0.3,                 
        blend_alpha: float = 0.6,                    
        warmup_steps: int = 100,                     
    ):
        self.debug = bool(debug)

        # Data
        self.ent_text = load_entity_text_map(entities_parquet)

        # Device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = int(max_len)
        self.pair_batch_size = int(pair_batch_size)

        # Mixed precision / optimization
        self.fp16 = bool(fp16 and str(self.device).startswith("cuda"))
        self.grad_accum = int(max(1, grad_accum))
        self.clip_grad = float(clip_grad)

        # Replay + pooling
        self.replay_size = int(replay_size)
        self.replay_p = float(replay_p)
        self.replay_balance = float(replay_balance)
        self.topk_pool = int(max(1, topk_pool))
        self.pool_mode = str(pool_mode)
        self.soft_topk_tau = float(soft_topk_tau)
        self.neg_update_min_prob = float(neg_update_min_prob)

        # New
        self.freeze_mode = str(freeze_mode)
        self.distill_weight = float(distill_weight)
        self.blend_alpha = float(blend_alpha)
        self.warmup_steps = int(max(0, warmup_steps))

        # RNG
        self.rng = random.Random(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Logging
        self.loss_log_path = str(loss_log_path)
        self.flush_every = int(max(1, flush_every))
        self._log_buf: List[str] = []

        logp = Path(self.loss_log_path)
        if not logp.exists():
            logp.parent.mkdir(parents=True, exist_ok=True)
            logp.write_text(
                "step,query,label,mil_loss,distill_loss,total_loss,bag_prob,num_candidates,used_replay,updated\n",
                encoding="utf-8",
            )
        if self.debug:
            print("[LOSS_LOG]", str(logp.resolve()))

        student_source = model_dir if model_dir else base_model
        self.tokenizer = AutoTokenizer.from_pretrained(student_source)
        self.model = AutoModelForSequenceClassification.from_pretrained(student_source, num_labels=1)
        self.model.to(self.device)

        if getattr(self.model.config, "num_labels", None) != 1:
            raise ValueError(
                f"Student model num_labels must be 1. Got num_labels={self.model.config.num_labels}."
            )

        teacher_source = teacher_model_name if teacher_model_name else base_model
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_source)
        self.teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_source, num_labels=1)
        self.teacher_model.to(self.device)
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

        if getattr(self.teacher_model.config, "num_labels", None) != 1:
            raise ValueError(
                f"Teacher model num_labels must be 1. Got num_labels={self.teacher_model.config.num_labels}."
            )

        self._apply_freeze_strategy(self.freeze_mode)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.fp16) if self.fp16 else None

        if pos_weight is None:
            self.bce = nn.BCEWithLogitsLoss()
        else:
            pw = torch.tensor([float(pos_weight)], device=self.device)
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)

        self._pending_bag: Optional[dict] = None
        self._replay_pos: List[dict] = []
        self._replay_neg: List[dict] = []
        self._step: int = 0

        self._train_mode: bool = True
        self.model.train()

    def _apply_freeze_strategy(self, freeze_mode: str):
        for p in self.model.parameters():
            p.requires_grad = (freeze_mode == "none")

        if freeze_mode == "none":
            return

        for name, p in self.model.named_parameters():
            if any(x in name for x in ["classifier", "score", "out_proj"]):
                p.requires_grad = True

        if freeze_mode == "head":
            return

        if freeze_mode == "last_layer":
            last_layer_patterns = [
                "encoder.layer.11",         
                "layer.11",
                "transformer.layer.11",
            ]
            for name, p in self.model.named_parameters():
                if any(pat in name for pat in last_layer_patterns):
                    p.requires_grad = True
            return

        raise ValueError("freeze_mode must be one of: none, head, last_layer")

    def _encode_pairs(self, query: str, texts: List[str], use_teacher: bool = False) -> Dict[str, torch.Tensor]:
        tok = self.teacher_tokenizer if use_teacher else self.tokenizer
        enc = tok(
            [query] * len(texts),
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def _forward_logits(self, query: str, texts: List[str]) -> torch.Tensor:
        all_logits = []
        for i in range(0, len(texts), self.pair_batch_size):
            chunk = texts[i : i + self.pair_batch_size]
            enc = self._encode_pairs(query, chunk, use_teacher=False)
            out = self.model(**enc)
            logits = out.logits
            if logits.dim() != 2 or logits.size(-1) != 1:
                raise ValueError(f"Expected student logits shape [B,1], got {tuple(logits.shape)}")
            all_logits.append(logits.squeeze(-1))
        return torch.cat(all_logits, dim=0)

    @torch.no_grad()
    def _forward_teacher_logits(self, query: str, texts: List[str]) -> torch.Tensor:
        all_logits = []
        for i in range(0, len(texts), self.pair_batch_size):
            chunk = texts[i : i + self.pair_batch_size]
            enc = self._encode_pairs(query, chunk, use_teacher=True)
            out = self.teacher_model(**enc)
            logits = out.logits
            if logits.dim() != 2 or logits.size(-1) != 1:
                raise ValueError(f"Expected teacher logits shape [B,1], got {tuple(logits.shape)}")
            all_logits.append(logits.squeeze(-1))
        return torch.cat(all_logits, dim=0)

    def _bag_logit_topk_mean(self, u: torch.Tensor) -> torch.Tensor:
        k = min(self.topk_pool, int(u.numel()))
        topv, _ = torch.topk(u, k=k, largest=True)
        return topv.mean()

    def _bag_logit_soft_topk(self, u: torch.Tensor) -> torch.Tensor:
        k = min(self.topk_pool, int(u.numel()))
        topv, _ = torch.topk(u, k=k, largest=True)
        tau = max(1e-6, float(self.soft_topk_tau))
        w = torch.softmax(topv / tau, dim=0)
        return (w * topv).sum()

    def _bag_logit(self, u: torch.Tensor) -> torch.Tensor:
        if self.pool_mode == "topk_mean":
            return self._bag_logit_topk_mean(u)
        if self.pool_mode == "soft_topk":
            return self._bag_logit_soft_topk(u)
        raise ValueError(f"Unknown pool_mode={self.pool_mode}")

    def _safe_query(self, q: str) -> str:
        return str(q).replace("\n", " ").replace("\r", " ").replace(",", " ")

    def _flush_log(self):
        if not self._log_buf:
            return
        with open(self.loss_log_path, "a", encoding="utf-8") as f:
            f.writelines(self._log_buf)
            f.flush()
        self._log_buf.clear()

    def _push_replay(self, bag: dict):
        if int(bag["label"]) == 1:
            self._replay_pos.append(bag)
            if len(self._replay_pos) > self.replay_size:
                self._replay_pos.pop(0)
        else:
            self._replay_neg.append(bag)
            if len(self._replay_neg) > self.replay_size:
                self._replay_neg.pop(0)

    def _sample_replay(self) -> dict:
        have_pos = len(self._replay_pos) > 0
        have_neg = len(self._replay_neg) > 0

        if have_pos and have_neg:
            use_pos = (self.rng.random() < self.replay_balance)
            buf = self._replay_pos if use_pos else self._replay_neg
            return self.rng.choice(buf)

        if have_pos:
            return self.rng.choice(self._replay_pos)
        if have_neg:
            return self.rng.choice(self._replay_neg)

        raise RuntimeError("Replay requested but buffers are empty.")

    def set_train_mode(self, train: bool):
        self._train_mode = bool(train)
        if self._train_mode:
            self.model.train()
        else:
            self.model.eval()

    def reset_optimizer(self, lr: Optional[float] = None):
        if lr is None:
            lr = self.opt.param_groups[0]["lr"]
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(
            trainable_params,
            lr=float(lr),
            weight_decay=self.opt.param_groups[0].get("weight_decay", 0.01),
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.fp16) if self.fp16 else None

    @torch.no_grad()
    def rerank(
        self,
        query: str,
        entities: List[object],
        k: int,
        entity_id_attr: str = "id",
    ) -> List[object]:
        kept_entities: List[object] = []
        kept_ids: List[str] = []
        texts: List[str] = []

        for e in entities:
            eid = getattr(e, entity_id_attr, None)
            if eid is None:
                continue
            eid = str(eid)
            t = self.ent_text.get(eid)
            if not t:
                continue
            kept_entities.append(e)
            kept_ids.append(eid)
            texts.append(t)

        if not kept_entities:
            self._pending_bag = None
            return []

        self.model.eval()

        u_student = self._forward_logits(query, texts)
        u_teacher = self._forward_teacher_logits(query, texts)

        u_final = self.blend_alpha * u_teacher + (1.0 - self.blend_alpha) * u_student

        student_scores = u_student.detach().float().cpu().numpy().tolist()
        teacher_scores = u_teacher.detach().float().cpu().numpy().tolist()
        final_scores = u_final.detach().float().cpu().numpy().tolist()

        z = self._bag_logit(u_final)
        bag_prob = float(torch.sigmoid(z).item())

        order = sorted(range(len(kept_entities)), key=lambda i: final_scores[i], reverse=True)
        selected = [kept_entities[i] for i in order[: int(k)]]

        self._pending_bag = {
            "query": str(query),
            "cand_ids": kept_ids,
            "cand_texts": texts,
            "bag_prob": bag_prob,
        }

        if self.debug:
            print(f"[RERANK] bag_prob={bag_prob:.4f} num_cand={len(texts)}")
            top = order[: min(k, len(order))]
            for r, i in enumerate(top, 1):
                print(
                    f"  top{r}: id={kept_ids[i]} "
                    f"student={student_scores[i]:.4f} teacher={teacher_scores[i]:.4f} final={final_scores[i]:.4f}"
                )

        if self._train_mode:
            self.model.train()

        return selected

    def update_last_bag(self, label: int):
        if self._pending_bag is None:
            if self.debug:
                print("[UPDATE] No pending bag (did rerank() run for this query?)")
            return

        label = int(label)
        self._pending_bag["label"] = label

        if label == 0 and self.neg_update_min_prob > 0:
            p = float(self._pending_bag.get("bag_prob", 0.0))
            if p < self.neg_update_min_prob:
                if self.debug:
                    print(f"[UPDATE] Skip negative update (bag_prob={p:.3f} < {self.neg_update_min_prob})")
                self._pending_bag = None
                return

        have_replay = (len(self._replay_pos) + len(self._replay_neg)) > 0
        use_replay = have_replay and (self.rng.random() < self.replay_p)
        bag = self._sample_replay() if use_replay else self._pending_bag

        q = bag["query"]
        texts = bag["cand_texts"]
        y = torch.tensor(float(bag["label"]), device=self.device)

        if len(texts) < 2:
            if self.debug:
                print("[UPDATE] Skip (too few candidates).")
            self._pending_bag = None
            return

        self._step += 1

        if self._train_mode:
            self.model.train()
        else:
            self.model.eval()

        if str(self.device).startswith("cuda"):
            ctx = torch.amp.autocast("cuda", enabled=self.fp16)
        else:
            ctx = nullcontext()

        with ctx:
            u_student = self._forward_logits(q, texts)
            with torch.no_grad():
                u_teacher = self._forward_teacher_logits(q, texts)

            u_final = self.blend_alpha * u_teacher + (1.0 - self.blend_alpha) * u_student
            z = self._bag_logit(u_final)
            mil_loss = self.bce(z.view(()), y.view(()))

            if self.distill_weight > 0:
                distill_loss = F.mse_loss(u_student, u_teacher)
            else:
                distill_loss = torch.zeros((), device=self.device)

            total_loss = mil_loss + self.distill_weight * distill_loss

        do_update = self._train_mode and (self._step > self.warmup_steps)

        if do_update:
            if self.scaler is not None:
                self.scaler.scale(total_loss / self.grad_accum).backward()
                if (self._step % self.grad_accum) == 0:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.clip_grad
                    )
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.opt.zero_grad(set_to_none=True)
            else:
                (total_loss / self.grad_accum).backward()
                if (self._step % self.grad_accum) == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.clip_grad
                    )
                    self.opt.step()
                    self.opt.zero_grad(set_to_none=True)

        self._push_replay(self._pending_bag)

        bag_prob = float(torch.sigmoid(z).item())
        num_cand = int(len(texts))
        used_replay_int = int(bool(use_replay))
        updated_int = int(bool(do_update))

        line = (
            f"{self._step},"
            f"{self._safe_query(q)},"
            f"{int(bag['label'])},"
            f"{float(mil_loss.item()):.6f},"
            f"{float(distill_loss.item()):.6f},"
            f"{float(total_loss.item()):.6f},"
            f"{bag_prob:.6f},"
            f"{num_cand},"
            f"{used_replay_int},"
            f"{updated_int}\n"
        )

        self._log_buf.append(line)
        if (self._step % self.flush_every) == 0:
            self._flush_log()

        if self.debug:
            print(
                f"[UPDATE] step={self._step} label={int(bag['label'])} "
                f"mil={float(mil_loss.item()):.4f} "
                f"distill={float(distill_loss.item()):.4f} "
                f"total={float(total_loss.item()):.4f} "
                f"bag_prob={bag_prob:.3f} used_replay={used_replay_int} "
                f"updated={updated_int} train={int(self._train_mode)} "
                f"replay_pos={len(self._replay_pos)} replay_neg={len(self._replay_neg)}"
            )

        self._pending_bag = None

    def save(self, out_dir: str):
        self._flush_log()

        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(outp)
        self.tokenizer.save_pretrained(outp)

        meta = {
            "max_len": self.max_len,
            "pair_batch_size": self.pair_batch_size,
            "device": self.device,
            "pool_mode": self.pool_mode,
            "topk_pool": self.topk_pool,
            "soft_topk_tau": self.soft_topk_tau,
            "online": True,
            "replay_size": self.replay_size,
            "replay_p": self.replay_p,
            "replay_balance": self.replay_balance,
            "neg_update_min_prob": self.neg_update_min_prob,
            "freeze_mode": self.freeze_mode,
            "distill_weight": self.distill_weight,
            "blend_alpha": self.blend_alpha,
            "warmup_steps": self.warmup_steps,
        }
        with open(outp / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if self.debug:
            print(f"[SAVE] Saved reranker to: {outp.resolve()}")

    def close(self):
        self._flush_log()
