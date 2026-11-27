"""
src/models/neural_cf.py
Neural Collaborative Filtering: user/item embeddings feeding an MLP that
learns non-linear interactions beyond what a dot product (SVD) captures.

Upgrades over the prototype:
  - mini-batch training (was full-batch)
  - held-out validation split with early stopping (was fixed 15 epochs, no validation)
  - reproducible (seeded) train/val split and weight init
  - device handling (uses GPU if available, else CPU)
  - model save/load so it doesn't retrain on every process start
  - kept intentionally small (2 hidden layers) — a bigger net would overfit
    ~11K ratings, not help
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.utils import set_seed, get_device


class NCFModel(nn.Module):
    def __init__(self, n_users, n_items, n_factors=16, hidden_dims=(32, 16)):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)
        layers = []
        in_dim = n_factors * 2
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, u, i):
        x = torch.cat([self.user_emb(u), self.item_emb(i)], dim=-1)
        return self.mlp(x).squeeze(-1)


class NeuralCFModel:
    def __init__(self, n_factors=16, hidden_dims=(32, 16), epochs=60, batch_size=256,
                 lr=0.005, val_fraction=0.15, early_stopping_patience=6, weight_decay=1e-5,
                 seed=42):
        self.n_factors = n_factors
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.val_fraction = val_fraction
        self.patience = early_stopping_patience
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = get_device()
        self.history = {"train_loss": [], "val_loss": []}

    def fit(self, ratings_df: pd.DataFrame):
        set_seed(self.seed)
        self.user_ids = sorted(ratings_df.userId.unique())
        self.movie_ids = sorted(ratings_df.movieId.unique())
        self.uid_map = {u: i for i, u in enumerate(self.user_ids)}
        self.mid_map = {m: i for i, m in enumerate(self.movie_ids)}

        u = np.array([self.uid_map[x] for x in ratings_df.userId])
        i = np.array([self.mid_map[x] for x in ratings_df.movieId])
        r = ratings_df.rating.values.astype(np.float32)

        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(u))
        u, i, r = u[perm], i[perm], r[perm]
        n_val = max(1, int(len(u) * self.val_fraction)) if len(u) > 20 else 0
        u_train, i_train, r_train = u[n_val:], i[n_val:], r[n_val:]
        u_val, i_val, r_val = u[:n_val], i[:n_val], r[:n_val]

        self.model = NCFModel(len(self.user_ids), len(self.movie_ids),
                               self.n_factors, self.hidden_dims).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        train_ds = TensorDataset(torch.tensor(u_train), torch.tensor(i_train),
                                  torch.tensor(r_train))
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        best_val = float("inf")
        best_state = None
        patience_left = self.patience

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            for bu, bi, br in train_loader:
                bu, bi, br = bu.to(self.device), bi.to(self.device), br.to(self.device)
                opt.zero_grad()
                preds = self.model(bu, bi)
                loss = loss_fn(preds, br)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * len(bu)
            train_loss = epoch_loss / len(train_ds)
            self.history["train_loss"].append(train_loss)

            if n_val > 0:
                self.model.eval()
                with torch.no_grad():
                    vu = torch.tensor(u_val).to(self.device)
                    vi = torch.tensor(i_val).to(self.device)
                    vr = torch.tensor(r_val).to(self.device)
                    val_loss = loss_fn(self.model(vu, vi), vr).item()
                self.history["val_loss"].append(val_loss)

                if val_loss < best_val - 1e-4:
                    best_val = val_loss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    patience_left = self.patience
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def predict(self, user_id, movie_id) -> float:
        if user_id not in self.uid_map or movie_id not in self.mid_map:
            return 3.0
        u = torch.tensor([self.uid_map[user_id]]).to(self.device)
        i = torch.tensor([self.mid_map[movie_id]]).to(self.device)
        with torch.no_grad():
            pred = self.model(u, i).item()
        return float(np.clip(pred, 0.5, 5.0))

    def score_all(self, user_id, candidate_movie_ids):
        """Batched prediction — one forward pass for all candidates instead
        of one tensor op per item (which dominated runtime during evaluation
        across hundreds of users x hundreds of candidates x multiple models)."""
        has_signal = user_id in self.uid_map
        if not candidate_movie_ids:
            return {}, has_signal
        if not has_signal:
            return {m: 3.0 for m in candidate_movie_ids}, has_signal

        known = [m for m in candidate_movie_ids if m in self.mid_map]
        unknown = [m for m in candidate_movie_ids if m not in self.mid_map]
        scores = {m: 3.0 for m in unknown}
        if known:
            u_idx = self.uid_map[user_id]
            u_batch = torch.full((len(known),), u_idx, dtype=torch.long).to(self.device)
            i_batch = torch.tensor([self.mid_map[m] for m in known], dtype=torch.long).to(self.device)
            with torch.no_grad():
                preds = self.model(u_batch, i_batch).cpu().numpy()
            preds = np.clip(preds, 0.5, 5.0)
            scores.update({m: float(p) for m, p in zip(known, preds)})
        return scores, has_signal

    def save(self, path="models/neural_cf.pt"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "uid_map": self.uid_map, "mid_map": self.mid_map,
            "n_factors": self.n_factors, "hidden_dims": self.hidden_dims,
        }, path)

    @classmethod
    def load(cls, path="models/neural_cf.pt"):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(n_factors=checkpoint["n_factors"], hidden_dims=checkpoint["hidden_dims"])
        obj.uid_map = checkpoint["uid_map"]
        obj.mid_map = checkpoint["mid_map"]
        obj.user_ids = list(obj.uid_map.keys())
        obj.movie_ids = list(obj.mid_map.keys())
        obj.model = NCFModel(len(obj.uid_map), len(obj.mid_map),
                              obj.n_factors, obj.hidden_dims)
        obj.model.load_state_dict(checkpoint["state_dict"])
        obj.model.eval()
        return obj
