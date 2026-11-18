"""Fit an MC461-specific one-step win predictor using only the training-roster sweep."""

import argparse
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from analyze_long_rewards import write_csv
from evaluate_challenger_transfer import digest, read
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str
from preview_sweep import BASE, COUNT, LINEUPS, RUN, read_games, save
from torch import nn

EPOCHS = (30, 90, 180)
FIT_SEED = 829
TYPES = "Bug Dark Dragon Electric Fairy Fighting Fire Flying Ghost Grass Ground Ice Normal Poison Psychic Rock Steel Water".split()
STATS = ("hp", "atk", "def", "spa", "spd", "spe")


def audit_sweep(root, stage):
    spec = read(root / "manifest.json")
    teams = spec["teams"][stage]
    rates, previews, table = [], [], []
    for i, team in enumerate(teams):
        folder = root / stage / Path(team).stem
        public = set()
        wins = []
        for lineup in LINEUPS:
            path = folder / f"{lineup}-{next(iter(spec['opponents']))}.csv.gz"
            games = read_games(path)
            expected = {
                (
                    family,
                    BASE[stage] + (i * len(spec["opponents"]) + j) * 1000 + k,
                    "p1" if k % 2 == 0 else "p2",
                )
                for j, family in enumerate(spec["opponents"])
                for k in range(COUNT[stage])
            }
            assert len(games) == len(expected)
            assert {(r["family"], int(r["seed"]), r["side"]) for r in games} == expected
            assert all(
                r["learner"] == spec["parent"]
                and r["team"] == spec["team"]
                and r["opponent_team"] == team
                and r["candidate"] == r["lineup"] == lineup
                and r["opponent"] == spec["opponents"][r["family"]]
                for r in games
            )
            public.update(r["preview"] for r in games)
            counts = Counter(r["outcome"] for r in games)
            assert set(counts) <= {"win", "loss", "tie", "cutoff"}
            wins.append(counts["win"])
            table.append(
                dict(
                    split=stage,
                    team=Path(team).stem,
                    lineup=lineup,
                    wins=counts["win"],
                    battles=len(games),
                    ties=counts["tie"],
                    cutoffs=counts["cutoff"],
                )
            )
        assert len(public) == 1
        preview = tuple(public.pop().split("|"))
        assert len(preview) == 6 and preview == tuple(sorted(preview))
        previews.append(preview)
        rates.append(wins)
    return teams, previews, np.asarray(rates), table


@lru_cache
def describe(public_species, vocabulary):
    dex = GenData.from_gen(9).pokedex
    values = []
    for name in public_species:
        mon = dex[to_id_str(name)]
        values.append(
            [
                *(float(t in mon["types"]) for t in TYPES),
                *(mon["baseStats"][s] / 150 for s in STATS),
                *(float(name == v) for v in vocabulary),
            ]
        )
    return torch.tensor(values, dtype=torch.float32)


class PreviewHead(nn.Module):
    """Score legal lead/back pairs after attention over the 12 team-preview slots."""

    def __init__(self, vocabulary):
        super().__init__()
        self.vocabulary = tuple(vocabulary)
        width = 32
        self.foe = nn.Linear(len(TYPES) + len(STATS) + len(vocabulary), width)
        self.own = nn.Parameter(torch.randn(6, width) * 0.15)
        self.attention = nn.TransformerEncoderLayer(
            width, 2, width * 2, dropout=0, activation="gelu", batch_first=True
        )
        self.scorer = nn.Sequential(nn.Linear(width * 3, width), nn.GELU(), nn.Linear(width, 1))
        self.bias = nn.Parameter(torch.zeros(len(LINEUPS)))
        self.register_buffer("slots", torch.tensor([[int(c) - 1 for c in a] for a in LINEUPS]))

    def forward(self, x):
        context = self.attention(torch.cat([self.own.expand(len(x), -1, -1), self.foe(x)], dim=1))
        lead = context[:, self.slots[:, :2]].mean(2)
        back = context[:, self.slots[:, 2:]].mean(2)
        opponent = context[:, 6:].mean(1, keepdim=True).expand_as(lead)
        return self.bias + self.scorer(torch.cat([lead, back, opponent], dim=-1)).squeeze(-1)

    @torch.inference_mode()
    def choose(self, public):
        return LINEUPS[self(describe(tuple(public), self.vocabulary)[None]).argmax().item()]


def load_model(path):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    assert tuple(obj["lineups"]) == LINEUPS
    model = PreviewHead(obj["vocabulary"])
    model.load_state_dict(obj["state"])
    return model.eval()


def fit(previews, targets, train, epochs):
    torch.manual_seed(FIT_SEED)
    vocabulary = sorted({s for i in train for s in previews[i]})
    model = PreviewHead(vocabulary)
    x = torch.stack([describe(previews[i], tuple(vocabulary)) for i in train])
    y = targets[train]
    with torch.no_grad():
        model.bias.copy_(torch.logit(y.mean(0).clamp(0.01, 0.99)))
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    for e in range(1, max(epochs) + 1):
        logits = model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if e in epochs:
            yield e, model


def train(root):
    torch.set_num_threads(1)
    if (root / "selection-lock.json").exists():
        raise FileExistsError("Preview selection is already locked")
    teams, previews, wins, table = audit_sweep(root, "train")
    n = COUNT["train"] * len(read(root / "manifest.json")["opponents"])
    targets = torch.tensor(wins / n, dtype=torch.float32)
    cv = []
    for val in range(len(teams)):
        indices = [i for i in range(len(teams)) if i != val]
        fold_fixed = wins[indices].sum(0).argmax()
        for epoch, model in fit(previews, targets, indices, EPOCHS):
            choice = model.choose(previews[val])
            cv.append(
                dict(
                    team=Path(teams[val]).stem,
                    epoch=epoch,
                    choice=choice,
                    wins=int(wins[val, LINEUPS.index(choice)]),
                    battles=n,
                    fold_fixed=LINEUPS[fold_fixed],
                    fold_fixed_wins=int(wins[val, fold_fixed]),
                )
            )
    peak = max(EPOCHS, key=lambda e: sum(r["wins"] for r in cv if r["epoch"] == e))
    _, model = next(fit(previews, targets, list(range(len(teams))), (peak,)))
    dest = root / "preview-head.pt"
    torch.save(dict(vocabulary=model.vocabulary, state=model.state_dict(), lineups=LINEUPS), dest)
    assert all(model.choose(p) == load_model(dest).choose(tuple(reversed(p))) for p in previews)
    write_csv(root / "training-sweep.csv", table)
    write_csv(root / "training-cv.csv", cv)
    save(
        root / "selection-lock.json",
        dict(
            train_teams=teams,
            epoch=peak,
            fit_seed=FIT_SEED,
            candidate_epochs=EPOCHS,
            fixed=LINEUPS[wins.sum(0).argmax()],
            head_sha256=digest(dest),
            training_table_sha256=digest(root / "training-sweep.csv"),
            tie_break="first canonical lineup or earliest epoch",
        ),
    )
    print(read(root / "selection-lock.json"), flush=True)


def winners(root):
    lock = read(root / "selection-lock.json")
    assert digest(root / "preview-head.pt") == lock["head_sha256"]
    selected = {}
    for stage in ("train", "test"):
        teams, _, wins, table = audit_sweep(root, stage)
        selected.update({t: LINEUPS[row.argmax()] for t, row in zip(teams, wins, strict=True)})
        if stage == "test":
            write_csv(root / "test-sweep.csv", table)
    save(root / "sweep-winners.json", selected)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("train", "winners"))
    p.add_argument("--root", type=Path, default=RUN)
    a = p.parse_args()
    (train if a.stage == "train" else winners)(a.root)
