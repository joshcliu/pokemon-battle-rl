"""Seeded Showdown battles with separate player-perspective poke-env objects."""

from __future__ import annotations

import json
import logging
import selectors
import shutil
import subprocess
from pathlib import Path

import numpy as np
from poke_env.battle import DoubleBattle
from poke_env.environment import DoublesEnv
from poke_env.player import Player
from poke_env.player.battle_order import DoubleBattleOrder

ROOT = Path(__file__).resolve().parents[2]
ACTION_SIZE = 107
DEFAULT_FORMAT = "gen9championsvgc2026regmc"


def node_binary() -> str:
    local = ROOT / ".cache/node-runtime/node_modules/node/bin/node"
    return str(local) if local.exists() else (shutil.which("node") or "node")


class LocalBattle:
    def __init__(self, battle_format: str = DEFAULT_FORMAT, timeout: float = 20):
        simulator = ROOT / ".cache/pokemon-showdown"
        if not (simulator / "dist/sim/index.js").exists():
            raise RuntimeError("Simulator missing; run python scripts/setup_simulator.py")
        self.format = battle_format
        self.timeout = timeout
        self.process = subprocess.Popen(
            [node_binary(), str(ROOT / "simulator/bridge.cjs"), str(simulator)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.players: dict[str, DoubleBattle] = {}
        self.requests: dict = {}
        self.log: list[str] = []
        self.input_log: list[dict] = []
        self.ended = False
        self.winner = None
        self.turn = 0

    def _call(self, command: dict) -> dict:
        assert self.process.stdin and self.process.stdout
        self.process.stdin.write(json.dumps(command) + "\n")
        self.process.stdin.flush()
        if not self.selector.select(self.timeout):
            self.close()
            raise TimeoutError("Showdown did not respond; battle was not counted as a draw")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Showdown exited unexpectedly")
        result = json.loads(line)
        if not result["ok"]:
            raise RuntimeError(result["error"])
        self.input_log.append(command)
        return result

    def reset(self, seed: int, team1: str, team2: str):
        self.log, self.input_log = [], []
        self.players = {
            side: DoubleBattle("battle-local-1", name, logging.getLogger(side), gen=9)
            for side, name in (("p1", "Learner"), ("p2", "Opponent"))
        }
        words = np.random.SeedSequence(seed).generate_state(4).astype(np.uint32)
        self._consume(
            self._call(
                {
                    "op": "reset",
                    "format": self.format,
                    "seed": [int(x % 65536) for x in words],
                    "teams": [team1, team2],
                }
            )
        )
        # Fixed input order determines leads. Team selection is outside the pilot scope.
        if any(req and req.get("teamPreview") for req in self.requests.values()):
            self.step(
                {
                    side: "team "
                    + "".join(
                        str(i + 1)
                        for i in range(min(req.get("maxTeamSize", 6), len(req["side"]["pokemon"])))
                    )
                    for side, req in self.requests.items()
                    if req and req.get("teamPreview")
                }
            )

    def _consume(self, result: dict):
        self.requests = result["requests"]
        self.log.extend(result["spectator"])
        self.ended, self.winner, self.turn = result["ended"], result["winner"], result["turn"]
        for side, player in self.players.items():
            for line in result["views"][side]:
                parts = line.split("|")
                if len(parts) < 2 or parts[1] in {"", "init", "title"} | Player.MESSAGES_TO_IGNORE:
                    continue
                if parts[1] == "win":
                    player.won_by(parts[2])
                elif parts[1] == "tie":
                    player.tied()
                else:
                    player.parse_message(parts)
            req = self.requests[side]
            if req and "side" in req:
                player.parse_request(req)

    def needs_action(self, side: str) -> bool:
        req = self.requests[side]
        return bool(not self.ended and req and not req.get("wait"))

    def legal_actions(self, side: str) -> dict[tuple[int, int], DoubleBattleOrder]:
        if not self.needs_action(side):
            return {}
        player = self.players[side]
        first, second = player.valid_orders
        # Mapping is isolated here because this upstream helper is version-specific.
        maps = [
            {
                id(order): int(DoublesEnv._order_to_action_individual(order, player, True, pos))
                for order in orders
            }
            for pos, orders in enumerate((first, second))
        ]
        result = {}
        for order in DoubleBattleOrder.join_orders(first, second):
            pair = (maps[0][id(order.first_order)], maps[1][id(order.second_order)])
            if min(pair) < 0:
                raise RuntimeError("Default/fallback actions are prohibited")
            result[pair] = order
        if not result:
            raise RuntimeError("No legal joint actions for an active request")
        return result

    def step(self, choices: dict[str, str]):
        self._consume(self._call({"op": "step", "choices": choices}))

    def save(self, directory: Path, stem: str):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{stem}.log").write_text("\n".join(self.log) + "\n")
        (directory / f"{stem}.input.json").write_text(json.dumps(self.input_log, indent=2))
        # Native poke-env replay export uses this player's visible history.
        self.players["p1"].save_replay(str(directory / f"{stem}.html"))

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.selector.close()
        for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
            if pipe:
                pipe.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
