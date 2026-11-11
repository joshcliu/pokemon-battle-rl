"""Fetch and build exactly the simulator revision recorded in simulator.lock.json."""

import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
lock = json.loads((ROOT / "simulator.lock.json").read_text())
cache = ROOT / ".cache"
cache.mkdir(exist_ok=True)
node_root = cache / "node-runtime"
node = node_root / "node_modules/node/bin/node"


def run(*args, cwd=ROOT, **kwargs):
    return subprocess.run(args, cwd=cwd, check=True, **kwargs)


if not node.exists():
    run(
        "npm",
        "install",
        "--prefix",
        str(node_root),
        "--no-audit",
        "--no-fund",
        "node@" + lock["node"],
    )
version = run(str(node), "--version", capture_output=True, text=True).stdout.strip()
if version != "v" + lock["node"]:
    raise RuntimeError(f"Expected Node {lock['node']}, found {version}")
sim = cache / "pokemon-showdown"
if not sim.exists():
    sim.mkdir()
    run("git", "init", cwd=sim)
    run("git", "remote", "add", "origin", lock["repository"], cwd=sim)
current = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=sim, capture_output=True, text=True
).stdout.strip()
if current != lock["commit"]:
    dirty = run("git", "status", "--porcelain", cwd=sim, capture_output=True, text=True).stdout
    if dirty.strip():
        raise RuntimeError("Simulator has local changes; preserve them before changing revision")
    run("git", "fetch", "--depth", "1", "origin", lock["commit"], cwd=sim)
    run("git", "checkout", "--detach", lock["commit"], cwd=sim)
npm = shutil.which("npm")
environment = os.environ.copy()
environment["PATH"] = str(node.parent) + os.pathsep + environment.get("PATH", "")
run(npm, "ci", "--omit=dev", "--omit=optional", "--no-audit", "--no-fund", cwd=sim, env=environment)
run(str(node), "build", cwd=sim, env=environment)
print(f"Ready: Showdown {lock['commit']} / Node {lock['node']}")
