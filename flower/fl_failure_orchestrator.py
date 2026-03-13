#!/usr/bin/env python3
"""
FL Failure Orchestrator
=======================
Generates deterministic client failure plans and injects them live into
Flower federated learning experiments running on GCP.

Designed for the Flybold benchmark setup:
  - 10 clients (0–9), distributed across 5 GCP VMs (2 clients/VM)
  - 15 rounds, ~1h each
  - Dim_4: 25% / 50% / 75% failing clients

The failure plan is generated ONCE per failure-pct and reused across ALL
experiments (different strategies, datasets) so cross-experiment comparisons
are fair — identical clients fail in identical rounds every time.

USAGE
─────
# Step 1 — Generate plans (ONCE before running any experiments)
  python3 fl_failure_orchestrator.py generate-plan --failure-pct 25
  python3 fl_failure_orchestrator.py generate-plan --failure-pct 50
  python3 fl_failure_orchestrator.py generate-plan --failure-pct 75

# Step 2 — During an experiment, run the orchestrator on the server VM
  python3 fl_failure_orchestrator.py run --failure-pct 25 --run-id EXP_11213141

# Inspect a saved plan without running anything
  python3 fl_failure_orchestrator.py show-plan --failure-pct 25

FAILURE SEMANTICS
─────────────────
  fail_round    → client is STOPPED after EVALUATION of this round is logged.
                  The client HAS already submitted its update — nothing is lost.
  restart_round → client is STARTED after EVALUATION of this round is logged.
                  The client rejoins beginning from round (restart_round + 1).
  absent_rounds → list of rounds where the client contributes NO update to aggregation.

ROUND DETECTION
───────────────
We tail the fl-server Docker container logs and match:
  [SERVER] ROUND <N> EVALUATION SUMMARY
This line appears after both training AND evaluation are complete, making it
the true signal that round N is fully done and it is safe to stop/start clients.
Round numbers in the log are 0-indexed; we convert to 1-indexed internally.

PLAN PERSISTENCE
────────────────
Plans saved to: ./failure_plans/failure_plan_dim4_{pct}.json
These are the canonical source of truth. Never regenerate mid-experiment set.

LOGS
────
Per-run logs: ./orchestrator_logs/{RUN_ID}_dim4_{pct}pct_{timestamp}.log
All actions, SSH output, round completions, and errors are written here.
"""

import argparse
import json
import math
import os
import random
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

TOTAL_CLIENTS    = 10       # fl-client-0 … fl-client-9
NUM_ROUNDS       = 30
PLANS_DIR        = Path("./failure_plans")
LOGS_DIR         = Path("./orchestrator_logs")
VM_INFO_FILE     = Path("./vm-info.txt")
SERVER_CONTAINER = "fl-server"

# The true end-of-round signal: evaluation summary (0-indexed in logs)
# Example: [SERVER] ROUND 0 EVALUATION SUMMARY
ROUND_EVAL_RE = re.compile(r'\[SERVER\] ROUND (\d+) EVALUATION SUMMARY')

# Client-to-VM mapping: clients 0,1 → VM1 | 2,3 → VM2 | … | 8,9 → VM5
def client_vm_num(client_id: int) -> int:
    return client_id // 2 + 1


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def utcnow_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sudo_mkdir(path: Path):
    """Create a directory using sudo - required on GCP VMs where /app is root-owned."""
    result = subprocess.run(
        ["sudo", "mkdir", "-p", str(path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise PermissionError(
            f"sudo mkdir -p {path} failed: {result.stderr.strip()}"
        )


def load_vm_info() -> dict:
    """Parse vm-info.txt into a flat key→value dict."""
    if not VM_INFO_FILE.exists():
        raise FileNotFoundError(
            f"vm-info.txt not found at {VM_INFO_FILE.resolve()}\n"
            "Run deploy-application.sh first, or place vm-info.txt in the current directory."
        )
    info = {}
    with open(VM_INFO_FILE) as fh:
        for line in fh:
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                info[k.strip()] = v.strip()
    return info


def get_client_vm(client_id: int, vm_info: dict) -> tuple:
    """Return (vm_name, zone) for a given client_id."""
    vm_num = client_vm_num(client_id)
    key_vm   = f"CLIENT_{vm_num}_VM"
    key_zone = f"CLIENT_{vm_num}_ZONE"
    if key_vm not in vm_info or key_zone not in vm_info:
        raise KeyError(
            f"vm-info.txt is missing '{key_vm}' or '{key_zone}' "
            f"(needed for client_id={client_id})"
        )
    return vm_info[key_vm], vm_info[key_zone]


# ══════════════════════════════════════════════════════════════════════════════
#  LOGGER
# ══════════════════════════════════════════════════════════════════════════════

class Logger:
    """Dual-output (stdout + file) logger with level icons. Line-buffered so
    data survives a crash mid-experiment."""

    LEVELS = {
        "DEBUG":   "🔍",
        "INFO":    "ℹ️ ",
        "ACTION":  "⚙️ ",
        "ROUND":   "🔄",
        "WARN":    "⚠️ ",
        "ERROR":   "❌",
        "SUCCESS": "✅",
    }

    def __init__(self, log_file: Path):
        self.log_file = log_file
        sudo_mkdir(log_file.parent)
        # Create the file as root then make it world-writable so open() works
        subprocess.run(["sudo", "touch", str(log_file)], check=True)
        subprocess.run(["sudo", "chmod", "666", str(log_file)], check=True)
        self._fh = open(log_file, "a", buffering=1, encoding="utf-8")
        self._header()

    def _header(self):
        sep = "═" * 72
        self._emit(sep)
        self._emit(f"  FL FAILURE ORCHESTRATOR  │  Log opened {utcnow_iso()}")
        self._emit(f"  Log file: {self.log_file}")
        self._emit(sep)

    def _emit(self, text: str):
        print(text)
        self._fh.write(text + "\n")

    def log(self, msg: str, level: str = "INFO"):
        ts   = utcnow_iso()
        icon = self.LEVELS.get(level, "  ")
        self._emit(f"[{ts}] [{level:<7}] {icon}  {msg}")

    def section(self, title: str):
        sep = "─" * 72
        self._emit(sep)
        self._emit(f"  {title}")
        self._emit(sep)

    def blank(self):
        self._emit("")

    def close(self):
        self.log("Logger closing — experiment session ended.", "INFO")
        self._fh.close()


# ══════════════════════════════════════════════════════════════════════════════
#  PLAN GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _distribute_fail_rounds(num_failing: int, max_fail_round: int,
                             rng: random.Random) -> list:
    """
    Spread fail_rounds evenly across [1 .. max_fail_round] using segment
    sampling so failures aren't clustered in one part of the experiment.
    Guarantees no two selected clients fail in the exact same round.
    """
    segment_size = max_fail_round / num_failing
    fail_rounds  = []
    for i in range(num_failing):
        seg_start = max(1, int(i * segment_size) + 1)
        seg_end   = min(max_fail_round, int((i + 1) * segment_size))
        if seg_start > seg_end:
            seg_end = seg_start
        fail_rounds.append(rng.randint(seg_start, seg_end))

    # Shuffle so assignment to specific clients is independent of timeline position
    rng.shuffle(fail_rounds)
    return fail_rounds


def generate_plan(failure_pct: int, seed: int,
                  num_rounds: int = NUM_ROUNDS,
                  num_clients: int = TOTAL_CLIENTS) -> dict:
    """
    Build a fully deterministic, reproducible failure plan.

    Rules:
      - Each failing client has exactly 1 failure episode
      - fail_round ∈ [1, num_rounds - 2]  (reserves ≥ 2 rounds for downtime)
      - absent duration: 2–4 rounds, capped so restart_round ≤ num_rounds
      - Failure events are spread evenly across the experiment timeline
    """
    rng         = random.Random(seed)
    num_failing = math.ceil(num_clients * failure_pct / 100)

    if num_failing > num_clients:
        raise ValueError(f"num_failing={num_failing} exceeds num_clients={num_clients}")

    # Latest a client can fail: leave at least 2 rounds for it to be absent
    max_fail_round   = num_rounds - 2

    failing_clients  = sorted(rng.sample(range(num_clients), num_failing))
    fail_rounds      = _distribute_fail_rounds(num_failing, max_fail_round, rng)

    episodes = {}
    for client_id, fail_round in zip(failing_clients, fail_rounds):
        max_down      = min(4, num_rounds - fail_round)
        down_duration = rng.randint(2, max(2, max_down))
        restart_round = fail_round + down_duration  # guaranteed ≤ num_rounds

        episodes[str(client_id)] = {
            "fail_round":    fail_round,
            "restart_round": restart_round,
            "absent_rounds": list(range(fail_round + 1, restart_round + 1)),
            "down_duration": down_duration,
        }

    # Pre-compute action schedule for easy inspection and orchestrator use
    action_schedule = {}
    for cid_str, ep in episodes.items():
        fr = ep["fail_round"]
        rr = ep["restart_round"]
        action_schedule.setdefault(str(fr), {"stop": [], "start": []})["stop"].append(int(cid_str))
        action_schedule.setdefault(str(rr), {"stop": [], "start": []})["start"].append(int(cid_str))

    plan = {
        "meta": {
            "generated_at":         utcnow_iso(),
            "seed":                 seed,
            "failure_pct":          failure_pct,
            "num_failing_clients":  num_failing,
            "total_clients":        num_clients,
            "num_rounds":           num_rounds,
            "episodes_per_client":  1,
        },
        "failing_clients":  failing_clients,
        "episodes":         episodes,
        "action_schedule":  action_schedule,
        "semantics": {
            "fail_round":    "Client is STOPPED after [SERVER] ROUND N EVALUATION SUMMARY is logged. "
                             "The client has already submitted its update for this round.",
            "restart_round": "Client is STARTED after [SERVER] ROUND N EVALUATION SUMMARY is logged. "
                             "Client rejoins from round (restart_round + 1) onwards.",
            "absent_rounds": "Round numbers where this client contributes NO update to aggregation.",
            "round_indexing": "All round numbers in this plan are 1-indexed. "
                              "The server logs are 0-indexed; the orchestrator adds 1 when parsing.",
        },
    }
    return plan


# ══════════════════════════════════════════════════════════════════════════════
#  PLAN I/O
# ══════════════════════════════════════════════════════════════════════════════

def plan_path(failure_pct: int) -> Path:
    return PLANS_DIR / f"failure_plan_dim4_{failure_pct}.json"


def save_plan(plan: dict, failure_pct: int) -> Path:
    sudo_mkdir(PLANS_DIR)
    p = plan_path(failure_pct)
    p.write_text(json.dumps(plan, indent=2))
    return p


def load_plan(failure_pct: int) -> dict:
    p = plan_path(failure_pct)
    if not p.exists():
        raise FileNotFoundError(
            f"No plan found at {p}\n"
            f"Generate it first:\n"
            f"  python3 {Path(sys.argv[0]).name} generate-plan --failure-pct {failure_pct}"
        )
    return json.loads(p.read_text())


# ══════════════════════════════════════════════════════════════════════════════
#  REMOTE ACTIONS (via gcloud compute ssh)
# ══════════════════════════════════════════════════════════════════════════════

def run_remote(vm_name: str, zone: str, command: str,
               logger: Logger, timeout: int = 90) -> tuple:
    """
    Execute a shell command on a GCP VM via gcloud compute ssh.
    Returns (returncode, combined_output_string).
    """
    gcloud_cmd = [
        "gcloud", "compute", "ssh", vm_name,
        f"--zone={zone}",
        "--quiet",
        "--command", command,
    ]
    logger.log(f"SSH → {vm_name} [{zone}] $ {command}", "DEBUG")

    try:
        result = subprocess.run(
            gcloud_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (result.stdout + result.stderr).strip()
        if output:
            for out_line in output.splitlines():
                logger.log(f"SSH ← {vm_name}: {out_line}", "DEBUG")
        return result.returncode, output

    except subprocess.TimeoutExpired:
        logger.log(f"SSH TIMEOUT after {timeout}s  →  {vm_name}: {command}", "ERROR")
        return -1, "TIMEOUT"
    except Exception as exc:
        logger.log(f"SSH EXCEPTION  →  {vm_name}: {exc}", "ERROR")
        return -1, str(exc)


def stop_client(client_id: int, vm_info: dict, logger: Logger):
    vm_name, zone = get_client_vm(client_id, vm_info)
    logger.log(f"STOP fl-client-{client_id}  on  {vm_name} ({zone})", "ACTION")
    rc, out = run_remote(
        vm_name, zone,
        f"cd /app && sudo docker compose stop fl-client-{client_id}",
        logger,
    )
    if rc == 0:
        logger.log(f"fl-client-{client_id} is DOWN", "SUCCESS")
    else:
        logger.log(
            f"Failed to stop fl-client-{client_id}  rc={rc}  "
            f"output={out!r}  — manual intervention may be needed", "ERROR"
        )


def start_client(client_id: int, vm_info: dict, logger: Logger):
    vm_name, zone = get_client_vm(client_id, vm_info)
    logger.log(f"START fl-client-{client_id}  on  {vm_name} ({zone})", "ACTION")
    rc, out = run_remote(
        vm_name, zone,
        f"cd /app && sudo docker compose up -d fl-client-{client_id}",
        logger,
    )
    if rc == 0:
        logger.log(f"fl-client-{client_id} is UP", "SUCCESS")
    else:
        logger.log(
            f"Failed to start fl-client-{client_id}  rc={rc}  "
            f"output={out!r}  — manual intervention may be needed", "ERROR"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def read_run_id_from_pyproject(path: Path = Path("./pyproject.toml")) -> str:
    """
    Extract run_id from [tool.flwr.app.config] in pyproject.toml and
    format it as EXP_<run_id>.

    Uses stdlib tomllib (Python 3.11+) with a regex fallback for older versions.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"pyproject.toml not found at {path.resolve()}\n"
            "Either place pyproject.toml in the current directory or pass --run-id explicitly."
        )

    # Try stdlib tomllib first (Python 3.11+), then tomli (pip install tomli)
    try:
        try:
            import tomllib
            with open(path, "rb") as fh:
                data = tomllib.load(fh)
        except ImportError:
            import tomli as tomllib  # type: ignore
            with open(path, "rb") as fh:
                data = tomllib.load(fh)

        run_id = data["tool"]["flwr"]["app"]["config"]["run_id"]

    except (ImportError, ModuleNotFoundError):
        # Fallback: plain regex — no toml library available
        text = path.read_text()
        match = re.search(r'^\s*run_id\s*=\s*([^\s#\n]+)', text, re.MULTILINE)
        if not match:
            raise ValueError(
                "Could not find 'run_id' under [tool.flwr.app.config] in pyproject.toml"
            )
        run_id = match.group(1).strip('"').strip("'")

    except KeyError:
        raise KeyError(
            "pyproject.toml is missing 'run_id' under [tool.flwr.app.config]"
        )

    return f"EXP_{run_id}"


def build_action_schedule(plan: dict) -> dict:
    """
    Convert plan episodes into a fast-lookup dict:
      { round_number (1-indexed): { "stop": [client_ids], "start": [client_ids] } }
    """
    schedule = {}
    for cid_str, ep in plan["episodes"].items():
        cid = int(cid_str)
        fr  = ep["fail_round"]
        rr  = ep["restart_round"]
        schedule.setdefault(fr, {"stop": [], "start": []})["stop"].append(cid)
        schedule.setdefault(rr, {"stop": [], "start": []})["start"].append(cid)
    return schedule


def currently_down_clients(plan: dict, after_round: int) -> list:
    """Return list of client_ids that are currently stopped after `after_round`."""
    down = []
    for cid_str, ep in plan["episodes"].items():
        if ep["fail_round"] < after_round <= ep["restart_round"]:
            down.append(int(cid_str))
    return sorted(down)


def run_experiment(failure_pct: int, run_id: str,
                   logger: Logger, vm_info: dict, plan: dict):
    """
    Core orchestration loop.

    Tails the fl-server Docker container logs on the local machine (this script
    runs ON the server VM). Detects round completions via:
      [SERVER] ROUND <N> EVALUATION SUMMARY   (0-indexed)
    and fires stop/start actions from the pre-built schedule.
    """
    num_rounds = plan["meta"]["num_rounds"]
    schedule   = build_action_schedule(plan)

    # ── Print plan summary ───────────────────────────────────────────────────
    logger.section(f"FAILURE PLAN SUMMARY  │  dim4={failure_pct}%  │  {run_id}")
    logger.log(f"Seed              : {plan['meta']['seed']}")
    logger.log(f"Failing clients   : {plan['failing_clients']}")
    logger.log(f"Num failing       : {plan['meta']['num_failing_clients']} / {plan['meta']['total_clients']}")
    logger.log(f"Num rounds        : {num_rounds}")
    logger.blank()

    logger.log("Per-client episode detail:")
    for cid_str, ep in sorted(plan["episodes"].items(), key=lambda x: int(x[0])):
        absent_str = ", ".join(str(r) for r in ep["absent_rounds"])
        logger.log(
            f"  client-{int(cid_str):>2}  │  "
            f"STOP after round {ep['fail_round']:>2}  │  "
            f"START after round {ep['restart_round']:>2}  │  "
            f"absent rounds: [{absent_str}]  │  down for {ep['down_duration']} rounds"
        )

    logger.blank()
    logger.section("FULL ACTION SCHEDULE  (actions fire after EVALUATION SUMMARY)")
    for rnd in sorted(schedule.keys()):
        acts  = schedule[rnd]
        parts = []
        if acts["stop"]:
            parts.append(f"STOP  clients {sorted(acts['stop'])}")
        if acts["start"]:
            parts.append(f"START clients {sorted(acts['start'])}")
        logger.log(f"  After round {rnd:>2} evaluation:  {' | '.join(parts)}")

    logger.blank()
    logger.section("STARTING LOG TAIL  —  watching for [SERVER] ROUND N EVALUATION SUMMARY")
    logger.log(
        "Note: actions fire AFTER evaluation. "
        "Stopped clients have already submitted their round update."
    )
    logger.blank()

    # ── Tail server logs ─────────────────────────────────────────────────────
    # --tail 0 = only lines appearing AFTER this process starts.
    # This is critical if the orchestrator is (re)started mid-experiment —
    # it won't replay old round signals and double-fire actions.
    proc = subprocess.Popen(
        ["sudo", "docker", "logs", "-f", "--tail", "0", SERVER_CONTAINER],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Graceful shutdown on Ctrl-C or SIGTERM
    def _shutdown(signum, frame):
        logger.log(
            f"Signal {signum} received — shutting down orchestrator gracefully.", "WARN"
        )
        proc.terminate()
        raise SystemExit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    completed_rounds: set = set()

    try:
        for raw_line in proc.stdout:
            line = raw_line.rstrip()

            match = ROUND_EVAL_RE.search(line)
            if not match:
                continue

            # Server logs 0-indexed → convert to 1-indexed
            round_0idx = int(match.group(1))
            round_num  = round_0idx + 1

            # Guard against duplicate lines (shouldn't happen but be safe)
            if round_num in completed_rounds:
                logger.log(
                    f"Duplicate EVALUATION SUMMARY signal for round {round_num} — ignoring.",
                    "WARN"
                )
                continue
            completed_rounds.add(round_num)

            logger.blank()
            logger.section(f"ROUND {round_num}/{num_rounds} EVALUATION COMPLETE")

            # ── Fire actions ─────────────────────────────────────────────────
            if round_num in schedule:
                actions = schedule[round_num]

                # Always STOP before START — don't bring up a client
                # on a VM that may still be cleaning up another container
                for cid in sorted(actions["stop"]):
                    stop_client(cid, vm_info, logger)

                for cid in sorted(actions["start"]):
                    start_client(cid, vm_info, logger)
            else:
                logger.log(f"No actions scheduled after round {round_num}.")

            # ── State snapshot ───────────────────────────────────────────────
            down_now   = currently_down_clients(plan, round_num)
            remaining  = num_rounds - round_num
            logger.blank()
            logger.log(f"── Cluster state after round {round_num} actions ──")
            logger.log(f"   Rounds completed   : {round_num}/{num_rounds}  ({remaining} remaining)")
            logger.log(f"   Clients currently DOWN : {down_now if down_now else 'none (all up)'}")
            logger.log(f"   Clients currently UP   : {sorted(set(range(TOTAL_CLIENTS)) - set(down_now))}")
            logger.blank()

            # ── Check termination ────────────────────────────────────────────
            if round_num >= num_rounds:
                logger.section("ALL ROUNDS COMPLETED")
                logger.log(
                    f"Experiment {run_id} finished after {num_rounds} rounds. "
                    "Orchestrator exiting cleanly.", "SUCCESS"
                )
                # Sanity check: make sure no clients are still stopped
                if down_now:
                    logger.log(
                        f"WARNING: clients {down_now} are still stopped at experiment end. "
                        "Starting them now for clean state.", "WARN"
                    )
                    for cid in down_now:
                        start_client(cid, vm_info, logger)
                break

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# ══════════════════════════════════════════════════════════════════════════════
#  CLI COMMAND IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_generate_plan(args):
    out_path = plan_path(args.failure_pct)

    if out_path.exists() and not args.overwrite:
        print(f"\n⚠️  Plan already exists: {out_path}")
        print("Pass --overwrite to regenerate it (WARNING: this breaks reproducibility\n"
              "if experiments with this failure-pct have already been run).\n")
        print("Existing plan:")
        existing = json.loads(out_path.read_text())
        print(json.dumps(existing, indent=2))
        return

    # Use provided seed or generate a random one and store it in the plan
    seed = args.seed if args.seed is not None else random.randint(10_000, 99_999)

    plan = generate_plan(
        failure_pct=args.failure_pct,
        seed=seed,
        num_rounds=args.num_rounds,
        num_clients=args.num_clients,
    )

    saved_path = save_plan(plan, args.failure_pct)
    print(f"\n✅ Plan saved → {saved_path}")
    print(f"   Seed: {seed}  (recorded inside the plan JSON for full reproducibility)\n")
    print(json.dumps(plan, indent=2))


def cmd_show_plan(args):
    try:
        plan = load_plan(args.failure_pct)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(json.dumps(plan, indent=2))

    # Pretty-print the schedule separately for quick human scanning
    print("\n── Readable Action Schedule ──")
    schedule = build_action_schedule(plan)
    for rnd in sorted(schedule.keys()):
        acts  = schedule[rnd]
        parts = []
        if acts["stop"]:
            parts.append(f"STOP  {sorted(acts['stop'])}")
        if acts["start"]:
            parts.append(f"START {sorted(acts['start'])}")
        print(f"  After round {rnd:>2} EVALUATION:  {' | '.join(parts)}")

    print("\n── Per-client absence summary ──")
    for cid_str, ep in sorted(plan["episodes"].items(), key=lambda x: int(x[0])):
        absent_str = ", ".join(str(r) for r in ep["absent_rounds"])
        print(
            f"  client-{int(cid_str):>2}  │  "
            f"fail after round {ep['fail_round']:>2}  │  "
            f"restart after round {ep['restart_round']:>2}  │  "
            f"absent: [{absent_str}]"
        )


def cmd_run(args):
    # Resolve run_id — from CLI flag or pyproject.toml
    if args.run_id:
        run_id = args.run_id
    else:
        try:
            run_id = read_run_id_from_pyproject()
            print(f"\u2139\ufe0f  run_id not specified \u2014 read from pyproject.toml: {run_id}")
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"ERROR: Could not resolve run_id: {exc}")
            sys.exit(1)

    args.run_id = run_id

    # Load VM info
    try:
        vm_info = load_vm_info()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    # Load failure plan
    try:
        plan = load_plan(args.failure_pct)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    # Validate plan num_rounds matches expectation
    plan_rounds = plan["meta"]["num_rounds"]
    if plan_rounds != args.num_rounds:
        print(
            f"WARNING: plan was generated for {plan_rounds} rounds, "
            f"but --num-rounds={args.num_rounds}. "
            "Proceeding with plan's round count."
        )

    # Set up log file
    ts       = utcnow_compact()
    log_name = f"{args.run_id}_dim4_{args.failure_pct}pct_{ts}.log"
    log_path = LOGS_DIR / log_name
    logger   = Logger(log_path)

    logger.section("EXPERIMENT SESSION START")
    logger.log(f"Run ID          : {args.run_id}")
    logger.log(f"Failure pct     : {args.failure_pct}%  (dim_4)")
    logger.log(f"Plan seed       : {plan['meta']['seed']}")
    logger.log(f"Plan generated  : {plan['meta']['generated_at']}")
    logger.log(f"Num rounds      : {plan['meta']['num_rounds']}")
    logger.log(f"Server container: {SERVER_CONTAINER}")
    logger.log(f"VM info file    : {VM_INFO_FILE.resolve()}")
    logger.log(f"Log file        : {log_path.resolve()}")
    logger.blank()

    try:
        run_experiment(
            failure_pct=args.failure_pct,
            run_id=args.run_id,
            logger=logger,
            vm_info=vm_info,
            plan=plan,
        )
    except Exception as exc:
        logger.log(f"FATAL EXCEPTION: {exc}", "ERROR")
        import traceback
        tb = traceback.format_exc()
        for tb_line in tb.splitlines():
            logger.log(tb_line, "ERROR")
        raise
    finally:
        logger.close()

    print(f"\n✅ Orchestrator finished. Log saved to: {log_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fl_failure_orchestrator.py",
        description="FL Failure Orchestrator — plan generation and live experiment injection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES
  # Generate plans (do this once before running any experiments)
  python3 fl_failure_orchestrator.py generate-plan --failure-pct 25
  python3 fl_failure_orchestrator.py generate-plan --failure-pct 50
  python3 fl_failure_orchestrator.py generate-plan --failure-pct 75

  # Inspect a plan
  python3 fl_failure_orchestrator.py show-plan --failure-pct 25

  # Run orchestrator during an experiment (run on server VM)
  python3 fl_failure_orchestrator.py run --failure-pct 25 --run-id EXP_11213141
  python3 fl_failure_orchestrator.py run --failure-pct 50 --run-id EXP_12213141
        """,
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ── generate-plan ────────────────────────────────────────────────────────
    p_gen = sub.add_parser(
        "generate-plan",
        help="Generate and save a failure plan for a given failure-pct",
    )
    p_gen.add_argument(
        "--failure-pct", type=int, required=True, choices=[25, 50, 75],
        help="Percentage of clients that fail (25, 50, or 75)",
    )
    p_gen.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: auto-generated and saved in the plan)",
    )
    p_gen.add_argument(
        "--num-rounds", type=int, default=NUM_ROUNDS,
        help=f"Number of rounds in the experiment (default: {NUM_ROUNDS})",
    )
    p_gen.add_argument(
        "--num-clients", type=int, default=TOTAL_CLIENTS,
        help=f"Total number of clients (default: {TOTAL_CLIENTS})",
    )
    p_gen.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing plan (WARNING: breaks reproducibility mid-experiment-set)",
    )
    p_gen.set_defaults(func=cmd_generate_plan)

    # ── show-plan ────────────────────────────────────────────────────────────
    p_show = sub.add_parser(
        "show-plan",
        help="Print an existing failure plan in human-readable form",
    )
    p_show.add_argument(
        "--failure-pct", type=int, required=True, choices=[25, 50, 75],
    )
    p_show.set_defaults(func=cmd_show_plan)

    # ── run ──────────────────────────────────────────────────────────────────
    p_run = sub.add_parser(
        "run",
        help="Tail fl-server logs and inject client failures according to the saved plan",
    )
    p_run.add_argument(
        "--failure-pct", type=int, required=True, choices=[25, 50, 75],
        help="Failure % dimension — must match a generated plan",
    )
    p_run.add_argument(
        "--run-id", type=str, default=None,
        help=(
            "Experiment run ID, e.g. EXP_11213141 (used in log filename). "
            "If omitted, extracted automatically from pyproject.toml "
            "[tool.flwr.app.config] run_id and formatted as EXP_<run_id>."
        ),
    )
    p_run.add_argument(
        "--num-rounds", type=int, default=NUM_ROUNDS,
        help=f"Expected number of rounds (default: {NUM_ROUNDS})",
    )
    p_run.set_defaults(func=cmd_run)

    return parser


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)