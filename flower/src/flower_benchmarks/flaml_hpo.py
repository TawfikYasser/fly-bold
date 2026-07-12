"""
FLAML HPO (Hyperparameter Optimization) integration for Federated Learning.
Provides an Optuna-compatible interface for FLAML's tune optimization.

Warm-start
----------
Pass `points_to_evaluate` + `evaluated_rewards` to FLAMLStudy / create_flaml_study
so BlendSearch seeds its local search from known-good configs rather than
exploring the full space cold.

Official FLAML docs behaviour:
  - points_to_evaluate : list of config dicts → always tried first (in order)
  - evaluated_rewards  : list of known scores → FLAML skips re-evaluating those configs;
                         can be SHORTER than points_to_evaluate (FLAML will re-run the
                         remaining un-scored configs while still using them as warm hints)
  - Pass None (not []) for either when there is no history; empty lists cause a
    ValueError in some BlendSearch versions.

Usage in server_app.py (mirrors Optuna's best_prev_params pattern):

    from flower_benchmarks.flaml_hpo import create_flaml_study, PREVIOUS_EXPERIMENTS

    prev = PREVIOUS_EXPERIMENTS.get(dataset_number)
    study = create_flaml_study(
        study_name=f"{experiment_name}_{run_id}_hpo",
        points_to_evaluate=prev["configs"] if prev else None,
        evaluated_rewards=prev["scores"]  if prev else None,
    )
    study.optimize(objective_flaml, n_trials=hpo_trials)
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from flaml import tune
    HAS_FLAML = True
except ImportError:
    HAS_FLAML = False

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Previous experiment registry
# Mirror of Optuna's best_prev_params — add entries as you get better results.
# Keys are dataset_number values set in the Flower run config.
#
# "configs" : list of full config dicts (all keys that exist in search_space)
# "scores"  : list of mAP@0.5 values, same order as configs
#             CAN be shorter than configs — FLAML will re-run the un-scored ones
#             but still use them as warm hints for local search direction.
# ─────────────────────────────────────────────────────────────────────────────
PREVIOUS_EXPERIMENTS: Dict[int, Dict[str, List]] = {
    # ── Non-IID  (dataset_number = 100) ─────────────────────────────────────
    # Source: Optuna study.add_trial() block (if dataset_number == 100)
    # Ordered best-first so BlendSearch seeds local search from the strongest prior.
    100: {
        "configs": [
            # FedAvg  ×4  (same config, different scores across runs)
            {"lr": 0.001, "local_epochs": 3, "batch_size": 16, "strategy": 1},
            {"lr": 0.001, "local_epochs": 3, "batch_size": 16, "strategy": 1},
            {"lr": 0.001, "local_epochs": 3, "batch_size": 16, "strategy": 1},
            {"lr": 0.001, "local_epochs": 3, "batch_size": 16, "strategy": 1},
            # FedAdam
            {
                "lr": 0.000561, "local_epochs": 5, "batch_size": 8, "strategy": 3,
                "adam_eta": 0.0814829,  "adam_eta_l": 0.00113647,
                "adam_beta_1": 0.984283, "adam_beta_2": 0.982412, "adam_tau": 0.000265875,
            },
            # FedYogi  (run 3)
            {
                "lr": 0.000231, "local_epochs": 1, "batch_size": 16, "strategy": 2,
                "yogi_eta": 0.00125628, "yogi_eta_l": 0.00816846,
                "yogi_beta_1": 0.949183, "yogi_beta_2": 0.919768, "yogi_tau": 0.00106775,
            },
            # FedYogi  (run 4  — pruned; no score → FLAML will re-evaluate it)
            {
                "lr": 0.001530, "local_epochs": 1, "batch_size": 8, "strategy": 2,
                "yogi_eta": 0.000196343, "yogi_eta_l": 0.0233596,
                "yogi_beta_1": 0.883629,  "yogi_beta_2": 0.912082, "yogi_tau": 0.000978034,
            },
            # FedProx
            {
                "lr": 0.000117, "local_epochs": 5, "batch_size": 16, "strategy": 4,
                "proximal_mu": 1.2604664585649468,
            },
        ],
        # evaluated_rewards is SHORTER than configs by design:
        # the pruned FedYogi trial (index 6) has no valid score, so we stop at 6.
        # FLAML will skip re-running indices 0-5 and actually execute index 6 & 7,
        # while still using all 8 configs as warm hints for BlendSearch.
        "scores": [0.5218, 0.5181, 0.5162, 0.5048, 0.5001, 0.5001],
    },

    # ── IID  (dataset_number = 0 / "000") ───────────────────────────────────
    # Source: Optuna study.add_trial() block (else branch)
    0: {
        "configs": [
            # FedAvg  ×3  (same config, different scores across runs)
            {"lr": 0.001, "local_epochs": 3, "batch_size": 16, "strategy": 1},
            {"lr": 0.001, "local_epochs": 3, "batch_size": 16, "strategy": 1},
            {"lr": 0.001, "local_epochs": 3, "batch_size": 16, "strategy": 1},
        ],
        "scores": [0.5309, 0.5211, 0.5200],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# History persistence helpers  (optional — use when you want runs to persist
# across experiments automatically instead of editing PREVIOUS_EXPERIMENTS)
# ─────────────────────────────────────────────────────────────────────────────

def load_flaml_history(path: str) -> Tuple[Optional[List], Optional[List]]:
    """Load (configs, scores) from a JSON history file.

    Returns (None, None) when the file does not exist so you can pass directly
    to create_flaml_study without extra guards.
    """
    p = Path(path)
    if not p.exists():
        logger.info(f"[FLAML] No history at {path} — cold start.")
        return None, None
    try:
        data = json.loads(p.read_text())
        configs = data.get("configs", [])
        scores  = data.get("scores",  [])
        if not configs:
            return None, None
        # scores can be shorter than configs per FLAML docs — that is valid
        if len(scores) > len(configs):
            logger.warning(
                f"[FLAML] History {path}: more scores than configs — truncating scores."
            )
            scores = scores[: len(configs)]
        logger.info(
            f"[FLAML] Loaded {len(configs)} config(s) / {len(scores)} score(s) from {path}."
        )
        return configs or None, scores or None
    except Exception as exc:
        logger.warning(f"[FLAML] Could not load history from {path}: {exc}")
        return None, None


def save_flaml_history(
    path: str,
    configs: List[Dict[str, Any]],
    scores:  List[float],
    new_config: Optional[Dict[str, Any]] = None,
    new_score:  Optional[float]          = None,
) -> Tuple[List, List]:
    """Append a result and persist the history file.

    Returns updated (configs, scores).
    """
    configs = list(configs or [])
    scores  = list(scores  or [])
    if new_config is not None and new_score is not None:
        configs.append(new_config)
        scores.append(new_score)
    try:
        Path(path).write_text(json.dumps({"configs": configs, "scores": scores}, indent=2))
        logger.info(f"[FLAML] Saved {len(configs)} trial(s) to {path}.")
    except Exception as exc:
        logger.warning(f"[FLAML] Could not save history to {path}: {exc}")
    return configs, scores


# ─────────────────────────────────────────────────────────────────────────────
# Trial / Study wrappers
# ─────────────────────────────────────────────────────────────────────────────

class TrialPruneSignal(Exception):
    """Raised when FLAML decides to prune a trial (early stopping)."""
    pass


class FLAMLTrial:
    """Wrapper to provide Optuna-like interface for FLAML trials."""

    def __init__(self, trial_id: int, config: Dict[str, Any]):
        self.number = trial_id
        self.id     = trial_id
        self.config = config
        self.intermediate_values: Dict[int, float] = {}
        self.should_prune_flag = False
        self.value: Optional[float] = None
        self.user_attrs: Dict[str, Any] = {}

    def set_user_attr(self, key: str, value: Any) -> None:
        self.user_attrs[key] = value

    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
        if name in self.config:
            return float(self.config[name])
        raise KeyError(f"Parameter '{name}' not found in FLAML config")

    def suggest_int(self, name: str, low: int, high: int) -> int:
        if name in self.config:
            return int(self.config[name])
        raise KeyError(f"Parameter '{name}' not found in FLAML config")

    def suggest_categorical(self, name: str, choices: list) -> Any:
        if name in self.config:
            return self.config[name]
        raise KeyError(f"Parameter '{name}' not found in FLAML config")

    def report(self, value: float, step: int) -> None:
        """Record intermediate score (pruner hook — not forwarded to tune.report)."""
        self.intermediate_values[step] = value

    def should_prune(self) -> bool:
        return self.should_prune_flag


class FLAMLStudy:
    """Optuna-compatible wrapper around flaml.tune.run.

    Warm-start parameters
    ─────────────────────
    points_to_evaluate : list of config dicts from previous runs.
        BlendSearch tries these first and seeds its local search around them.

    evaluated_rewards : list of float scores (same order as points_to_evaluate).
        FLAML skips re-evaluating those configs.
        Per official docs: can be SHORTER than points_to_evaluate — FLAML will
        re-run the remaining un-scored configs while still using them as hints.

    Pass None (not []) when there is no prior history.
    """

    def __init__(
        self,
        study_name: str,
        direction:           str                            = "maximize",
        time_budget:         int                            = 3600,
        metric_name:         str                            = "mAP50",
        seed:                int                            = 42,
        points_to_evaluate:  Optional[List[Dict[str, Any]]] = None,
        evaluated_rewards:   Optional[List[float]]           = None,
        search_space:        Optional[Dict[str, Any]]        = None,
    ):
        self.study_name  = study_name
        self.direction   = direction
        self.time_budget = time_budget
        self.metric_name = metric_name
        self.seed        = seed
        self.search_space = search_space  # None -> optimize() falls back to lr-only default
        self.best_trial  = None
        self.best_params: Dict[str, Any] = {}
        self.best_value  = float("-inf") if direction == "maximize" else float("inf")
        self.trials:      List[FLAMLTrial] = []

        # ── Validate & store warm-start data ──────────────────────────────────
        if points_to_evaluate:
            if evaluated_rewards and len(evaluated_rewards) > len(points_to_evaluate):
                raise ValueError(
                    f"evaluated_rewards ({len(evaluated_rewards)}) is longer than "
                    f"points_to_evaluate ({len(points_to_evaluate)}). "
                    "Per FLAML docs it must be the same length or shorter."
                )
            self._points_to_evaluate = points_to_evaluate
            self._evaluated_rewards  = evaluated_rewards if evaluated_rewards else None

            n_pts    = len(points_to_evaluate)
            n_scores = len(evaluated_rewards) if evaluated_rewards else 0
            best_known = (
                max(evaluated_rewards) if direction == "maximize" and evaluated_rewards
                else min(evaluated_rewards) if evaluated_rewards
                else "N/A"
            )
            logger.info(
                f"[FLAML] Warm-start: {n_pts} hint config(s), "
                f"{n_scores} known score(s), best known = {best_known}"
            )
        else:
            self._points_to_evaluate = None
            self._evaluated_rewards  = None
            logger.info("[FLAML] No warm-start data — cold start.")

    def optimize(self, objective_func: Callable[[FLAMLTrial], float], n_trials: int) -> None:
        """Run HPO using flaml.tune.run."""
        logger.info(f"[FLAML] Starting study: {self.study_name}")

        # ── Search space ────────────────────────────────────────────────────
        # NESTED HPO MODE: the server only searches `lr`. `local_epochs` and
        # `batch_size` are NOT server dimensions anymore -- each client runs
        # its own inner search over those two (see client_app.py
        # run_client_local_hpo / run_client_local_hpo_flaml), conditioned on
        # whatever `lr` this server trial suggests. `strategy` and its
        # meta-params (yogi_*/adam_*/proximal_mu) are fixed for the whole run
        # (read from pyproject.toml / _env), not searched, so trials are
        # comparable on `lr` alone.
        search_space = self.search_space or {
            "lr": tune.loguniform(lower=0.0001, upper=0.01),
        }

        bad_score = float("inf") if self.direction == "minimize" else float("-inf")

        def _flaml_objective(config: Dict[str, Any]) -> Dict[str, Any]:
            trial_id = len(self.trials)
            trial    = FLAMLTrial(trial_id, config)
            self.trials.append(trial)
            try:
                score       = objective_func(trial)
                trial.value = score
                is_better   = (
                    score > self.best_value if self.direction == "maximize"
                    else score < self.best_value
                )
                if is_better:
                    self.best_value  = score
                    self.best_params = config
                    self.best_trial  = trial
                return {self.metric_name: score, "status": "completed"}
            except TrialPruneSignal:
                logger.info(f"[FLAML] Trial {trial_id} pruned.")
                return {self.metric_name: bad_score, "status": "pruned"}
            except Exception as exc:
                logger.error(f"[FLAML] Trial {trial_id} failed: {exc}", exc_info=True)
                return {self.metric_name: bad_score, "status": "failed"}

        # ── flaml.tune.run ────────────────────────────────────────────────────
        # points_to_evaluate : tried first in order; BlendSearch seeds local
        #                      search around them
        # evaluated_rewards  : known scores → FLAML skips re-running those configs
        #                      (can be shorter than points_to_evaluate per docs)
        try:
            analysis = tune.run(
                _flaml_objective,
                config=search_space,
                metric=self.metric_name,
                mode="max" if self.direction == "maximize" else "min",
                num_samples=n_trials,
                time_budget_s=self.time_budget,
                use_ray=False,
                points_to_evaluate=self._points_to_evaluate,
                evaluated_rewards=self._evaluated_rewards,
            )
            # Sync best from analysis (may differ if FLAML found better in warm-start phase)
            if analysis and analysis.best_config:
                best_from_analysis = analysis.best_result.get(self.metric_name)
                is_better = (
                    best_from_analysis is not None and (
                        best_from_analysis > self.best_value if self.direction == "maximize"
                        else best_from_analysis < self.best_value
                    )
                )
                if is_better:
                    self.best_params = analysis.best_config
                    self.best_value  = best_from_analysis
        except Exception as exc:
            logger.error(f"[FLAML] tune.run failed: {exc}", exc_info=True)


# ─────────────────────────────────────────────────────────────────────────────
# Public factory — call this from server_app.py
# ─────────────────────────────────────────────────────────────────────────────

def create_flaml_study(
    study_name: str,
    direction:           str                            = "maximize",
    time_budget:         int                            = 3600,
    metric_name:         str                            = "mAP50",
    seed:                int                            = 42,
    # ── Warm-start: Option A — explicit lists (e.g. from PREVIOUS_EXPERIMENTS) ──
    points_to_evaluate:  Optional[List[Dict[str, Any]]] = None,
    evaluated_rewards:   Optional[List[float]]           = None,
    # ── Warm-start: Option B — load from a persisted JSON history file ─────────
    # If provided, this overrides the explicit lists above.
    history_path:        Optional[str]                  = None,
    search_space:        Optional[Dict[str, Any]]        = None,
    **kwargs,
) -> FLAMLStudy:
    """Create and return a FLAMLStudy ready to run.

    Warm-start examples
    ───────────────────
    # Option A — use the hardcoded registry (recommended for Flybold):
    from flower_benchmarks.flaml_hpo import create_flaml_study, PREVIOUS_EXPERIMENTS
    prev = PREVIOUS_EXPERIMENTS.get(dataset_number)
    study = create_flaml_study(
        study_name="...",
        points_to_evaluate=prev["configs"] if prev else None,
        evaluated_rewards =prev["scores"]  if prev else None,
    )

    # Option B — load from an auto-saved file:
    study = create_flaml_study(
        study_name="...",
        history_path=f"flaml_history_{dataset_number}.json",
    )
    """
    if not HAS_FLAML:
        raise ImportError(
            "FLAML is not installed. Run: pip install 'flaml[blendsearch]'"
        )

    # history_path takes priority over explicit lists
    if history_path:
        points_to_evaluate, evaluated_rewards = load_flaml_history(history_path)

    # Guard: pass None rather than [] — empty list → BlendSearch ValueError
    if not points_to_evaluate:
        points_to_evaluate = None
        evaluated_rewards  = None
    if not evaluated_rewards:
        evaluated_rewards = None

    return FLAMLStudy(
        study_name=study_name,
        direction=direction,
        time_budget=time_budget,
        metric_name=metric_name,
        seed=seed,
        points_to_evaluate=points_to_evaluate,
        evaluated_rewards=evaluated_rewards,
        search_space=search_space,
    )


def lr_only_warm_start(dataset_number: int) -> Tuple[Optional[List[Dict]], Optional[List[float]]]:
    """Reduce a PREVIOUS_EXPERIMENTS entry (full lr+epochs+batch+strategy+meta
    configs) down to just the `lr` dimension, for use as warm-start points in
    the nested-HPO mode where the server only searches `lr`.

    Duplicate lr values collapse to a single point using their best (for
    "maximize") associated score, so BlendSearch/TPE doesn't see the same
    lr suggested twice with conflicting rewards.
    """
    prev = PREVIOUS_EXPERIMENTS.get(dataset_number)
    if not prev:
        return None, None
    by_lr: Dict[float, float] = {}
    for cfg, score in zip(prev["configs"], prev["scores"]):
        lr = cfg["lr"]
        if lr not in by_lr or score > by_lr[lr]:
            by_lr[lr] = score
    if not by_lr:
        return None, None
    lrs = sorted(by_lr, key=lambda k: -by_lr[k])  # best-first
    return [{"lr": lr} for lr in lrs], [by_lr[lr] for lr in lrs]


def get_flaml_config() -> Dict[str, Any]:
    """Get FLAML configuration from environment variables."""
    return {
        "use_flaml":    os.getenv("USE_FLAML",         "false").lower() in ("true", "1"),
        "time_budget":  int(os.getenv("FLAML_TIME_BUDGET", "3600")),
        "metric":       os.getenv("FLAML_METRIC",      "mAP50"),
        "estimator":    os.getenv("FLAML_ESTIMATOR",   "lgb"),
    }