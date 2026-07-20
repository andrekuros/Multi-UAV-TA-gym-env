"""
WPS_escort full comparison: Coalition-Hungarian, CBBA, PI, Urgency/MLP/Att, Global.

Usage:
  python experiments/escort_eval.py --episodes 100
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from TaskAllocation.Hybrid.AttentionEscort import AttentionEscort, UrgencyCoalition
from TaskAllocation.MarketBased.CBBA_Replan import CBBAReplan
from TaskAllocation.MarketBased.PerformanceImpact import PerformanceImpact
from TaskAllocation.OptimizationBased.HungarianAllocator import HungarianAllocator
from experiments.paper_eval import make_config, _open_tasks, _events
from experiments.paper_scenarios import CASE_SPECS, WPS_ENV_FLAGS
from mUAV_TA.DroneEnv import MultiUAVEnv

RESULTS = os.path.join(ROOT, "experiments", "results")

REPLAN_EVENTS = (
    "Reset_Allocation",
    "New_Threat",
    "Agent_Fail",
    "Escort_Created",
    "Escort_Retired",
)

ALGOS = [
    "Coalition-Hungarian",
    "Local-CBBA-Coalition",
    "Local-PI-Coalition",
    "Urgency-Coalition",
    "MLP-Coalition",
    "Att-Coalition",
    "Global-Coalition",
]


def _should_replan(env, events, interval: int) -> bool:
    if env.time_steps == 0 or env.time_steps % interval == 0:
        return True
    return any(
        (ev[0] if isinstance(ev, (list, tuple)) else ev) in REPLAN_EVENTS
        for ev in events
    )


def _flatten_pairs(result) -> List[Tuple[str, object]]:
    """Normalize CBBA/PI (name, [tasks]) and Hungarian (name, task) to flat pairs."""
    pairs = []
    for item in result or []:
        if not item:
            continue
        name, payload = item[0], item[1]
        if isinstance(payload, list):
            for task in payload:
                pairs.append((name, task))
        else:
            pairs.append((name, payload))
    return pairs


def _apply_assign(env, pairs) -> dict:
    actions = {}
    for agent_name, task in _flatten_pairs(pairs):
        if env.last_tasks_info and task in env.last_tasks_info:
            # First assignment wins for this step (one task per agent)
            if agent_name not in actions:
                actions[agent_name] = env.last_tasks_info.index(task)
    return actions


def run_escort_episode(
    algorithm: str,
    case_id: str,
    seed: int,
    replan_interval: int = 12,
    att: Optional[AttentionEscort] = None,
    mlp: Optional[AttentionEscort] = None,
    urg: Optional[UrgencyCoalition] = None,
) -> Dict[str, float]:
    spec = CASE_SPECS[case_id]
    flags = dict(WPS_ENV_FLAGS)
    flags["capability_mask"] = False
    flags["saturate_mask"] = False
    cfg = make_config(spec, flags)
    cfg.multiple_tasks_per_agent = True
    env = MultiUAVEnv(cfg)
    observation, info = env.reset(seed=seed)
    done = {a: False for a in env.agents}
    trunc = {a: False for a in env.agents}

    hung = HungarianAllocator(replan_interval=replan_interval, max_coord=env.max_coord)
    hung_force = HungarianAllocator(replan_interval=10**9, max_coord=env.max_coord)
    cbba = CBBAReplan(
        env.agents_obj, env.tasks, env.max_coord, seed=seed, replan_interval=replan_interval
    )
    pi = PerformanceImpact(
        max_coord=env.max_coord, seed=seed, replan_interval=replan_interval
    )

    n_replans = 0
    replan_ms: List[float] = []
    latest: Dict[str, Any] = {}

    while not all(done.values()) and not all(trunc.values()):
        events = _events(info)
        actions = {}
        did_replan = False
        t0 = time.perf_counter()

        if algorithm == "Global-Coalition":
            result = hung.allocate_tasks(
                env.get_live_agents(),
                _open_tasks(env),
                time_step=env.time_steps,
                events=events,
                force=False,
            )
            if result or hung.n_replans != n_replans:
                did_replan = hung.n_replans > n_replans
                n_replans = hung.n_replans
            actions = _apply_assign(env, result)
        elif algorithm == "Coalition-Hungarian":
            result = hung.allocate_tasks(
                env.get_live_agents(),
                _open_tasks(env),
                time_step=env.time_steps,
                events=events,
                agent_known_ids=env.agent_visibility_map(),
            )
            if hung.n_replans > n_replans:
                did_replan = True
                n_replans = hung.n_replans
            actions = _apply_assign(env, result)
        elif algorithm == "Local-CBBA-Coalition":
            result = cbba.allocate_tasks(
                env.get_live_agents(),
                _open_tasks(env),
                time_step=env.time_steps,
                events=events,
                agent_known_ids=env.agent_visibility_map(),
                max_tasks_per_agent=1,
            )
            if cbba.n_replans > n_replans:
                did_replan = True
                n_replans = cbba.n_replans
            actions = _apply_assign(env, result)
        elif algorithm == "Local-PI-Coalition":
            result = pi.allocate_tasks(
                env.get_live_agents(),
                _open_tasks(env),
                time_step=env.time_steps,
                events=events,
                agent_known_ids=env.agent_visibility_map(),
                max_tasks_per_agent=1,
            )
            if pi.n_replans > n_replans:
                did_replan = True
                n_replans = pi.n_replans
            actions = _apply_assign(env, result)
        elif algorithm == "Urgency-Coalition" and urg is not None:
            if _should_replan(env, events, replan_interval):
                result = urg.plan(env, hung_force, events=events, force=True)
                did_replan = True
                n_replans = urg.n_replans
                actions = _apply_assign(env, result)
        elif algorithm == "MLP-Coalition" and mlp is not None:
            if _should_replan(env, events, replan_interval):
                result, *_ = mlp.plan(
                    env, hung_force, events=events, explore=False, force=True
                )
                did_replan = True
                n_replans = mlp.n_replans
                actions = _apply_assign(env, result)
        elif algorithm == "Att-Coalition" and att is not None:
            if _should_replan(env, events, replan_interval):
                result, *_ = att.plan(
                    env, hung_force, events=events, explore=False, force=True
                )
                did_replan = True
                n_replans = att.n_replans
                actions = _apply_assign(env, result)
        else:
            raise ValueError(f"Unknown or unavailable algorithm: {algorithm}")

        dt_ms = (time.perf_counter() - t0) * 1000.0
        if did_replan:
            replan_ms.append(dt_ms)

        observation, reward, done, trunc, info = env.step(actions)
        if isinstance(info, dict) and "metrics" in info:
            latest = info["metrics"]

    out = {
        "algorithm": algorithm,
        "case": case_id,
        "seed": seed,
        "S_ESC": float(latest.get("S_ESC", env.compute_s_esc())),
        "S_WPS": float(latest.get("S_WPS", env.compute_s_wps())),
        "escort_coverage_rate": float(latest.get("escort_coverage_rate", 0.0)),
        "protected_rec_completed": int(latest.get("protected_rec_completed", 0)),
        "recon_losses": int(latest.get("recon_losses", 0)),
        "escort_losses": int(latest.get("escort_losses", 0)),
        "threats_intercepted": int(latest.get("threats_intercepted", 0)),
        "mutual_support_engagements": int(latest.get("mutual_support_engagements", 0)),
        "n_missed_windows": int(latest.get("n_missed_windows", 0)),
        "on_time_rate": float(latest.get("on_time_rate", 0.0)),
        "n_task_switches": int(latest.get("n_task_switches", 0)),
        "n_replans": int(n_replans),
        "replan_ms_mean": float(np.mean(replan_ms) if replan_ms else 0.0),
    }
    return out


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 2000, alpha: float = 0.05):
    rng = np.random.RandomState(0)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        means.append(float(np.mean(sample)))
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(np.mean(diffs)), lo, hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="WPS_escort")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--replan-interval", type=int, default=12)
    parser.add_argument(
        "--att-ckpt",
        default=os.path.join(ROOT, "dqn_Custom", "policy_AttCoalition_WPS_escort_v2.pth"),
    )
    parser.add_argument(
        "--mlp-ckpt",
        default=os.path.join(ROOT, "dqn_Custom", "policy_MLPCoalition_WPS_escort_v2.pth"),
    )
    parser.add_argument(
        "--tag",
        default="n100_seed0",
        help="Suffix for output files to avoid overwriting prior runs.",
    )
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)

    if not os.path.isfile(args.att_ckpt):
        raise FileNotFoundError(f"Missing Att checkpoint: {args.att_ckpt}")
    if not os.path.isfile(args.mlp_ckpt):
        raise FileNotFoundError(f"Missing MLP checkpoint: {args.mlp_ckpt}")

    att = AttentionEscort(use_attention=True)
    att.load(args.att_ckpt)
    att.eps = 0.0
    mlp = AttentionEscort(use_attention=False)
    mlp.load(args.mlp_ckpt)
    mlp.eps = 0.0
    urg = UrgencyCoalition()

    rows: List[Dict[str, float]] = []
    for ep in range(args.episodes):
        seed = args.seed0 + ep
        for algo in ALGOS:
            row = run_escort_episode(
                algo,
                args.case,
                seed,
                replan_interval=args.replan_interval,
                att=att,
                mlp=mlp,
                urg=urg,
            )
            rows.append(row)
            print(
                f"ep={ep} seed={seed} {algo}: S_ESC={row['S_ESC']:.1f} "
                f"cov={row['escort_coverage_rate']:.2f} recon_loss={row['recon_losses']} "
                f"replans={row['n_replans']} ms={row['replan_ms_mean']:.1f}",
                flush=True,
            )

    # Completeness check
    by_algo_seeds: Dict[str, set] = {}
    for r in rows:
        by_algo_seeds.setdefault(r["algorithm"], set()).add(int(r["seed"]))
    expected = set(range(args.seed0, args.seed0 + args.episodes))
    for algo in ALGOS:
        got = by_algo_seeds.get(algo, set())
        if got != expected:
            raise RuntimeError(
                f"Incomplete run for {algo}: got {len(got)} seeds, expected {len(expected)}"
            )

    tag = args.tag
    csv_path = os.path.join(RESULTS, f"{args.case}_escort_eval_{tag}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_algo: Dict[str, List[float]] = {}
    by_cov: Dict[str, List[float]] = {}
    by_ms: Dict[str, List[float]] = {}
    for r in rows:
        by_algo.setdefault(r["algorithm"], []).append(float(r["S_ESC"]))
        by_cov.setdefault(r["algorithm"], []).append(float(r["escort_coverage_rate"]))
        by_ms.setdefault(r["algorithm"], []).append(float(r["replan_ms_mean"]))

    summary = {
        algo: {
            "mean_S_ESC": float(np.mean(by_algo[algo])),
            "std_S_ESC": float(np.std(by_algo[algo])),
            "mean_coverage": float(np.mean(by_cov[algo])),
            "mean_replan_ms": float(np.mean(by_ms[algo])),
            "n": len(by_algo[algo]),
        }
        for algo in ALGOS
    }

    att_rows = {r["seed"]: r for r in rows if r["algorithm"] == "Att-Coalition"}
    comparisons = {}
    for other in (
        "Coalition-Hungarian",
        "Local-CBBA-Coalition",
        "Local-PI-Coalition",
        "Urgency-Coalition",
        "MLP-Coalition",
    ):
        other_rows = {r["seed"]: r for r in rows if r["algorithm"] == other}
        seeds = sorted(set(att_rows) & set(other_rows))
        diffs = np.array(
            [att_rows[s]["S_ESC"] - other_rows[s]["S_ESC"] for s in seeds], dtype=float
        )
        mean, lo, hi = bootstrap_ci(diffs)
        comparisons[f"Att_vs_{other}"] = {
            "mean_diff": mean,
            "ci95": [lo, hi],
            "excludes_zero": not (lo <= 0.0 <= hi),
            "n": len(seeds),
        }

    # Descriptive Att vs Global
    glob_rows = {r["seed"]: r for r in rows if r["algorithm"] == "Global-Coalition"}
    seeds_g = sorted(set(att_rows) & set(glob_rows))
    if seeds_g:
        diffs_g = np.array(
            [att_rows[s]["S_ESC"] - glob_rows[s]["S_ESC"] for s in seeds_g], dtype=float
        )
        mean, lo, hi = bootstrap_ci(diffs_g)
        comparisons["Att_vs_Global-Coalition_descriptive"] = {
            "mean_diff": mean,
            "ci95": [lo, hi],
            "excludes_zero": not (lo <= 0.0 <= hi),
            "n": len(seeds_g),
        }

    claim = False
    if (
        "Att_vs_MLP-Coalition" in comparisons
        and "Att_vs_Local-CBBA-Coalition" in comparisons
    ):
        claim = (
            comparisons["Att_vs_MLP-Coalition"]["excludes_zero"]
            and comparisons["Att_vs_MLP-Coalition"]["mean_diff"] > 0
            and comparisons["Att_vs_Local-CBBA-Coalition"]["excludes_zero"]
            and comparisons["Att_vs_Local-CBBA-Coalition"]["mean_diff"] > 0
        )

    out = {
        "case": args.case,
        "episodes": args.episodes,
        "seed0": args.seed0,
        "replan_interval": args.replan_interval,
        "algorithms": ALGOS,
        "summary": summary,
        "comparisons": comparisons,
        "attention_claim": claim,
    }
    json_path = os.path.join(RESULTS, f"{args.case}_escort_summary_{tag}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    md_path = os.path.join(RESULTS, f"WPS_ESCORT_ANALYSIS_{tag}.md")
    lines = [
        f"# WPS_escort Analysis ({tag})",
        "",
        f"**Suite:** escort + coalition; replan_interval={args.replan_interval}; claim={claim}.",
        "",
        "## Mean S_ESC",
        "",
    ]
    for algo in ALGOS:
        stats = summary[algo]
        lines.append(
            f"- {algo}: {stats['mean_S_ESC']:.2f} ± {stats['std_S_ESC']:.2f} "
            f"(cov={stats['mean_coverage']:.2f}, replan_ms={stats['mean_replan_ms']:.1f}, n={stats['n']})"
        )
    lines.append("")
    lines.append("## Att-Coalition paired diffs")
    lines.append("")
    for k, v in comparisons.items():
        lines.append(
            f"- {k}: mean={v['mean_diff']:.2f}, CI95={v['ci95']}, excludes_zero={v['excludes_zero']}"
        )
    lines.append("")
    lines.append(
        "**Verdict:** "
        + (
            "Attention claim supported on S_ESC (Att beats MLP and CBBA; CIs exclude zero)."
            if claim
            else "Attention claim not supported (Att must beat MLP and CBBA with CIs excluding zero)."
        )
    )
    lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Also refresh the canonical analysis pointer file
    with open(os.path.join(RESULTS, "WPS_ESCORT_ANALYSIS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(json.dumps(out, indent=2), flush=True)
    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
