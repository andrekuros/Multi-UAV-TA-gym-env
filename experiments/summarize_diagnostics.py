"""Summarize AWACS full-COP and oversized-fleet diagnostic CSVs."""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "experiments", "results")


def load_ep(path: str):
    by = defaultdict(lambda: defaultdict(dict))  # case -> algo -> seed -> metrics
    for r in csv.DictReader(open(path, encoding="utf-8")):
        case = r.get("case", "WPS_attn")
        algo = r["algorithm"]
        seed = int(r["seed"])
        by[case][algo][seed] = {
            "S_WPS": float(r["S_WPS"]),
            "on_time_rate": float(r.get("on_time_rate", 0) or 0),
            "n_missed_windows": float(r.get("n_missed_windows", 0) or 0),
            "idle": float(r.get("reserve_idle_fraction", r.get("idle", "nan"))),
        }
    return by


def load_agg_idle(path: str):
    """Fallback idle from aggregate CSV if episodes lack idle column."""
    out = {}
    if not os.path.isfile(path):
        return out
    for r in csv.DictReader(open(path, encoding="utf-8")):
        out[(r["case"], r["algorithm"])] = float(r.get("mean_reserve_idle", "nan"))
    return out


def paired_ci(a_map, b_map, key, rng, n=4000):
    keys = sorted(set(a_map) & set(b_map))
    d = np.array([a_map[k][key] - b_map[k][key] for k in keys])
    boots = [float(np.mean(rng.choice(d, len(d), replace=True))) for _ in range(n)]
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def summarize_awacs():
    ep = os.path.join(RESULTS, "wps_attn_awacs_episodes.csv")
    if not os.path.isfile(ep):
        print(f"Skip AWACS: missing {ep}")
        return
    rng = np.random.RandomState(0)
    data = load_ep(ep)
    case = "WPS_attn_AWACS" if "WPS_attn_AWACS" in data else next(iter(data))
    algos = list(data[case].keys())
    incomplete_path = os.path.join(RESULTS, "wps_attn_final_s0_episodes.csv")
    inc = None
    if os.path.isfile(incomplete_path):
        inc_all = load_ep(incomplete_path)
        inc_case = "WPS_attn" if "WPS_attn" in inc_all else next(iter(inc_all))
        inc = inc_all[inc_case]

    lines = [
        "# WPS_attn_AWACS — full COP (ground + AWACS radar)",
        "",
        "Same mission as WPS_attn but sense_radius=0, threat_delay=0, share_knowledge=True.",
        "Learned checkpoints are zero-shot from incomplete-COP raw IL (seed 0).",
        "",
        "| Method | AWACS mean | Incomplete mean | AWACS - incomplete 95% CI |",
        "|---|---|---|---|",
    ]
    order = [
        "Local-Hungarian",
        "Local-CBBA-Replan",
        "Local-PI",
        "Urgency-Pair",
        "MLP-ContextPair",
        "Att-ContextPair",
        "GNN-ContextPair",
        "Global-Hungarian",
    ]
    for algo in order:
        if algo not in data[case]:
            continue
        aw = float(np.mean([v["S_WPS"] for v in data[case][algo].values()]))
        if inc and algo in inc:
            m, lo, hi = paired_ci(data[case][algo], inc[algo], "S_WPS", rng)
            inc_m = float(np.mean([v["S_WPS"] for v in inc[algo].values()]))
            lines.append(f"| {algo} | {aw:.1f} | {inc_m:.1f} | {m:+.1f} [{lo:+.1f}, {hi:+.1f}] |")
        else:
            lines.append(f"| {algo} | {aw:.1f} | — | — |")

    local = data[case]["Local-Hungarian"]
    lines += [
        "",
        "## vs Local-Hungarian (AWACS)",
        "",
        "| Contrast | mean | 95% CI |",
        "|---|---|---|",
    ]
    for algo in order:
        if algo not in data[case] or algo == "Local-Hungarian":
            continue
        m, lo, hi = paired_ci(data[case][algo], local, "S_WPS", rng)
        lines.append(f"| {algo} - Local | {m:+.1f} | [{lo:+.1f}, {hi:+.1f}] |")

    if "Global-Hungarian" in data[case]:
        m, lo, hi = paired_ci(local, data[case]["Global-Hungarian"], "S_WPS", rng)
        lines += ["", f"Local - Global under AWACS: {m:+.1f} [{lo:+.1f}, {hi:+.1f}]", ""]

    out = os.path.join(RESULTS, "WPS_ATTN_AWACS.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out}")


def summarize_oversized():
    ep = os.path.join(RESULTS, "wps_attn_oversized_episodes.csv")
    agg = os.path.join(RESULTS, "wps_attn_oversized.csv")
    if not os.path.isfile(ep):
        print(f"Skip oversized: missing {ep}")
        return
    rng = np.random.RandomState(0)
    data = load_ep(ep)
    idle_agg = load_agg_idle(agg)

    fleet_map = {"WPS_attn": 12, "WPS_attn_OS18": 18, "WPS_attn_OS24": 24}
    lines = [
        "# WPS oversized fleet diagnostic (same task load, more UAVs)",
        "",
        "Incomplete COP. Task/threat load fixed at WPS_attn; fleets **12 / 18 / 24**.",
        "Att/MLP: zero-shot RawScale (pad 48). Idle = env `reserve_idle_fraction`.",
        "",
        "## Local-Hungarian marginal gain from extra UAVs",
        "",
        "| Change | Delta S_WPS | idle |",
        "|--------|-------------|------|",
    ]

    def mean_s(case, algo):
        return float(np.mean([v["S_WPS"] for v in data[case][algo].values()]))

    def mean_idle(case, algo):
        vals = [v["idle"] for v in data[case][algo].values() if np.isfinite(v["idle"])]
        if vals:
            return float(np.mean(vals))
        return idle_agg.get((case, algo), float("nan"))

    local12, idle12 = mean_s("WPS_attn", "Local-Hungarian"), mean_idle("WPS_attn", "Local-Hungarian")
    local18, idle18 = mean_s("WPS_attn_OS18", "Local-Hungarian"), mean_idle("WPS_attn_OS18", "Local-Hungarian")
    local24, idle24 = mean_s("WPS_attn_OS24", "Local-Hungarian"), mean_idle("WPS_attn_OS24", "Local-Hungarian")
    lines.append(f"| 12 -> 18 | **{local18 - local12:+.1f}** | {idle12:.3f} -> {idle18:.3f} |")
    lines.append(f"| 18 -> 24 | **{local24 - local18:+.1f}** | {idle18:.3f} -> {idle24:.3f} |")
    lines.append(f"| 12 -> 24 | **{local24 - local12:+.1f}** | {idle12:.3f} -> {idle24:.3f} |")
    lines += [
        "",
        "Extra UAVs barely help Local under incomplete COP; idle fraction rises. "
        "Global improves a lot with fleet size (full COP uses the extras).",
        "",
        "## Mean S_WPS and idle",
        "",
        "| Fleet | Method | S_WPS | idle | vs Local CI |",
        "|------:|--------|------:|-----:|-------------|",
    ]
    order = [
        "Local-Hungarian",
        "Urgency-Pair",
        "MLP-ContextPair",
        "Att-ContextPair",
        "Global-Hungarian",
    ]
    for case, fleet in (("WPS_attn", 12), ("WPS_attn_OS18", 18), ("WPS_attn_OS24", 24)):
        if case not in data:
            continue
        local = data[case]["Local-Hungarian"]
        for algo in order:
            if algo not in data[case]:
                continue
            s = mean_s(case, algo)
            idle = mean_idle(case, algo)
            if algo == "Local-Hungarian":
                ci = "—"
            else:
                m, lo, hi = paired_ci(data[case][algo], local, "S_WPS", rng)
                ci = f"[{lo:+.0f}, {hi:+.0f}]"
            lines.append(f"| {fleet} | {algo} | {s:.1f} | {idle:.3f} | {ci} |")

    lines += [
        "",
        "## Takeaway",
        "",
        "Oversizing the fleet under **incomplete COP** does not close the Global gap and yields "
        "diminishing (then flat) Local returns with **higher idle**. Global *does* use extra airframes. "
        "Efficiency story: without better COP, more UAVs mostly idle; with full information they convert "
        "into mission value.",
        "",
    ]
    out = os.path.join(RESULTS, "WPS_ATTN_OVERSIZED.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out}")


def main():
    summarize_awacs()
    summarize_oversized()


if __name__ == "__main__":
    main()
