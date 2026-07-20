"""
Train Attention-RAH on WPS_hard.

Usage:
  python experiments/train_att_rah.py --episodes 280 --case WPS_hard
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from TaskAllocation.Hybrid.AttentionRAH import AttentionRAH, build_att_tokens
from TaskAllocation.OptimizationBased.HungarianAllocator import HungarianAllocator
from experiments.paper_eval import make_config, _events
from experiments.paper_scenarios import CASE_SPECS, WPS_ENV_FLAGS
from mUAV_TA.DroneEnv import MultiUAVEnv

OUT_DIR = os.path.join(ROOT, "dqn_Custom")


def run_episode(env, rah: AttentionRAH, hung: HungarianAllocator, explore: bool):
    observation, info = env.reset()
    done = {a: False for a in env.agents}
    trunc = {a: False for a in env.agents}
    s_prev = 0.0
    while not all(done.values()) and not all(trunc.values()):
        events = _events(info)
        should = (
            env.time_steps == 0
            or env.time_steps % 12 == 0
            or any(
                (ev[0] if isinstance(ev, (list, tuple)) else ev)
                in ("Reset_Allocation", "New_Threat", "Agent_Fail")
                for ev in events
            )
        )
        actions = {}
        tok = build_att_tokens(env, rah.max_tasks, rah.max_agents)
        rho, pri = 0.0, np.zeros(rah.max_tasks, dtype=np.float32)
        if should:
            from TaskAllocation.Hybrid.AttentionRAH import _urgency, _scarcity

            rho, pri = rah.act(tok, explore=explore)
            open_known = tok["open_tasks"]
            vis = tok["vis"]
            live = tok["live"]
            n_urgent = tok["n_urgent"]
            task_pri = {}
            for i, t in enumerate(open_known[: rah.max_tasks]):
                urg = _urgency(t, env.time_steps)
                scar = _scarcity(t, vis, max(len(live), 1))
                task_pri[t.id] = 0.35 * urg + 0.40 * float(pri[i]) + 0.25 * scar
            rho = min(float(rho), 0.25)
            if n_urgent >= 3:
                rho = max(rho, min(0.2, 0.05 * (n_urgent - 2)))
            elif n_urgent <= 1:
                rho = min(rho, 0.05)
            n_reserve = int(round(rho * len(live)))
            reserved = []
            if n_reserve > 0 and open_known:
                scores = []
                for a in live:
                    known_ids = None if vis is None else vis.get(a.name, set())
                    visible = [tt for tt in open_known if known_ids is None or tt.id in known_ids]
                    if not visible:
                        scores.append((1e9, a.name))
                        continue
                    dmin = min(float(np.linalg.norm(a.position - tt.position)) for tt in visible)
                    scores.append((dmin, a.name))
                scores.sort(reverse=True)
                reserved = [n for _, n in scores[:n_reserve]]
            result = hung.allocate_tasks(
                live,
                open_known,
                time_step=env.time_steps,
                events=events,
                force=True,
                task_priorities=task_pri,
                reserved_agent_names=reserved,
                agent_known_ids=vis,
            )
            for agent_name, task in result:
                if env.last_tasks_info and task in env.last_tasks_info:
                    actions[agent_name] = env.last_tasks_info.index(task)

        observation, reward, done, trunc, info = env.step(actions)
        s_now = float(env.compute_s_wps())
        ot_now = float(getattr(env, "n_on_time", 0))
        miss_now = float(getattr(env, "n_missed_windows", 0))
        step_r = (s_now - s_prev) / 20.0
        s_prev = s_now
        next_tok = build_att_tokens(env, rah.max_tasks, rah.max_agents)
        ep_done = all(done.values()) or all(trunc.values())
        if should:
            rah.push(tok, rho, pri, step_r, next_tok, ep_done)
            rah.update(batch_size=64)
    final = info.get("metrics", {}) if isinstance(info, dict) else {}
    return (
        float(final.get("S_WPS", env.compute_s_wps())),
        float(final.get("on_time_rate", 0.0)),
        int(final.get("n_missed_windows", 0)),
        float(final.get("F_Reward", env.F_Reward)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="WPS_hard")
    parser.add_argument("--episodes", type=int, default=280)
    parser.add_argument("--eval-every", type=int, default=40)
    parser.add_argument("--eval-eps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch_seed = args.seed
    import torch

    torch.manual_seed(torch_seed)

    out = args.out or os.path.join(OUT_DIR, f"policy_AttRAH_{args.case}_SWPS.pth")
    os.makedirs(OUT_DIR, exist_ok=True)

    flags = dict(WPS_ENV_FLAGS)
    flags["capability_mask"] = False
    flags["saturate_mask"] = False
    cfg = make_config(CASE_SPECS[args.case], flags)
    cfg.multiple_tasks_per_agent = True

    rah = AttentionRAH()
    best_score = -1e9

    for ep in range(1, args.episodes + 1):
        rah.eps = max(0.05, 0.45 - 0.4 * ep / args.episodes)
        env = MultiUAVEnv(cfg)
        hung = HungarianAllocator(replan_interval=10**9, max_coord=1000.0)
        s_wps, ot, miss, f_legacy = run_episode(env, rah, hung, explore=True)
        if ep % 20 == 0:
            print(
                f"[AttRAH] ep={ep}/{args.episodes} S_WPS={s_wps:.1f} on_time={ot:.2f} miss={miss} F={f_legacy:.1f} eps={rah.eps:.2f}",
                flush=True,
            )
        if ep % args.eval_every == 0 or ep == args.episodes:
            rah.eps = 0.0
            evals_s, evals_ot = [], []
            for i in range(args.eval_eps):
                ev = MultiUAVEnv(cfg)
                hung_e = HungarianAllocator(replan_interval=10**9, max_coord=1000.0)
                ss, oot, _, _ = run_episode(ev, rah, hung_e, explore=False)
                evals_s.append(ss)
                evals_ot.append(oot)
            mean_s = float(np.mean(evals_s))
            mean_ot = float(np.mean(evals_ot))
            score = mean_s + 100.0 * mean_ot
            print(f"  EVAL S_WPS={mean_s:.1f} on_time={mean_ot:.2f} score={score:.1f}", flush=True)
            if score > best_score:
                best_score = score
                rah.save(out)
                # also write canonical name for eval defaults
                canon = os.path.join(OUT_DIR, f"policy_AttRAH_{args.case}.pth")
                rah.save(canon)
                print(f"  Best saved -> {out}", flush=True)
            rah.eps = max(0.05, 0.45 - 0.4 * ep / args.episodes)

    print(f"Done. checkpoint={out} best_score={best_score:.1f}", flush=True)


if __name__ == "__main__":
    main()
