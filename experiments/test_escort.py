"""Compact regression tests for escort / unique IDs / coalition Hungarian."""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from TaskAllocation.OptimizationBased.HungarianAllocator import HungarianAllocator
from experiments.paper_eval import make_config, _open_tasks
from experiments.paper_scenarios import CASE_SPECS, WPS_ENV_FLAGS
from mUAV_TA.DroneEnv import MultiUAVEnv


def test_unique_task_ids():
    flags = dict(WPS_ENV_FLAGS)
    flags["capability_mask"] = False
    flags["saturate_mask"] = False
    cfg = make_config(CASE_SPECS["WPS_escort"], flags)
    cfg.multiple_tasks_per_agent = True
    env = MultiUAVEnv(cfg)
    env.reset(seed=7)
    hung = HungarianAllocator(replan_interval=5, max_coord=env.max_coord)
    for _ in range(80):
        events = []
        result = hung.allocate_tasks(
            env.get_live_agents(),
            _open_tasks(env),
            time_step=env.time_steps,
            events=events,
            force=True,
            agent_known_ids=env.agent_visibility_map(),
        )
        actions = {}
        for name, task in result:
            if env.last_tasks_info and task in env.last_tasks_info:
                actions[name] = env.last_tasks_info.index(task)
        _, _, done, trunc, info = env.step(actions)
        ids = [t.id for t in env.tasks]
        assert len(ids) == len(set(ids)), f"duplicate task ids: {ids}"
        if all(done.values()) or all(trunc.values()):
            break
    print("OK unique_task_ids")


def test_escort_lifecycle_and_follow():
    flags = dict(WPS_ENV_FLAGS)
    flags["capability_mask"] = False
    flags["saturate_mask"] = False
    cfg = make_config(CASE_SPECS["WPS_escort"], flags)
    cfg.multiple_tasks_per_agent = True
    env = MultiUAVEnv(cfg)
    env.reset(seed=3)
    # Force a recon onto a Rec task
    recon = next(a for a in env.agents_obj if a.type.startswith("R"))
    rec = next(t for t in env.tasks if t.type == "Rec")
    assert env._is_task_action_valid(recon, rec)
    assert recon.allocate(rec, env.time_steps)
    env._create_escort_for(recon, rec)
    escort = env._escort_by_recon[recon.name]
    assert escort.kind == "Escort"
    assert escort.eligible_agent_types == {"F1", "F2"}
    # Rec UAV cannot escort itself
    assert not env._is_task_action_valid(recon, escort)
    fighter = next(a for a in env.agents_obj if a.type == "F1")
    assert env._is_task_action_valid(fighter, escort)
    # Follow
    recon.position = np.array([400.0, 300.0])
    env._sync_escorts()
    assert np.allclose(escort.position, recon.position)
    env._retire_escort(escort, failed=False)
    assert escort.status == 2
    assert recon.name not in env._escort_by_recon
    print("OK escort_lifecycle")


def test_coalition_two_slot_and_legacy():
    flags = dict(WPS_ENV_FLAGS)
    cfg = make_config(CASE_SPECS["WPS_escort"], flags)
    cfg.multiple_tasks_per_agent = True
    env = MultiUAVEnv(cfg)
    env.reset(seed=1)
    hung = HungarianAllocator(replan_interval=1, max_coord=env.max_coord)

    # Synthetic escort needing 2 fighters
    recon = next(a for a in env.agents_obj if a.type.startswith("R"))
    rec = next(t for t in env.tasks if t.type == "Rec")
    escort = env._create_escort_for(recon, rec)
    fighters = [a for a in env.get_live_agents() if a.type in ("F1", "F2")][:3]
    for a in fighters:
        a.tasks = [env.task_idle]
        a.state = 0
    result = hung.allocate_tasks(
        fighters,
        [escort],
        time_step=env.time_steps,
        force=True,
        edge_scores={(f.name, escort.id): 1.0 for f in fighters},
    )
    assert len(result) >= 2, f"expected >=2 escort assigns, got {result}"

    # Legacy required_agents=0 / residual1 stays single-slot capable
    att = next(t for t in env.tasks if t.type == "Att")
    att.required_agents = 0
    single = hung.allocate_tasks(
        fighters[:1],
        [att],
        time_step=env.time_steps,
        force=True,
    )
    assert len(single) <= 1
    print("OK coalition_hungarian")


def test_threat_diversion_inputs():
    flags = dict(WPS_ENV_FLAGS)
    cfg = make_config(CASE_SPECS["WPS_escort"], flags)
    env = MultiUAVEnv(cfg)
    env.reset(seed=2)
    recon = next(a for a in env.agents_obj if a.type.startswith("R"))
    rec = next(t for t in env.tasks if t.type == "Rec")
    assert recon.allocate(rec, env.time_steps)
    escort = env._create_escort_for(recon, rec)
    f1 = next(a for a in env.agents_obj if a.type == "F1")
    f2 = next(a for a in env.agents_obj if a.type == "F2")
    recon.position = np.array([500.0, 400.0])
    f1.position = recon.position + np.array([10.0, 0.0])
    f2.position = recon.position + np.array([0.0, 10.0])
    assert f1.allocate(escort, env.time_steps)
    assert f2.allocate(escort, env.time_steps)
    env._sync_escorts()
    nearby = env._escort_fighters_near(recon)
    assert len(nearby) >= 2, f"nearby={nearby} escort_status={escort.status}"
    print("OK threat_diversion_inputs")


def test_cbba_pi_coalition_eligibility_visibility():
    from TaskAllocation.MarketBased.CBBA import CBBA
    from TaskAllocation.MarketBased.CBBA_Replan import CBBAReplan, REPLAN_EVENTS
    from TaskAllocation.MarketBased.PerformanceImpact import PerformanceImpact

    flags = dict(WPS_ENV_FLAGS)
    cfg = make_config(CASE_SPECS["WPS_escort"], flags)
    cfg.multiple_tasks_per_agent = True
    env = MultiUAVEnv(cfg)
    env.reset(seed=5)

    recon = next(a for a in env.agents_obj if a.type.startswith("R"))
    rec = next(t for t in env.tasks if t.type == "Rec")
    assert recon.allocate(rec, env.time_steps)
    escort = env._create_escort_for(recon, rec)
    fighters = [a for a in env.get_live_agents() if a.type in ("F1", "F2")]
    for a in fighters:
        a.tasks = [env.task_idle]
        a.state = 0

    # Visibility: only first two fighters know the escort
    known = {f.name: set() for f in fighters}
    known[fighters[0].name].add(escort.id)
    known[fighters[1].name].add(escort.id)

    cbba = CBBA(fighters, [escort], env.max_coord, seed=1)
    cbba_out = cbba.allocate_tasks(
        fighters + [recon],
        [escort],
        agent_known_ids={**known, recon.name: {escort.id}},
        max_tasks_per_agent=1,
    )
    assigned = []
    for name, tasks in cbba_out:
        for t in tasks:
            assigned.append((name, t.id))
    assert len(assigned) >= 2, f"CBBA expected >=2 escort assigns, got {assigned}"
    assert all(n != recon.name for n, _ in assigned), "recon must not escort"
    assert all(n in known and escort.id in known[n] for n, _ in assigned)

    # Incumbent residual: one fighter already on escort -> one remaining slot
    assert fighters[0].allocate(escort, env.time_steps)
    rem = CBBA(fighters, [escort], env.max_coord, seed=2).allocate_tasks(
        fighters,
        [escort],
        agent_known_ids=known,
        max_tasks_per_agent=1,
    )
    rem_names = [n for n, _ in rem]
    assert fighters[0].name not in rem_names
    assert len(rem_names) >= 1

    pi = PerformanceImpact(max_coord=env.max_coord, seed=3, replan_interval=1)
    pi_out = pi.allocate_tasks(
        fighters,
        [escort],
        time_step=env.time_steps,
        force=True,
        agent_known_ids=known,
        max_tasks_per_agent=1,
    )
    pi_assigned = [(n, t.id) for n, ts in pi_out for t in ts]
    # After incumbent, residual may be 1; force fresh escort for PI two-slot check
    escort2_recon = next(
        a for a in env.agents_obj if a.type.startswith("R") and a.name != recon.name
    )
    escort2_rec = next(t for t in env.tasks if t.type == "Rec" and t.id != rec.id)
    assert escort2_recon.allocate(escort2_rec, env.time_steps)
    escort2 = env._create_escort_for(escort2_recon, escort2_rec)
    for a in fighters:
        if a.tasks and a.tasks[0].id == escort.id:
            continue
        a.tasks = [env.task_idle]
        a.state = 0
    known2 = {f.name: {escort2.id} for f in fighters[:3]}
    pi_out2 = pi.allocate_tasks(
        fighters[:3],
        [escort2],
        time_step=env.time_steps,
        force=True,
        agent_known_ids=known2,
        max_tasks_per_agent=1,
    )
    pi_ids = [n for n, ts in pi_out2 for _t in ts]
    assert len(set(pi_ids)) >= 2, f"PI expected >=2 distinct fighters, got {pi_out2}"
    assert len(pi_ids) == len(set(pi_ids)), "no duplicate agent assignment"

    # Escort events trigger replan
    assert "Escort_Created" in REPLAN_EVENTS and "Escort_Retired" in REPLAN_EVENTS
    rp = CBBAReplan(env.agents_obj, env.tasks, env.max_coord, seed=0, replan_interval=1000)
    assert rp.should_replan(10, [["Escort_Created", escort.id]])
    assert rp.should_replan(10, [["Escort_Retired", escort.id]])
    assert pi.should_replan(10, [["Escort_Created", escort.id]])

    # Action conversion
    env.last_tasks_info = list(env.tasks)
    from experiments.escort_eval import _apply_assign

    actions = _apply_assign(env, pi_out2)
    assert actions, "expected convertible actions"
    print("OK cbba_pi_coalition")


def test_pi_schedule_impact_prefers_nearer():
    from TaskAllocation.MarketBased.PerformanceImpact import PerformanceImpact

    flags = dict(WPS_ENV_FLAGS)
    cfg = make_config(CASE_SPECS["WPS_escort"], flags)
    env = MultiUAVEnv(cfg)
    env.reset(seed=9)
    recon = next(a for a in env.agents_obj if a.type.startswith("R"))
    rec = next(t for t in env.tasks if t.type == "Rec")
    assert recon.allocate(rec, env.time_steps)
    escort = env._create_escort_for(recon, rec)
    near = next(a for a in env.agents_obj if a.type == "F1")
    far = next(a for a in env.agents_obj if a.type == "F2")
    recon.position = np.array([500.0, 400.0])
    escort.position = recon.position.copy()
    near.position = recon.position + np.array([5.0, 0.0])
    far.position = recon.position + np.array([400.0, 0.0])
    for a in (near, far):
        a.tasks = [env.task_idle]
        a.state = 0
    known = {near.name: {escort.id}, far.name: {escort.id}}
    pi = PerformanceImpact(max_coord=env.max_coord, seed=0, replan_interval=1)
    out = pi.allocate_tasks(
        [near, far],
        [escort],
        time_step=0,
        force=True,
        agent_known_ids=known,
        max_tasks_per_agent=1,
    )
    names = [n for n, _ in out]
    assert near.name in names, f"near fighter should be chosen, got {out}"
    print("OK pi_schedule_impact")


def test_att_coalition_v2_tokens_and_buffer():
    from TaskAllocation.Hybrid.AttentionEscort import AttentionEscort, build_escort_tokens
    from TaskAllocation.OptimizationBased.HungarianAllocator import HungarianAllocator

    flags = dict(WPS_ENV_FLAGS)
    cfg = make_config(CASE_SPECS["WPS_escort"], flags)
    env = MultiUAVEnv(cfg)
    env.reset(seed=1)
    tok = build_escort_tokens(env, max_tasks=48, max_agents=16)
    assert tok["task_feats"].shape == (48, 22)
    assert tok["agent_feats"].shape == (16, 16)
    assert tok["edge_valid"].shape == (16, 48)
    assert tok["edge_valid"].sum() > 0

    att = AttentionEscort(
        use_attention=True, max_tasks=48, max_agents=16, d_model=64, n_layers=2, lr=1e-3
    )
    hung = HungarianAllocator(replan_interval=10**9, max_coord=1000.0)
    result, tok2, scores, noise, logits, selected = att.plan(
        env, hung, events=[], explore=True, force=True
    )
    assert scores.shape == (16, 48)
    assert selected.shape == (16, 48)
    assert selected.sum() >= 0
    next_tok = att.build_tokens(env)
    att.push(tok2, scores, noise, logits, selected, 0.1, next_tok, False)
    assert len(att.buffer) == 1
    # Fill enough for one update
    for _ in range(20):
        att.push(tok2, scores, noise, logits, selected, 0.05, next_tok, False)
    loss = att.update(batch_size=16)
    assert loss is not None
    path = os.path.join(os.path.dirname(__file__), "..", "dqn_Custom", "_smoke_att_v2.pth")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    att.save(path)
    att2 = AttentionEscort(use_attention=True)
    att2.load(path)
    assert att2.max_tasks == 48
    assert att2.d_model == 64
    assert att2.n_layers == 2
    print("OK att_coalition_v2")


if __name__ == "__main__":
    test_unique_task_ids()
    test_escort_lifecycle_and_follow()
    test_coalition_two_slot_and_legacy()
    test_threat_diversion_inputs()
    test_cbba_pi_coalition_eligibility_visibility()
    test_pi_schedule_impact_prefers_nearer()
    test_att_coalition_v2_tokens_and_buffer()
    print("ALL PASSED")
