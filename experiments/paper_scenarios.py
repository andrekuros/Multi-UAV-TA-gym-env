"""Publication evaluation scenarios (static + dynamic D1–D3 + WPS)."""
from __future__ import annotations

from typing import Any, Dict

# Domain-friendly names for paper writing
CASE_SPECS: Dict[str, Dict[str, Any]] = {
    # Static
    "static_strike": {
        "label": "Static Strike",
        "agents": {"F1": 0, "F2": 2, "R1": 0, "R2": 0},
        "tasks": {"Att": 15, "Rec": 0, "Hold": 0},
        "fail_rate": 0.0,
        "threats_list": [],
        "arrival_rate": 0.0,
    },
    "recon_strike_mix": {
        "label": "Recon-Strike Mix",
        "agents": {"F1": 2, "F2": 0, "R1": 4, "R2": 0},
        "tasks": {"Att": 6, "Rec": 12, "Hold": 0},
        "fail_rate": 0.0,
        "threats_list": [],
        "arrival_rate": 0.0,
    },
    "agent_scaling_mid": {
        "label": "Agent Scaling",
        "agents": {"F1": 3, "F2": 0, "R1": 6, "R2": 0},
        "tasks": {"Att": 6, "Rec": 24, "Hold": 0},
        "fail_rate": 0.0,
        "threats_list": [],
        "arrival_rate": 0.0,
    },
    # Dynamic
    "D1_attrition": {
        "label": "Attrition (fail_rate=0.1)",
        "agents": {"F1": 2, "F2": 0, "R1": 4, "R2": 0},
        "tasks": {"Att": 6, "Rec": 12, "Hold": 0},
        "fail_rate": 0.1,
        "threats_list": [],
        "arrival_rate": 0.0,
    },
    "D2_popup_threats": {
        "label": "Pop-up Threats",
        "agents": {"F1": 2, "F2": 2, "R1": 2, "R2": 2},
        "tasks": {"Att": 4, "Rec": 8, "Hold": 0},
        "fail_rate": 0.0,
        "threats_list": [("T1", 3), ("T2", 2)],
        "arrival_rate": 0.0,
    },
    "D3_combined": {
        "label": "Attrition+Threats",
        "agents": {"F1": 2, "F2": 2, "R1": 2, "R2": 2},
        "tasks": {"Att": 4, "Rec": 8, "Hold": 0},
        "fail_rate": 0.1,
        "threats_list": [("T1", 3), ("T2", 2)],
        "arrival_rate": 0.02,
    },
    # WPS — Windowed Pop-up Strike (harder claim)
    "WPS_easy": {
        "label": "WPS Easy (windows+delay)",
        "agents": {"F1": 2, "F2": 2, "R1": 2, "R2": 2},
        "tasks": {"Att": 4, "Rec": 6, "Hold": 0},
        "fail_rate": 0.05,
        "threats_list": [("T1", 4), ("T2", 3)],
        "arrival_rate": 0.08,
        "sense_radius": 250.0,
        "threat_delay": 8,
        "hard_windows": True,
        "window_length": 40,
        "burst_mode": False,
        "burst_size": 2,
        "miss_penalty": 25.0,
        "on_time_bonus": 10.0,
    },
    "WPS_hard": {
        "label": "WPS Hard (tight+local+burst)",
        "agents": {"F1": 2, "F2": 2, "R1": 2, "R2": 2},
        "tasks": {"Att": 3, "Rec": 5, "Hold": 0},
        "fail_rate": 0.08,
        "threats_list": [("T1", 5), ("T2", 4)],
        "arrival_rate": 0.12,
        "sense_radius": 120.0,
        "threat_delay": 15,
        "hard_windows": True,
        "window_length": 25,
        "burst_mode": True,
        "burst_size": 3,
        "miss_penalty": 30.0,
        "on_time_bonus": 12.0,
    },
    "WPS_burst": {
        "label": "WPS Burst stress",
        "agents": {"F1": 2, "F2": 2, "R1": 2, "R2": 2},
        "tasks": {"Att": 2, "Rec": 4, "Hold": 0},
        "fail_rate": 0.1,
        "threats_list": [("T1", 6), ("T2", 4)],
        "arrival_rate": 0.15,
        "sense_radius": 150.0,
        "threat_delay": 12,
        "hard_windows": True,
        "window_length": 20,
        "burst_mode": True,
        "burst_size": 4,
        "miss_penalty": 35.0,
        "on_time_bonus": 15.0,
    },
    # Attention stress: scale + dual fronts + scarce specialists + no knowledge share
    "WPS_attn": {
        "label": "WPS Attn stress (multi-front)",
        "agents": {"F1": 4, "F2": 2, "R1": 4, "R2": 2},
        "tasks": {"Att": 4, "Rec": 8, "Hold": 0},
        "fail_rate": 0.08,
        "threats_list": [("T1", 8), ("T2", 6)],
        "arrival_rate": 0.18,
        "sense_radius": 90.0,
        "threat_delay": 18,
        "hard_windows": True,
        "window_length": 22,
        "burst_mode": True,
        "burst_size": 4,
        "miss_penalty": 30.0,
        "on_time_bonus": 12.0,
        "dual_region_bursts": True,
        "share_knowledge": False,
    },
    # Full COP / AWACS+ground radar: same WPS_attn stress, instant team-wide detection
    "WPS_attn_AWACS": {
        "label": "WPS Attn + full COP (AWACS/ground)",
        "agents": {"F1": 4, "F2": 2, "R1": 4, "R2": 2},
        "tasks": {"Att": 4, "Rec": 8, "Hold": 0},
        "fail_rate": 0.08,
        "threats_list": [("T1", 8), ("T2", 6)],
        "arrival_rate": 0.18,
        "sense_radius": 0.0,
        "threat_delay": 0,
        "hard_windows": True,
        "window_length": 22,
        "burst_mode": True,
        "burst_size": 4,
        "miss_penalty": 30.0,
        "on_time_bonus": 12.0,
        "dual_region_bursts": True,
        "share_knowledge": True,
    },
    # Oversized fleets: same task/threat load as WPS_attn, more UAVs (efficiency diagnostic)
    "WPS_attn_OS18": {
        "label": "WPS Attn oversized 18 agents (1.5x)",
        "agents": {"F1": 6, "F2": 3, "R1": 6, "R2": 3},
        "tasks": {"Att": 4, "Rec": 8, "Hold": 0},
        "fail_rate": 0.08,
        "threats_list": [("T1", 8), ("T2", 6)],
        "arrival_rate": 0.18,
        "sense_radius": 90.0,
        "threat_delay": 18,
        "hard_windows": True,
        "window_length": 22,
        "burst_mode": True,
        "burst_size": 4,
        "miss_penalty": 30.0,
        "on_time_bonus": 12.0,
        "dual_region_bursts": True,
        "share_knowledge": False,
    },
    "WPS_attn_OS24": {
        "label": "WPS Attn oversized 24 agents (2x)",
        "agents": {"F1": 8, "F2": 4, "R1": 8, "R2": 4},
        "tasks": {"Att": 4, "Rec": 8, "Hold": 0},
        "fail_rate": 0.08,
        "threats_list": [("T1", 8), ("T2", 6)],
        "arrival_rate": 0.18,
        "sense_radius": 90.0,
        "threat_delay": 18,
        "hard_windows": True,
        "window_length": 22,
        "burst_mode": True,
        "burst_size": 4,
        "miss_penalty": 30.0,
        "on_time_bonus": 12.0,
        "dual_region_bursts": True,
        "share_knowledge": False,
    },
    # Scale clones of WPS_attn (~2.5× / ~3.3× fleet) for ContextPair transfer eval
    "WPS_attn_L": {
        "label": "WPS Attn L (~30 agents)",
        "agents": {"F1": 10, "F2": 5, "R1": 10, "R2": 5},
        "tasks": {"Att": 10, "Rec": 20, "Hold": 0},
        "fail_rate": 0.08,
        "threats_list": [("T1", 20), ("T2", 15)],
        "arrival_rate": 0.18,
        "sense_radius": 90.0,
        "threat_delay": 18,
        "hard_windows": True,
        "window_length": 22,
        "burst_mode": True,
        "burst_size": 4,
        "miss_penalty": 30.0,
        "on_time_bonus": 12.0,
        "dual_region_bursts": True,
        "share_knowledge": False,
    },
    "WPS_attn_XL": {
        "label": "WPS Attn XL (~40 agents)",
        "agents": {"F1": 14, "F2": 6, "R1": 14, "R2": 6},
        "tasks": {"Att": 13, "Rec": 26, "Hold": 0},
        "fail_rate": 0.08,
        "threats_list": [("T1", 26), ("T2", 20)],
        "arrival_rate": 0.18,
        "sense_radius": 90.0,
        "threat_delay": 18,
        "hard_windows": True,
        "window_length": 22,
        "burst_mode": True,
        "burst_size": 4,
        "miss_penalty": 30.0,
        "on_time_bonus": 12.0,
        "dual_region_bursts": True,
        "share_knowledge": False,
    },
    # Commit/hold stress: same fronts + rematch cost so free Hungarian thrashing hurts
    "WPS_commit": {
        "label": "WPS Commit (dual-front + rematch)",
        "agents": {"F1": 4, "F2": 2, "R1": 4, "R2": 2},
        "tasks": {"Att": 4, "Rec": 8, "Hold": 0},
        "fail_rate": 0.08,
        "threats_list": [("T1", 8), ("T2", 6)],
        "arrival_rate": 0.18,
        "sense_radius": 90.0,
        "threat_delay": 18,
        "hard_windows": True,
        "window_length": 22,
        "burst_mode": True,
        "burst_size": 4,
        "miss_penalty": 30.0,
        "on_time_bonus": 12.0,
        "dual_region_bursts": True,
        "share_knowledge": False,
        "commit_horizon": 25,
        "reassign_penalty": 2.0,
    },
    # Escort / coalition: protect Rec UAVs; T1 needs two fighters; attention on pair scores
    "WPS_escort": {
        "label": "WPS Escort (coalition protect)",
        "agents": {"F1": 5, "F2": 3, "R1": 4, "R2": 2},
        "tasks": {"Att": 2, "Rec": 6, "Hold": 0},
        "fail_rate": 0.03,
        "threats_list": [("T1", 4), ("T2", 6)],
        "arrival_rate": 0.15,
        "sense_radius": 100.0,
        "threat_delay": 15,
        "hard_windows": True,
        "window_length": 28,
        "burst_mode": True,
        "burst_size": 3,
        "miss_penalty": 30.0,
        "on_time_bonus": 12.0,
        "dual_region_bursts": True,
        "share_knowledge": False,
        "commit_horizon": 20,
        "reassign_penalty": 2.0,
        "escort_enabled": True,
        "escort_radius": 70.0,
        "escort_requirement": 1.2,
        "escort_intercept_radius": 100.0,
        "mutual_support_radius": 80.0,
        "escort_agent_types": ("F1", "F2"),
    },
}

# COP-quality 1D sweeps (base = WPS_attn mission load; share_knowledge=False)
# Sense: R in {60,90,150,250} at d=18; Delay: d in {0,6,12,18} at R=90.
# Unique cells: R60, R90(=d18), R150, R250, d0, d6, d12.
import copy as _copy

_WPS_ATTN_BASE = CASE_SPECS["WPS_attn"]
for _r in (60, 90, 150, 250):
    _k = f"WPS_attn_COP_R{_r}"
    _s = _copy.deepcopy(_WPS_ATTN_BASE)
    _s["sense_radius"] = float(_r)
    _s["threat_delay"] = 18
    _s["share_knowledge"] = False
    _s["label"] = f"WPS Attn COP R={_r} d=18"
    CASE_SPECS[_k] = _s
for _d in (0, 6, 12, 18):
    _k = f"WPS_attn_COP_d{_d}"
    _s = _copy.deepcopy(_WPS_ATTN_BASE)
    _s["sense_radius"] = 90.0
    _s["threat_delay"] = int(_d)
    _s["share_knowledge"] = False
    _s["label"] = f"WPS Attn COP R=90 d={_d}"
    CASE_SPECS[_k] = _s

COP_SWEEP_CASES = [
    "WPS_attn_COP_R60",
    "WPS_attn_COP_R90",
    "WPS_attn_COP_R150",
    "WPS_attn_COP_R250",
    "WPS_attn_COP_d0",
    "WPS_attn_COP_d6",
    "WPS_attn_COP_d12",
    # d18 == R90 under share_knowledge=False; omitted to avoid duplicate eval
]

# Cueing-delay sweep: team broadcast (share_knowledge=True), no local radius.
# Under share=False, threat_delay is inactive (discovery is radius-only).
for _d in (0, 6, 12, 18):
    _k = f"WPS_attn_COP_cue_d{_d}"
    _s = _copy.deepcopy(_WPS_ATTN_BASE)
    _s["sense_radius"] = 0.0
    _s["threat_delay"] = int(_d)
    _s["share_knowledge"] = True
    _s["label"] = f"WPS Attn COP cueing d={_d} (share)"
    CASE_SPECS[_k] = _s

COP_CUE_CASES = [
    "WPS_attn_COP_cue_d0",
    "WPS_attn_COP_cue_d6",
    "WPS_attn_COP_cue_d12",
    "WPS_attn_COP_cue_d18",
]

# Aliases matching older eval names
CASE_SPECS["scal_None"] = CASE_SPECS["static_strike"]
CASE_SPECS["train_mixed"] = CASE_SPECS["recon_strike_mix"]
CASE_SPECS["scal_Agents_mid"] = CASE_SPECS["agent_scaling_mid"]

DEFAULT_ENV_FLAGS = {
    "early_terminate": True,
    "capability_mask": True,
    "saturate_mask": True,
    "include_time_windows": True,
    "dynamic_idle_penalty": 0.05,
    "reward_weights": {
        "action": 0.0,
        "distance": 1.0,
        "quality": 1.0,
        "s_quality": 1.0,
        "time": 0.1,
        "alloc": 0.1,
        "time_penaulty": 0.25,
        "step": 0.0,
    },
}

TBTA_E3_FLAGS = {
    **DEFAULT_ENV_FLAGS,
    "reward_weights": {
        "action": 0.0,
        "distance": 1.0,
        "quality": 1.0,
        "s_quality": 1.0,
        "time": 0.0,
        "alloc": 0.0,
        "time_penaulty": 0.0,
        "step": 0.0,
    },
    "dynamic_idle_penalty": 0.0,
    "include_time_windows": False,
}

WPS_ENV_FLAGS = {
    **TBTA_E3_FLAGS,
    "include_time_windows": True,
    "dynamic_idle_penalty": 0.05,
    # Full horizon so pop-up threats/arrivals can spawn after t>40
    "early_terminate": False,
}
