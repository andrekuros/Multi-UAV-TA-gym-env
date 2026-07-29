# Confirmatory + COP sweep evals for paper camera-ready numbers.
# Run from repo root: powershell -File experiments/run_final_camera_ready.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

$FinalAlgos = "Local-Hungarian,Local-CBBA-Replan,Local-PI,Urgency-Pair,MLP-ContextPair,Att-ContextPair,GNN-ContextPair,Global-Hungarian"
$AwacsAlgos = $FinalAlgos
$OverAlgos = "Local-Hungarian,Urgency-Pair,MLP-ContextPair,Att-ContextPair,Global-Hungarian"
$CopAlgos = "Local-Hungarian,Urgency-Pair,Global-Hungarian"

function Run-Eval($ArgsList) {
    Write-Host "==== $($ArgsList -join ' ') ====" -ForegroundColor Cyan
    & $Py experiments/wps_eval.py @ArgsList
    if ($LASTEXITCODE -ne 0) { throw "wps_eval failed: $LASTEXITCODE" }
}

# --- FINAL seed 0 ---
Run-Eval @(
    "--suite","WPS_attn","--episodes","100","--exp","final_s0",
    "--algorithms",$FinalAlgos,"--raw",
    "--att-ctx","dqn_Custom/policy_AttContextPairRaw_WPS_attn.pth",
    "--mlp-ctx","dqn_Custom/policy_MLPContextPairRaw_WPS_attn.pth",
    "--gnn-ctx","dqn_Custom/policy_GNNContextPairRawGnn_WPS_attn.pth",
    "--out","experiments/results/wps_attn_final_s0.csv",
    "--episodes-out","experiments/results/wps_attn_final_s0_episodes.csv"
)

# --- FINAL seed 1 ---
Run-Eval @(
    "--suite","WPS_attn","--episodes","100","--exp","final_s1",
    "--algorithms",$FinalAlgos,"--raw",
    "--att-ctx","dqn_Custom/policy_AttContextPairRawRawS1_WPS_attn.pth",
    "--mlp-ctx","dqn_Custom/policy_MLPContextPairRawRawS1_WPS_attn.pth",
    "--gnn-ctx","dqn_Custom/policy_GNNContextPairRawGnnS1_WPS_attn.pth",
    "--out","experiments/results/wps_attn_final_s1.csv",
    "--episodes-out","experiments/results/wps_attn_final_s1_episodes.csv"
)

# --- FINAL seed 2 ---
Run-Eval @(
    "--suite","WPS_attn","--episodes","100","--exp","final_s2",
    "--algorithms",$FinalAlgos,"--raw",
    "--att-ctx","dqn_Custom/policy_AttContextPairRawRawS2_WPS_attn.pth",
    "--mlp-ctx","dqn_Custom/policy_MLPContextPairRawRawS2_WPS_attn.pth",
    "--gnn-ctx","dqn_Custom/policy_GNNContextPairRawGnnS2_WPS_attn.pth",
    "--out","experiments/results/wps_attn_final_s2.csv",
    "--episodes-out","experiments/results/wps_attn_final_s2_episodes.csv"
)

& $Py experiments/summarize_final.py

# --- AWACS ---
Run-Eval @(
    "--suite","WPS_attn_AWACS","--episodes","100","--exp","awacs100",
    "--algorithms",$AwacsAlgos,"--raw",
    "--att-ctx","dqn_Custom/policy_AttContextPairRaw_WPS_attn.pth",
    "--mlp-ctx","dqn_Custom/policy_MLPContextPairRaw_WPS_attn.pth",
    "--gnn-ctx","dqn_Custom/policy_GNNContextPairRawGnn_WPS_attn.pth",
    "--out","experiments/results/wps_attn_awacs.csv",
    "--episodes-out","experiments/results/wps_attn_awacs_episodes.csv"
)

# --- OVERSIZED ---
Run-Eval @(
    "--suite","WPS_oversized","--episodes","100","--exp","oversized100",
    "--algorithms",$OverAlgos,
    "--att-ctx","dqn_Custom/policy_AttContextPairRawScale_WPS_attn.pth",
    "--mlp-ctx","dqn_Custom/policy_MLPContextPairRawScale_WPS_attn.pth",
    "--out","experiments/results/wps_attn_oversized.csv",
    "--episodes-out","experiments/results/wps_attn_oversized_episodes.csv"
)

& $Py experiments/summarize_diagnostics.py

# --- COP sweep ---
Run-Eval @(
    "--suite","WPS_attn_COP","--episodes","100","--exp","cop100",
    "--algorithms",$CopAlgos,
    "--out","experiments/results/wps_attn_cop_sweep.csv",
    "--episodes-out","experiments/results/wps_attn_cop_sweep_episodes.csv"
)

& $Py experiments/summarize_cop_sweep.py
& $Py experiments/plot_cop_sweep.py

# --- COP cueing (share=True; delay is active) ---
Run-Eval @(
    "--suite","WPS_attn_COP_cue","--episodes","100","--exp","cue100",
    "--algorithms",$CopAlgos,
    "--out","experiments/results/wps_attn_cop_cue.csv",
    "--episodes-out","experiments/results/wps_attn_cop_cue_episodes.csv"
)

& $Py experiments/summarize_cop_sweep.py
& $Py experiments/plot_cop_sweep.py

Write-Host "ALL DONE" -ForegroundColor Green
