<#
.SYNOPSIS
    Quick-launch AMP training. Only specify what changes per run.
.EXAMPLE
    .\scripts\launch_amp.ps1 v25b_hi_style -StyleWeight 1.0
.EXAMPLE
    .\scripts\launch_amp.ps1 v26 -BCVersion v26 -Segments 10
.EXAMPLE
    .\scripts\launch_amp.ps1 v25_r2 -Resume -Iter 15
.EXAMPLE
    .\scripts\launch_amp.ps1 v25c -DryRun
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0, HelpMessage = "Run suffix → --run amp_to_<Run>")]
    [string]$Run,

    [double]$StyleWeight = 0.5,
    [int]$Segments       = 20,
    [long]$StepsPerSeg   = 50000000,
    [string]$BCVersion   = "v25",
    [string]$Priors      = "v23",

    # Resume from mid-training checkpoint instead of BC pretrain
    [switch]$Resume,
    [string]$ResumeRun,   # Run to resume from (default: same as $Run)
    [int]$Iter,           # Iteration number to resume from

    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# ── Paths ──
$wslProj = "/mnt/c/Users/Eth4ck1e/Documents/Hexapod AI Project"
$python  = "~/.venv-mjx/bin/python"
$runName = "amp_to_$Run"

# ── Restore path ──
if ($Resume) {
    if (-not $Iter) { throw "-Iter is required when using -Resume" }
    $base        = if ($ResumeRun) { "amp_to_$ResumeRun" } else { $runName }
    $restorePath = "checkpoints/$base/iter$Iter/final/params.pkl"
} else {
    $restorePath = "checkpoints/bc_pretrained_jax_$BCVersion/params.pkl"
}

# ── Log path ──
$logTag  = if ($Resume) { "${Run}_resume" } else { "${Run}_train" }
$logPath = "logs/stdout/${logTag}.log"

# Ensure log directory exists
$localLogDir = Join-Path $PSScriptRoot "..\logs\stdout"
if (-not (Test-Path $localLogDir)) { New-Item -ItemType Directory -Path $localLogDir -Force | Out-Null }

# ── Build WSL command ──
$trainArgs = @(
    "--restore $restorePath",
    "--priors checkpoints/amp_priors_${Priors}.npz",
    "--cmd-mask paper_stance",
    "--action-space foot",
    "--style-weight $StyleWeight",
    "--segments $Segments",
    "--steps-per-segment $StepsPerSeg",
    "--partition-disc",
    "--run $runName"
) -join " "

$bashCmd = "cd '$wslProj' && PYTHONPATH=. $python -u scripts/train_jax_amp.py $trainArgs 2>&1 | tee $logPath"
$fullCmd = "wsl bash -lc `"$bashCmd`""

# ── Summary ──
$totalSteps = [math]::Round(($Segments * $StepsPerSeg) / 1e9, 1)

Write-Host ""
Write-Host "  Run:      $runName"          -ForegroundColor Cyan
Write-Host "  Restore:  $restorePath"
Write-Host "  Priors:   amp_priors_${Priors}.npz"
Write-Host "  Style:    $StyleWeight"
Write-Host "  Steps:    ${totalSteps}B  ($Segments seg x $([math]::Round($StepsPerSeg / 1e6))M)"
Write-Host "  Log:      $logPath"
Write-Host ""

if ($DryRun) {
    Write-Host "  [DRY RUN] Would execute:" -ForegroundColor Yellow
    Write-Host "  $fullCmd"                 -ForegroundColor DarkGray
} else {
    Write-Host "  Launching..." -ForegroundColor Green
    Invoke-Expression $fullCmd
}
