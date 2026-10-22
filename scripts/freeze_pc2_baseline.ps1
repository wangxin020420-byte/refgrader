param(
    [string]$ExperimentRepo = '',
    [string]$ArtifactsRepo = '',
    [switch]$DryRun,
    [switch]$Stage
)

$ErrorActionPreference = 'Stop'
$ToolRepo = Split-Path -Parent $PSScriptRoot
$RunId = 'initial85_glm52_20261020_pc2'

if (-not $ExperimentRepo) {
    $Candidates = @(
        $ToolRepo,
        (Join-Path (Split-Path -Parent $ToolRepo) 'refgrader-initial-ablation'),
        (Get-Location).Path
    ) | Select-Object -Unique
    foreach ($Candidate in $Candidates) {
        $StatePath = Join-Path $Candidate "results_runs\initial_rubric_ablation\$RunId\workflow.json"
        if (Test-Path -LiteralPath $StatePath -PathType Leaf) {
            $ExperimentRepo = $Candidate
            break
        }
    }
}
if (-not $ExperimentRepo) {
    throw 'PC2 workflow not found. Pass -ExperimentRepo with the existing ablation checkout path; do not rerun grading.'
}
$ExperimentRepo = (Resolve-Path -LiteralPath $ExperimentRepo).Path
if (-not $ArtifactsRepo) {
    $ArtifactsRepo = Join-Path (Split-Path -Parent $ExperimentRepo) 'refgrader-artifacts'
}
if (-not (Test-Path -LiteralPath (Join-Path $ArtifactsRepo '.git'))) {
    throw 'The artifacts Git repository was not found. Pass -ArtifactsRepo with its actual path.'
}
$ArtifactsRepo = (Resolve-Path -LiteralPath $ArtifactsRepo).Path

# Only the standard library is needed; no API key or main/OCR dependencies.
$Executable = $null
$Prefix = @()
$Launcher = Get-Command py.exe -ErrorAction SilentlyContinue
if ($Launcher) {
    try {
        $Probe = & $Launcher.Source -3.11 -c "import sys; print('FREEZE_PYTHON_OK')" 2>&1
        if ($LASTEXITCODE -eq 0 -and ($Probe -match 'FREEZE_PYTHON_OK')) {
            $Executable = $Launcher.Source
            $Prefix = @('-3.11')
        }
    } catch { }
}
if (-not $Executable) {
    $Candidates = @(
        (Join-Path $ToolRepo 'venv\Scripts\python.exe'),
        (Join-Path $ExperimentRepo 'venv\Scripts\python.exe')
    )
    $Candidates += @(Get-Command python.exe -All -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty Source)
    foreach ($Candidate in ($Candidates | Select-Object -Unique)) {
        if (-not $Candidate -or $Candidate -like '*\WindowsApps\*' -or
            -not (Test-Path -LiteralPath $Candidate -PathType Leaf)) { continue }
        try {
            $Probe = & $Candidate -c "import sys; print('FREEZE_PYTHON_OK' if sys.version_info >= (3,8) else 'OLD_PYTHON')" 2>&1
            if ($LASTEXITCODE -eq 0 -and ($Probe -match 'FREEZE_PYTHON_OK')) {
                $Executable = $Candidate
                break
            }
        } catch { }
    }
}
if (-not $Executable) { throw 'No Python >= 3.8 found. No installation was attempted.' }

$env:PYTHONUTF8 = '1'
$Freezer = Join-Path $PSScriptRoot 'freeze_experiment_baseline.py'
$Arguments = @($Prefix) + @($Freezer, 'freeze', '--arm', 'pc2-initial',
    '--repo-root', $ExperimentRepo, '--artifacts-repo', $ArtifactsRepo)
if ($DryRun) { $Arguments += '--dry-run' }
Write-Host "Experiment: $ExperimentRepo"
Write-Host "Artifacts:  $ArtifactsRepo"
Write-Host "Python:     $Executable $($Prefix -join ' ')"
& $Executable @Arguments
if ($LASTEXITCODE -ne 0) { throw 'Freeze validation failed. Original results were not modified.' }

if ($Stage -and -not $DryRun) {
    $Git = Get-Command git.exe -ErrorAction Stop
    $Relative = "csbench/frozen_baselines/$RunId"
    & $Git.Source -c "safe.directory=$ArtifactsRepo" -C $ArtifactsRepo add -- $Relative
    if ($LASTEXITCODE -ne 0) { throw 'Local Git staging failed; the verified snapshot remains on disk.' }
    Write-Host 'Only this frozen run was staged. No commit or push was performed.'
}
Write-Host 'Original checkpoints, active sample policy, model settings and review decisions were not changed.'
