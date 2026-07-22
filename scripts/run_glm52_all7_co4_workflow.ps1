param(
    [switch]$Background,
    [switch]$ResumeOptimize,
    [switch]$RequireDeploymentGate,
    [switch]$AllowExperimentalA3wa,
    [ValidateRange(1, 20)]
    [int]$OptimizeAttempts = 4,
    [ValidateRange(1, 100)]
    [int]$GradeAttempts = 36,
    [ValidateRange(1, 20)]
    [int]$ApiSmokeAttempts = 6,
    [ValidateRange(1, 3600)]
    [int]$InitialRetrySeconds = 60,
    [ValidateRange(1, 3600)]
    [int]$MaximumRetrySeconds = 600
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$Root = Split-Path -Parent $PSScriptRoot
$Logs = Join-Path $Root "logs"
New-Item -ItemType Directory -Force $Logs | Out-Null

if ($Background) {
    $PythonPath = Join-Path $Root "venv\Scripts\python.exe"
    $ArtifactsPath = Join-Path (Split-Path -Parent $Root) "refgrader-artifacts"
    $LatestPidPath = Join-Path $Logs "glm52_all7_co4_latest.pid"
    if (-not (Test-Path -LiteralPath $PythonPath)) {
        throw "Main Python environment not found: $PythonPath"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $ArtifactsPath ".git"))) {
        throw "Sibling refgrader-artifacts repository not found: $ArtifactsPath"
    }
    if (Test-Path -LiteralPath $LatestPidPath) {
        $ExistingPidText = (Get-Content $LatestPidPath -Raw).Trim()
        if ($ExistingPidText -match "^\d+$") {
            $ExistingProcess = Get-CimInstance Win32_Process `
                -Filter "ProcessId = $ExistingPidText" `
                -ErrorAction SilentlyContinue
            if (
                $ExistingProcess -and
                $ExistingProcess.CommandLine -match
                    "run_glm52_all7_co4_workflow\.ps1"
            ) {
                throw "An unattended workflow is already running (PID $ExistingPidText)."
            }
        }
        Remove-Item -LiteralPath $LatestPidPath -Force
    }
    if (-not $ResumeOptimize) {
        $ArtifactChanges = @(git -C $ArtifactsPath status --porcelain)
        if ($LASTEXITCODE -ne 0) {
            throw "Unable to inspect refgrader-artifacts."
        }
        if ($ArtifactChanges.Count -gt 0) {
            throw "refgrader-artifacts has uncommitted changes. Commit them first."
        }
    }

    $Tag = Get-Date -Format "yyyyMMdd_HHmmss"
    $Log = Join-Path $Logs "glm52_all7_co4_$Tag.log"
    $Err = Join-Path $Logs "glm52_all7_co4_$Tag.err"
    $PowerShellExe = Join-Path $env:SystemRoot `
        "System32\WindowsPowerShell\v1.0\powershell.exe"
    $Arguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", ('"{0}"' -f $MyInvocation.MyCommand.Path)
    )
    if ($ResumeOptimize) {
        $Arguments += "-ResumeOptimize"
    }
    if ($RequireDeploymentGate) {
        $Arguments += "-RequireDeploymentGate"
    }
    if ($AllowExperimentalA3wa) {
        $Arguments += "-AllowExperimentalA3wa"
    }
    $Arguments += @(
        "-OptimizeAttempts", $OptimizeAttempts.ToString(),
        "-GradeAttempts", $GradeAttempts.ToString(),
        "-ApiSmokeAttempts", $ApiSmokeAttempts.ToString(),
        "-InitialRetrySeconds", $InitialRetrySeconds.ToString(),
        "-MaximumRetrySeconds", $MaximumRetrySeconds.ToString()
    )
    $Process = Start-Process `
        -FilePath $PowerShellExe `
        -ArgumentList $Arguments `
        -WorkingDirectory $Root `
        -WindowStyle Hidden `
        -RedirectStandardOutput $Log `
        -RedirectStandardError $Err `
        -PassThru
    $Process.Id | Set-Content `
        (Join-Path $Logs "glm52_all7_co4_latest.pid") -Encoding ASCII
    $Log | Set-Content `
        (Join-Path $Logs "glm52_all7_co4_latest_log.txt") -Encoding UTF8
    $Err | Set-Content `
        (Join-Path $Logs "glm52_all7_co4_latest_err.txt") -Encoding UTF8
    Write-Host "Background workflow started."
    Write-Host "PID: $($Process.Id)"
    Write-Host "Log: $Log"
    Write-Host "Error log: $Err"
    exit 0
}

Set-Location -LiteralPath $Root
$Python = Join-Path $Root "venv\Scripts\python.exe"
$Artifacts = Join-Path (Split-Path -Parent $Root) "refgrader-artifacts"
$Questions = @("CO_1", "CO_2", "CO_3", "CO_4", "CO_5", "CO_6", "CO_7")
$Tag = Get-Date -Format "yyyyMMdd_HHmmss"
$ValidationRun = "glm52_off_all7_validation_$Tag"
$CalibrationRun = "glm52_off_all7_calibration_$Tag"
$TestRun = "glm52_off_co4_test_$Tag"
$Config = Join-Path $Root "results_runs\glm52_off_all7_a3wa_$Tag.json"
$StatePath = Join-Path $Logs "glm52_all7_co4_$Tag.state.json"

function Invoke-PythonStage {
    param(
        [string]$Name,
        [string[]]$Arguments
    )
    Write-Host ""
    Write-Host "===== $Name ====="
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

function Invoke-RetryablePythonStage {
    param(
        [string]$Name,
        [string[]]$InitialArguments,
        [string[]]$ResumeArguments,
        [int]$MaximumAttempts
    )

    for ($Attempt = 1; $Attempt -le $MaximumAttempts; $Attempt++) {
        $StageArguments = if ($Attempt -eq 1) {
            $InitialArguments
        } else {
            $ResumeArguments
        }
        Write-Host ""
        Write-Host "===== $Name | attempt $Attempt/$MaximumAttempts ====="
        Save-State `
            -Status "running" `
            -Stage $Name `
            -Attempt $Attempt `
            -Message "stage attempt started"

        & $Python @StageArguments
        $ExitCode = $LASTEXITCODE
        if ($ExitCode -eq 0) {
            Write-Host "===== $Name completed ====="
            return
        }
        if ($Attempt -ge $MaximumAttempts) {
            throw "$Name exhausted $MaximumAttempts attempts; last exit code $ExitCode"
        }

        $Exponent = [Math]::Min($Attempt - 1, 10)
        $Delay = [int][Math]::Min(
            $MaximumRetrySeconds,
            $InitialRetrySeconds * [Math]::Pow(2, $Exponent)
        )
        Save-State `
            -Status "retry_wait" `
            -Stage $Name `
            -Attempt $Attempt `
            -Message "exit code $ExitCode; retry in $Delay seconds"
        Write-Warning (
            "$Name failed with exit code $ExitCode. " +
            "The same run will resume automatically in $Delay seconds."
        )
        Start-Sleep -Seconds $Delay
    }
}

function Save-State {
    param(
        [string]$Status,
        [string]$Stage = "",
        [int]$Attempt = 0,
        [string]$Message = ""
    )
    @{
        status = $Status
        stage = $Stage
        attempt = $Attempt
        message = $Message
        updated_at = (Get-Date).ToString("s")
        validation_run_id = $ValidationRun
        calibration_run_id = $CalibrationRun
        test_run_id = $TestRun
        a3wa_config = $Config
        model = "glm-5.2"
        thinking = "disabled"
    } | ConvertTo-Json | Set-Content $StatePath -Encoding UTF8
    $StatePath | Set-Content `
        (Join-Path $Logs "glm52_all7_co4_latest_state.txt") -Encoding UTF8
}

function Enable-KeepAwake {
    try {
        Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class RefGraderPowerState {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
        $Continuous = [uint32]0x80000000
        $SystemRequired = [uint32]0x00000001
        [void][RefGraderPowerState]::SetThreadExecutionState(
            $Continuous -bor $SystemRequired
        )
        return $true
    } catch {
        Write-Warning "Could not disable automatic sleep: $($_.Exception.Message)"
        return $false
    }
}

function Disable-KeepAwake {
    param([bool]$WasEnabled)
    if ($WasEnabled) {
        $Continuous = [uint32]0x80000000
        [void][RefGraderPowerState]::SetThreadExecutionState($Continuous)
    }
}

$KeepAwakeEnabled = Enable-KeepAwake
$WorkflowFailed = $false
try {
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Main Python environment not found: $Python"
}
if (-not (Test-Path -LiteralPath (Join-Path $Artifacts ".git"))) {
    throw "Sibling refgrader-artifacts repository not found: $Artifacts"
}
if (-not $ResumeOptimize) {
    $ArtifactChanges = @(git -C $Artifacts status --porcelain)
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to inspect refgrader-artifacts."
    }
    if ($ArtifactChanges.Count -gt 0) {
        throw "refgrader-artifacts has uncommitted changes. Commit them first."
    }
}

Save-State -Status "preflight" -Stage "Stage 0/6"
Invoke-PythonStage "Stage 0/6: snapshot audit" @(
    "scripts\audit_csbench_snapshot.py",
    "--prepared-dir", "data\csbench"
)
Invoke-PythonStage "Stage 0/6: question and answer split audit" @(
    "scripts\audit_question_splits.py",
    "--prepared-dir", "data\csbench"
)
Invoke-PythonStage "Stage 0/6: offline tests" @(
    "-m", "unittest", "discover",
    "-p", "test_*.py",
    "-q"
)
$ApiSmokeArguments = @(
    "-c",
    "from step4_vlm_grader import call_text_model; print(call_text_model([{'role':'user','content':'Reply with exactly OK.'}], temperature=0, timeout=60))"
)
Invoke-RetryablePythonStage `
    -Name "Stage 0/6: GLM-5.2 API smoke test" `
    -InitialArguments $ApiSmokeArguments `
    -ResumeArguments $ApiSmokeArguments `
    -MaximumAttempts $ApiSmokeAttempts

Save-State "optimizing_rubrics"
$OptimizeInitialMode = if ($ResumeOptimize) { "--resume" } else { "--force" }
$OptimizeInitialArguments = @(
    "scripts\run_csbench.py", "optimize"
) + $Questions + @(
    $OptimizeInitialMode,
    "--text-provider", "glm5",
    "--thinking-mode", "disabled",
    "--vlm-provider", "glm4v"
)
$OptimizeResumeArguments = @(
    "scripts\run_csbench.py", "optimize"
) + $Questions + @(
    "--resume",
    "--text-provider", "glm5",
    "--thinking-mode", "disabled",
    "--vlm-provider", "glm4v"
)
Invoke-RetryablePythonStage `
    -Name "Stage 1/6: optimize CO_1-CO_7 rubrics" `
    -InitialArguments $OptimizeInitialArguments `
    -ResumeArguments $OptimizeResumeArguments `
    -MaximumAttempts $OptimizeAttempts

$ManifestArguments = @(
    "scripts\run_csbench.py", "grade"
) + $Questions + @(
    "--split", "validation",
    "--dry-run",
    "--no-artifacts",
    "--text-provider", "glm5",
    "--thinking-mode", "disabled",
    "--vlm-provider", "glm4v"
)
Invoke-PythonStage `
    "Stage 2/6: validate rubric manifests" `
    $ManifestArguments

Save-State "grading_validation"
$ValidationArguments = @(
    "scripts\run_csbench.py", "grade"
) + $Questions + @(
    "--split", "validation",
    "--force",
    "--run-id", $ValidationRun,
    "--require-complete",
    "--text-provider", "glm5",
    "--thinking-mode", "disabled",
    "--vlm-provider", "glm4v"
)
$ValidationResumeArguments = @(
    "scripts\run_csbench.py", "grade"
) + $Questions + @(
    "--split", "validation",
    "--run-id", $ValidationRun,
    "--require-complete",
    "--text-provider", "glm5",
    "--thinking-mode", "disabled",
    "--vlm-provider", "glm4v"
)
Invoke-RetryablePythonStage `
    -Name "Stage 3/6: grade CO_1-CO_7 validation" `
    -InitialArguments $ValidationArguments `
    -ResumeArguments $ValidationResumeArguments `
    -MaximumAttempts $GradeAttempts

Save-State "calibrating_a3wa"
$CalibrationArguments = @(
    "scripts\run_csbench.py", "calibrate"
) + $Questions + @(
    "--source-run-id", $ValidationRun,
    "--run-id", $CalibrationRun,
    "--output", $Config,
    "--score-calibration",
    "--text-provider", "glm5",
    "--thinking-mode", "disabled",
    "--vlm-provider", "glm4v"
)
if ($AllowExperimentalA3wa) {
    $CalibrationArguments += "--allow-experimental-a3wa"
}
Invoke-PythonStage `
    "Stage 4/6: calibrate A3WA and residual layer" `
    $CalibrationArguments

$Calibration = Get-Content $Config -Raw -Encoding UTF8 | ConvertFrom-Json
$GatePassed = $Calibration.deployment_gate.passed -eq $true
$ResidualEnabled = $Calibration.score_calibration.enabled -eq $true
Write-Host ""
Write-Host "Deployment gate passed: $GatePassed"
Write-Host "Residual calibration enabled: $ResidualEnabled"
if (-not $ResidualEnabled) {
    throw "Residual calibration was not enabled."
}
if (-not $GatePassed -and -not $AllowExperimentalA3wa) {
    throw "Deployment gate failed; formal test is blocked."
}
if (-not $GatePassed -and $AllowExperimentalA3wa) {
    Write-Warning "Gate failed; continuing CO_4 as a development diagnostic only."
}

$TestArguments = @(
    "scripts\run_csbench.py", "grade", "CO_4",
    "--split", "test",
    "--force",
    "--run-id", $TestRun,
    "--a3wa-config", $Config,
    "--require-complete",
    "--text-provider", "glm5",
    "--thinking-mode", "disabled",
    "--vlm-provider", "glm4v"
)
if ($AllowExperimentalA3wa) {
    $TestArguments += "--allow-experimental-a3wa"
}
$TestResumeArguments = @(
    "scripts\run_csbench.py", "grade", "CO_4",
    "--split", "test",
    "--run-id", $TestRun,
    "--a3wa-config", $Config,
    "--require-complete",
    "--text-provider", "glm5",
    "--thinking-mode", "disabled",
    "--vlm-provider", "glm4v"
)
if ($AllowExperimentalA3wa) {
    $TestResumeArguments += "--allow-experimental-a3wa"
}
Save-State "grading_co4_test"
Invoke-RetryablePythonStage `
    -Name "Stage 5/6: grade CO_4 test" `
    -InitialArguments $TestArguments `
    -ResumeArguments $TestResumeArguments `
    -MaximumAttempts $GradeAttempts

Save-State "finalizing"
Invoke-PythonStage "Stage 6/6: show final outputs" @(
    "scripts\run_csbench.py", "outputs", "CO_4",
    "--split", "test",
    "--run-id", $TestRun
)

Save-State "completed"
Write-Host ""
Write-Host "===== WORKFLOW COMPLETED ====="
Write-Host "Validation run: $ValidationRun"
Write-Host "Calibration run: $CalibrationRun"
Write-Host "A3WA config: $Config"
Write-Host "CO_4 test run: $TestRun"
Write-Host "State: $StatePath"
Write-Host "Artifacts changes:"
git -C $Artifacts status --short
} catch {
    $WorkflowFailed = $true
    Save-State `
        -Status "failed" `
        -Stage "workflow" `
        -Message $_.Exception.Message
    Write-Error $_ -ErrorAction Continue
} finally {
    Disable-KeepAwake -WasEnabled $KeepAwakeEnabled
    $LatestPidPath = Join-Path $Logs "glm52_all7_co4_latest.pid"
    if (Test-Path -LiteralPath $LatestPidPath) {
        $RecordedPid = (Get-Content $LatestPidPath -Raw).Trim()
        if ($RecordedPid -eq $PID.ToString()) {
            Remove-Item -LiteralPath $LatestPidPath -Force
        }
    }
}

if ($WorkflowFailed) {
    exit 1
}
