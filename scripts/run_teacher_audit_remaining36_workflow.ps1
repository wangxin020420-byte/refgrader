param(
    [switch]$Background,
    [switch]$Resume,
    [switch]$Status,
    [int]$MaximumAttempts = 4,
    [int]$InitialRetrySeconds = 60
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Logs = Join-Path $Root "logs"
$StatePath = Join-Path $Logs "teacher_audit_remaining36_state.json"
$PidPath = Join-Path $Logs "teacher_audit_remaining36.pid"
$LatestLogPath = Join-Path $Logs "teacher_audit_remaining36_latest_log.txt"
$LatestErrorPath = Join-Path $Logs "teacher_audit_remaining36_latest_err.txt"

function Read-State {
    if (-not (Test-Path -LiteralPath $StatePath)) {
        return $null
    }
    return Get-Content -LiteralPath $StatePath -Raw -Encoding UTF8 |
        ConvertFrom-Json
}

function Save-State {
    param([object]$State)
    $State.updated_at = (Get-Date).ToString("s")
    $State | ConvertTo-Json -Depth 12 |
        Set-Content -LiteralPath $StatePath -Encoding UTF8
}

function Show-Status {
    $State = Read-State
    Write-Host "===== Remaining 36-question teacher audit ====="
    if ($null -eq $State) {
        Write-Host "Status: not started"
        return
    }

    $Running = $false
    if (Test-Path -LiteralPath $PidPath) {
        $WorkflowPid = Get-Content -LiteralPath $PidPath -Raw
        $WorkflowPid = $WorkflowPid.Trim()
        if ($WorkflowPid) {
            $Running = $null -ne (
                Get-Process -Id ([int]$WorkflowPid) -ErrorAction SilentlyContinue
            )
        }
    }

    Write-Host "Process: $(if ($Running) { 'RUNNING' } else { 'NOT RUNNING' })"
    Write-Host "Status:  $($State.status)"
    Write-Host "Stage:   $($State.stage)"
    Write-Host "Run tag: $($State.run_tag)"
    Write-Host "Updated: $($State.updated_at)"
    Write-Host "Message: $($State.message)"
    Write-Host ""
    foreach ($Batch in $State.batches) {
        Write-Host ("{0,-8} optimize={1,-9} grade={2,-9} run_id={3}" -f `
            $Batch.name, $Batch.optimize_status, $Batch.grade_status, `
            $Batch.run_id)
    }

    if (Test-Path -LiteralPath $LatestLogPath) {
        $LogFile = (Get-Content -LiteralPath $LatestLogPath -Raw).Trim()
        if ($LogFile -and (Test-Path -LiteralPath $LogFile)) {
            Write-Host "`n===== Latest log lines ====="
            Get-Content -LiteralPath $LogFile -Encoding UTF8 -Tail 40
        }
    }
    if (Test-Path -LiteralPath $LatestErrorPath) {
        $ErrorFile = (Get-Content -LiteralPath $LatestErrorPath -Raw).Trim()
        if ($ErrorFile -and (Test-Path -LiteralPath $ErrorFile)) {
            $ErrorLines = @(Get-Content -LiteralPath $ErrorFile -Encoding UTF8)
            if ($ErrorLines.Count -gt 0) {
                Write-Host "`n===== Latest error lines ====="
                $ErrorLines | Select-Object -Last 40
            }
        }
    }
}

if ($Status) {
    Show-Status
    exit 0
}

New-Item -ItemType Directory -Force -Path $Logs | Out-Null

if ($Background) {
    if (Test-Path -LiteralPath $PidPath) {
        $ExistingPid = (Get-Content -LiteralPath $PidPath -Raw).Trim()
        if (
            $ExistingPid -and
            (Get-Process -Id ([int]$ExistingPid) -ErrorAction SilentlyContinue)
        ) {
            throw "The remaining-question audit is already running (PID=$ExistingPid)."
        }
    }

    $Tag = Get-Date -Format "yyyyMMdd_HHmmss"
    $LogFile = Join-Path $Logs "teacher_audit_remaining36_$Tag.log"
    $ErrorFile = Join-Path $Logs "teacher_audit_remaining36_$Tag.err"
    $PowerShell = Join-Path $env:SystemRoot `
        "System32\WindowsPowerShell\v1.0\powershell.exe"
    $Arguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "`"$PSCommandPath`"",
        "-MaximumAttempts", $MaximumAttempts.ToString(),
        "-InitialRetrySeconds", $InitialRetrySeconds.ToString()
    )
    if ($Resume) {
        $Arguments += "-Resume"
    }

    $Process = Start-Process `
        -FilePath $PowerShell `
        -ArgumentList $Arguments `
        -WorkingDirectory $Root `
        -WindowStyle Hidden `
        -RedirectStandardOutput $LogFile `
        -RedirectStandardError $ErrorFile `
        -PassThru

    $Process.Id | Set-Content -LiteralPath $PidPath -Encoding ASCII
    $LogFile | Set-Content -LiteralPath $LatestLogPath -Encoding UTF8
    $ErrorFile | Set-Content -LiteralPath $LatestErrorPath -Encoding UTF8
    Write-Host "Background teacher-audit workflow started."
    Write-Host "PID: $($Process.Id)"
    Write-Host "Log: $LogFile"
    Write-Host "Error log: $ErrorFile"
    exit 0
}

Set-Location -LiteralPath $Root
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:REFGRADER_SAMPLE_POLICY_MODE = "raw"
$Python = (Resolve-Path ".\venv\Scripts\python.exe").Path
$Artifacts = (Resolve-Path "..\refgrader-artifacts").Path

$BatchDefinitions = @(
    @{
        Name = "CO_8_13"
        Questions = @("CO_8", "CO_9", "CO_10", "CO_11", "CO_12", "CO_13")
    },
    @{
        Name = "CPL_1_8"
        Questions = @(
            "CPL_1", "CPL_2", "CPL_3", "CPL_4",
            "CPL_5", "CPL_6", "CPL_7", "CPL_8"
        )
    },
    @{
        Name = "DM_1_3"
        Questions = @("DM_1", "DM_2", "DM_3")
    },
    @{
        Name = "ISC_1_9"
        Questions = @(
            "ISC_1", "ISC_2", "ISC_3", "ISC_4", "ISC_5",
            "ISC_6", "ISC_7", "ISC_8", "ISC_9"
        )
    },
    @{
        Name = "ML_1_4"
        Questions = @("ML_1", "ML_2", "ML_3", "ML_4")
    },
    @{
        Name = "POC_1_6"
        Questions = @("POC_1", "POC_2", "POC_3", "POC_4", "POC_5", "POC_6")
    }
)

if ($Resume) {
    $State = Read-State
    if ($null -eq $State) {
        throw "No previous state exists. Start without -Resume first."
    }
} else {
    $RunTag = Get-Date -Format "yyyyMMdd_HHmmss"
    $Batches = @()
    foreach ($Definition in $BatchDefinitions) {
        $Slug = $Definition.Name.ToLower()
        $Batches += [ordered]@{
            name = $Definition.Name
            questions = @($Definition.Questions)
            run_id = "teacher_audit_${Slug}_$RunTag"
            optimize_status = "pending"
            grade_status = "pending"
            grade_started = $false
            message = ""
        }
    }
    $State = [ordered]@{
        schema_version = 1
        run_tag = $RunTag
        status = "starting"
        stage = "preflight"
        message = ""
        created_at = (Get-Date).ToString("s")
        updated_at = (Get-Date).ToString("s")
        batches = $Batches
    }
    Save-State $State
}

function Update-WorkflowState {
    param(
        [string]$Status,
        [string]$Stage,
        [string]$Message
    )
    $State.status = $Status
    $State.stage = $Stage
    $State.message = $Message
    Save-State $State
}

function Invoke-RetryableStage {
    param(
        [string]$Name,
        [scriptblock]$Action
    )

    $script:LastStageSucceeded = $false
    for ($Attempt = 1; $Attempt -le $MaximumAttempts; $Attempt++) {
        Update-WorkflowState `
            -Status "running" `
            -Stage $Name `
            -Message "attempt $Attempt/$MaximumAttempts"
        Write-Host "`n===== $Name | attempt $Attempt/$MaximumAttempts ====="
        try {
            & $Action $Attempt
            $ExitCode = $LASTEXITCODE
        } catch {
            Write-Error $_
            $ExitCode = 1
        }
        if ($ExitCode -eq 0) {
            $script:LastStageSucceeded = $true
            return
        }
        if ($Attempt -lt $MaximumAttempts) {
            $Delay = [Math]::Min(
                600,
                $InitialRetrySeconds * [Math]::Pow(2, $Attempt - 1)
            )
            Update-WorkflowState `
                -Status "retry_wait" `
                -Stage $Name `
                -Message "exit code $ExitCode; retry in $Delay seconds"
            Write-Host "Stage failed with exit code $ExitCode; retry in $Delay seconds."
            Start-Sleep -Seconds $Delay
        }
    }
}

try {
    Update-WorkflowState -Status "preflight" -Stage "preflight" `
        -Message "checking repositories, data, and tests"

    if (-not $Resume) {
        $ArtifactChanges = @(git -C $Artifacts status --porcelain)
        if ($LASTEXITCODE -ne 0) {
            throw "Unable to inspect refgrader-artifacts."
        }
        if ($ArtifactChanges.Count -gt 0) {
            throw "refgrader-artifacts has uncommitted changes. Commit them before starting."
        }
    }

    & git lfs fsck
    if ($LASTEXITCODE -ne 0) {
        throw "Git LFS check failed."
    }
    & $Python scripts\audit_csbench_snapshot.py --prepared-dir data\csbench
    if ($LASTEXITCODE -ne 0) {
        throw "CSBench snapshot audit failed."
    }
    & $Python -m unittest test_sample_quality.py test_audit_teacher_labels.py -q
    if ($LASTEXITCODE -ne 0) {
        throw "Teacher-audit tests failed."
    }

    foreach ($Batch in $State.batches) {
        $Questions = @($Batch.questions)

        if ($Batch.optimize_status -ne "complete") {
            Invoke-RetryableStage `
                -Name "optimize $($Batch.name)" `
                -Action {
                    param($Attempt)
                    & $Python scripts\run_csbench.py optimize `
                        @Questions `
                        --resume `
                        --allow-baseline-rubric-fallback `
                        --text-provider glm5 `
                        --thinking-mode disabled `
                        --vlm-provider glm4v
                }
            if ($script:LastStageSucceeded) {
                $Batch.optimize_status = "complete"
                $Batch.message = "rubrics ready"
                Save-State $State
            } else {
                $Batch.optimize_status = "failed"
                $Batch.grade_status = "blocked"
                $Batch.message = "optimization exhausted retries"
                Save-State $State
                Write-Host "Skipping grade for $($Batch.name); continuing with later batches."
                continue
            }
        }

        if ($Batch.grade_status -eq "complete") {
            Write-Host "Grade $($Batch.name) already complete; skipped."
            continue
        }

        Invoke-RetryableStage `
            -Name "grade all $($Batch.name)" `
            -Action {
                param($Attempt)
                $GradeArguments = @(
                    "scripts\run_csbench.py", "grade"
                ) + $Questions + @(
                    "--split", "all",
                    "--run-id", $Batch.run_id,
                    "--no-active-a3wa",
                    "--require-complete",
                    "--text-provider", "glm5",
                    "--thinking-mode", "disabled",
                    "--vlm-provider", "glm4v"
                )
                if (-not $Batch.grade_started) {
                    $GradeArguments += "--force"
                    $Batch.grade_started = $true
                    Save-State $State
                }
                & $Python @GradeArguments
            }
        if ($script:LastStageSucceeded) {
            $Batch.grade_status = "complete"
            $Batch.message = "all answers complete and artifacts published"
        } else {
            $Batch.grade_status = "failed"
            $Batch.message = "grading exhausted retries; resume with same run_id"
        }
        Save-State $State
    }

    $FailedBatches = @(
        $State.batches |
            Where-Object {
                $_.optimize_status -ne "complete" -or
                $_.grade_status -ne "complete"
            }
    )
    if ($FailedBatches.Count -eq 0) {
        Update-WorkflowState -Status "complete" -Stage "finished" `
            -Message "all remaining 36 questions completed and published"
        Write-Host "`nAll remaining 36 questions completed and published."
        exit 0
    }

    $FailedNames = ($FailedBatches | ForEach-Object { $_.name }) -join ", "
    Update-WorkflowState -Status "partial" -Stage "finished" `
        -Message "resume required for: $FailedNames"
    Write-Host "`nWorkflow finished with incomplete batches: $FailedNames"
    Write-Host "Run the same script with -Resume -Background."
    exit 1
} catch {
    Update-WorkflowState -Status "failed" -Stage $State.stage `
        -Message $_.Exception.Message
    throw
}
