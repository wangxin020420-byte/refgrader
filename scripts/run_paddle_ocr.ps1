param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,

    [string]$OutputDir = "ocr_cache/manual_test",

    [ValidateSet("cpu", "gpu:0")]
    [string]$Device = "cpu",

    [double]$MinConfidence = 0.5,

    [switch]$Recursive,

    [switch]$Force
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$OcrPython = Join-Path $ProjectRoot ".venv-ocr\Scripts\python.exe"
$Worker = Join-Path $ProjectRoot "ocr\paddle_ocr_worker.py"

if (-not (Test-Path $OcrPython)) {
    throw "未找到独立 OCR 环境: $OcrPython"
}

$Arguments = @(
    $Worker,
    "--input", $InputPath,
    "--output-dir", $OutputDir,
    "--device", $Device,
    "--min-confidence", $MinConfidence
)

if ($Recursive) {
    $Arguments += "--recursive"
}
if ($Force) {
    $Arguments += "--force"
}

& $OcrPython @Arguments
exit $LASTEXITCODE
