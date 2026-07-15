param(
    [string]$PythonCommand = "python",
    [string]$PaddleCache = ""
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$OcrEnv = Join-Path $ProjectRoot ".venv-ocr"
$OcrPython = Join-Path $OcrEnv "Scripts\python.exe"
$Requirements = Join-Path $ProjectRoot "ocr\requirements.txt"

if (-not $PaddleCache) {
    $ProjectDrive = [System.IO.Path]::GetPathRoot($ProjectRoot)
    $PaddleCache = Join-Path $ProjectDrive "paddlex_cache"
}
$PaddleCache = [System.IO.Path]::GetFullPath($PaddleCache)
try {
    New-Item -ItemType Directory -Force -Path $PaddleCache -ErrorAction Stop | Out-Null
}
catch {
    throw "Cannot create PaddleX model cache: $PaddleCache"
}

$env:PADDLE_PDX_CACHE_HOME = $PaddleCache
$env:PADDLE_PDX_MODEL_SOURCE = "modelscope"
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"

[Environment]::SetEnvironmentVariable(
    "PADDLE_PDX_CACHE_HOME", $PaddleCache, "User"
)
[Environment]::SetEnvironmentVariable(
    "PADDLE_PDX_MODEL_SOURCE", "modelscope", "User"
)
[Environment]::SetEnvironmentVariable(
    "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True", "User"
)

if (-not (Test-Path $OcrPython)) {
    & $PythonCommand -m venv $OcrEnv
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create .venv-ocr."
    }
}

& $OcrPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade OCR environment tooling."
}

& $OcrPython -m pip install -r $Requirements
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install PaddleOCR dependencies."
}

$VersionCheck = "import paddle,paddleocr; print('PaddlePaddle', paddle.__version__); print('PaddleOCR', paddleocr.__version__); print('CUDA', paddle.is_compiled_with_cuda())"
& $OcrPython -c $VersionCheck
Write-Host "Authoritative PaddleX model cache: $PaddleCache"
exit $LASTEXITCODE
