param(
    [string]$PythonCommand = "python"
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$OcrEnv = Join-Path $ProjectRoot ".venv-ocr"
$OcrPython = Join-Path $OcrEnv "Scripts\python.exe"
$Requirements = Join-Path $ProjectRoot "ocr\requirements.txt"

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
exit $LASTEXITCODE
