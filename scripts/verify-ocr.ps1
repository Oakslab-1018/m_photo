[CmdletBinding()]
param()

# REAL OCR smoke: downloads PP-OCRv5 models if needed (cached under
# backend/storage/ocr-models) and runs actual ONNX Runtime inference on
# synthetic Windows-font signage. Not part of pytest.
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:M_PHOTO_DATA_DIR = Join-Path $repoRoot "backend\storage"
$env:OCR_PROVIDER = "rapidocr"

Push-Location $repoRoot
try {
    & uv run --frozen python (Join-Path $repoRoot "scripts\verify-ocr.py")
    if ($LASTEXITCODE -ne 0) {
        throw "OCR smoke verification failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}