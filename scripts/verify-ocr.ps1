[CmdletBinding()]
param(
    [ValidateSet("server", "mobile")]
    [string]$ModelType = "server"
)

# REAL OCR smoke: uses the selected PP-OCRv5 det/cls/rec profile, verifies the
# pinned model hashes, and runs actual ONNX Runtime inference on synthetic
# Windows-font signage. Models are cached under backend/storage/ocr-models.
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:M_PHOTO_DATA_DIR = Join-Path $repoRoot "backend\storage"
$env:OCR_PROVIDER = "rapidocr"
$env:OCR_MODEL_TYPE = $ModelType

Push-Location $repoRoot
try {
    & uv run --frozen python (Join-Path $repoRoot "scripts\verify-ocr.py") --model-type $ModelType
    if ($LASTEXITCODE -ne 0) {
        throw "OCR smoke verification failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
