[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [ValidateSet("server", "mobile")]
    [string]$ModelType = "server",
    [switch]$Force
)

# Precache one complete PP-OCRv5 det/cls/rec profile into the repo-local cache
# (`backend/storage/ocr-models`, git-ignored). Server is the accuracy-oriented
# default; mobile remains available for lower-latency deployments. The script
# verifies every file against RapidOCR 3.10.0's pinned resolver. The Docker
# image exposes the same downloader at /app/scripts/ocr_download_models.py so
# the named /data/ocr-models volume can be populated without baking weights
# into the image.
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$modelDir = Join-Path $repoRoot "backend\storage\ocr-models"

if (-not $PythonExe) {
    $backendPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $backendPython) {
        $PythonExe = $backendPython
    } else {
        $PythonExe = "python"
    }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required for setup-ocr.ps1"
}

& uv run --frozen python -m pip --version *> $null
if ($LASTEXITCODE -ne 0) {
    throw "uv project environment is not ready; run 'uv sync --frozen' first"
}

New-Item -ItemType Directory -Path $modelDir -Force | Out-Null

Write-Host "Preparing PP-OCRv5 $ModelType det/cls/rec ONNX models in $modelDir"
& uv run --frozen python (Join-Path $repoRoot "scripts\ocr_download_models.py") --model-type $ModelType
if ($LASTEXITCODE -ne 0) {
    throw "OCR model download failed with exit code $LASTEXITCODE"
}

Get-ChildItem -LiteralPath $modelDir -Filter *.onnx -File |
    Select-Object Name, Length |
    Format-Table -AutoSize

Write-Host ""
Write-Host "OCR setup complete. Enable real OCR with:"
Write-Host "  `$env:OCR_PROVIDER='rapidocr'"
Write-Host "  `$env:OCR_MODEL_TYPE='$ModelType'"
Write-Host "Models were cached outside source control: $modelDir"
Write-Host "Container prefetch: docker compose run --rm backend python scripts/ocr_download_models.py --model-type $ModelType"
