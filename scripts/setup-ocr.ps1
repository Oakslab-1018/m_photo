[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [switch]$Force
)

# Predownload PP-OCRv5 (mobile det/cls/rec) ONNX models into the repo-local
# backend storage (`backend/storage/ocr-models`, git-ignored). The Docker
# image caches the same files under /data/ocr-models via the backend-data
# volume; no models are baked into the image or committed to the repo.
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

Write-Host "Downloading PP-OCRv5 mobile det/cls/rec ONNX models to $modelDir"
& uv run --frozen python (Join-Path $repoRoot "scripts\ocr_download_models.py")
if ($LASTEXITCODE -ne 0) {
    throw "OCR model download failed with exit code $LASTEXITCODE"
}

Get-ChildItem -LiteralPath $modelDir -Filter *.onnx -File |
    Select-Object Name, Length |
    Format-Table -AutoSize

Write-Host ""
Write-Host "OCR setup complete. Enable real OCR with:"
Write-Host "  `$env:OCR_PROVIDER='rapidocr'"
Write-Host "Models were cached outside source control: $modelDir"