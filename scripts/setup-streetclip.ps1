[CmdletBinding()]
param(
    [switch]$SkipDownload,
    [switch]$SkipTorch,
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$serviceRoot = Join-Path $repoRoot "inference\streetclip"
$venvRoot = Join-Path $serviceRoot ".venv"
$python = Join-Path $venvRoot "Scripts\python.exe"
$requirements = Join-Path $serviceRoot "requirements.txt"
$storage = Join-Path $serviceRoot "storage"
$cacheDir = Join-Path $repoRoot ".uv-cache"

function Assert-LastExit([string]$Step) {
    if ($LASTEXITCODE -ne 0) {
        Write-Error "$Step failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

if (-not (Test-Path -LiteralPath $serviceRoot)) {
    throw "StreetCLIP service directory not found: $serviceRoot"
}

if (-not $PythonExe) {
    $backendPython = Join-Path $repoRoot "backend\.venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $backendPython) {
        $PythonExe = $backendPython
    } else {
        $PythonExe = "python"
    }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required for setup-streetclip.ps1"
}

if (-not (Test-Path -LiteralPath $venvRoot)) {
    Write-Host "Creating isolated Python 3.12 venv: $venvRoot"
    & uv venv --python $PythonExe --cache-dir $cacheDir $venvRoot
    Assert-LastExit "uv venv"
}

& $python -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version"
Assert-LastExit "Python 3.12 check"

if (-not $SkipTorch) {
    Write-Host "Installing torch==2.9.1+cu128 from the official CUDA 12.8 index"
    & uv pip install --python $python --cache-dir $cacheDir --index-url https://download.pytorch.org/whl/cu128 "torch==2.9.1+cu128"
    Assert-LastExit "torch install"
}

Write-Host "Installing pinned StreetCLIP requirements"
& uv pip install --python $python --cache-dir $cacheDir --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match -r $requirements
Assert-LastExit "requirements install"

if (-not (Test-Path -LiteralPath $storage)) {
    New-Item -ItemType Directory -Path $storage -Force | Out-Null
}

if (-not $SkipDownload) {
    Write-Host "Downloading pinned StreetCLIP revision e3561ba2ad9bf14c9efd6b0092607b8497efbfea"
    $env:HF_HOME = Join-Path $storage "huggingface"
    $env:HF_HUB_DOWNLOAD_TIMEOUT = "60"
    $patterns = "config.json,preprocessor_config.json,merges.txt,pytorch_model.bin,special_tokens_map.json,tokenizer.json,tokenizer_config.json,vocab.json,nagasaki.jpg"
    & $python -c "from huggingface_hub import snapshot_download; snapshot_download('geolocal/StreetCLIP', revision='e3561ba2ad9bf14c9efd6b0092607b8497efbfea', cache_dir=r'$env:HF_HOME', allow_patterns=r'$patterns'.split(','), max_workers=2)"
    Assert-LastExit "model download"
}

Write-Host "StreetCLIP setup complete: $venvRoot"