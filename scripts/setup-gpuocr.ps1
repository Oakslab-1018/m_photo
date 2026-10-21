[CmdletBinding()]
param(
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$serviceRoot = Join-Path $repoRoot "inference\gpuocr"
$venvRoot = Join-Path $serviceRoot ".venv"
$python = Join-Path $venvRoot "Scripts\python.exe"
$requirements = Join-Path $serviceRoot "requirements.txt"
$cacheDir = Join-Path $repoRoot ".uv-cache"

function Assert-LastExit([string]$Step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path -LiteralPath $serviceRoot)) {
    throw "GPU OCR service directory not found: $serviceRoot"
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required for setup-gpuocr.ps1"
}
if (-not $PythonExe) {
    $rootPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
    $PythonExe = if (Test-Path -LiteralPath $rootPython) { $rootPython } else { "python" }
}

if (-not (Test-Path -LiteralPath $python)) {
    Write-Host "Creating isolated GPU OCR venv: $venvRoot"
    & uv venv --python $PythonExe --cache-dir $cacheDir $venvRoot
    Assert-LastExit "uv venv"
}

& $python -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version"
Assert-LastExit "Python 3.12 check"

Write-Host "Installing isolated GPU OCR requirements (root CPU venv is not modified)"
& uv pip install --python $python --cache-dir $cacheDir -r $requirements
Assert-LastExit "GPU OCR dependency install"

& $python -c "import importlib.metadata as m; print('rapidocr='+m.version('rapidocr')); print('onnxruntime-gpu='+m.version('onnxruntime-gpu')); print('fastapi='+m.version('fastapi'))"
Assert-LastExit "installed version check"

Write-Host "GPU OCR setup complete: $venvRoot"
Write-Host "Start with: powershell -File scripts/start-gpuocr.ps1"
