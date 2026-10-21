[CmdletBinding()]
param([int]$Port = 8003)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$storage = Join-Path $repoRoot "inference\gpuocr\storage"
$pidFile = Join-Path $storage "server.pid"
$candidates = @()

if (Test-Path -LiteralPath $pidFile) {
    try {
        $candidates += [int](Get-Content -LiteralPath $pidFile -Raw).Trim()
    } catch {
        # Ignore malformed stale pid files and fall back to the listener.
    }
}
$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($listener) {
    $candidates += [int]$listener.OwningProcess
}

$stopped = $false
foreach ($pidValue in ($candidates | Select-Object -Unique)) {
    if (-not $pidValue) { continue }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $pidValue" -ErrorAction SilentlyContinue
    if (-not $process) { continue }
    $owned = $process.CommandLine -like "*inference.gpuocr.runtime*"
    if (-not $owned) {
        $owned = ($process.CommandLine -like "*uvicorn*" -and $process.CommandLine -like "*8003*")
    }
    if (-not $owned) {
        Write-Warning "PID $pidValue does not belong to GPU OCR; refusing to stop it."
        continue
    }
    Stop-Process -Id $pidValue -Force -ErrorAction SilentlyContinue
    Write-Output "Stopped GPU OCR PID $pidValue."
    $stopped = $true
}

if (Test-Path -LiteralPath $pidFile) {
    Remove-Item -LiteralPath $pidFile -Force
}
if (-not $stopped) {
    Write-Output "No running GPU OCR process found on port $Port."
}
