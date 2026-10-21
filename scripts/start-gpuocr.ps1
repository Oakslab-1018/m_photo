[CmdletBinding()]
param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8003,
    [int]$ReadyTimeoutSeconds = 240,
    [switch]$Foreground
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$serviceRoot = Join-Path $repoRoot "inference\gpuocr"
$python = Join-Path $serviceRoot ".venv\Scripts\python.exe"
$storage = Join-Path $serviceRoot "storage"
$pidFile = Join-Path $storage "server.pid"
$stdout = Join-Path $storage "server.stdout.log"
$stderr = Join-Path $storage "server.stderr.log"
$healthUrl = "http://127.0.0.1:$Port/health"

if (-not (Test-Path -LiteralPath $python)) {
    throw "GPU OCR venv is missing. Run scripts/setup-gpuocr.ps1 first."
}
New-Item -ItemType Directory -Path $storage -Force | Out-Null
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$repoRoot;$env:PYTHONPATH" } else { $repoRoot }

# Never create a second service when the existing one is already healthy.
try {
    $existing = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 5
    if ($existing.state -eq "ok" -and $existing.effective_device -eq "cuda") {
        $existingListener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($existingListener) {
            Set-Content -LiteralPath $pidFile -Value ([int]$existingListener.OwningProcess) -Encoding ascii
        }
        Write-Output "GPU OCR already healthy on $healthUrl"
        exit 0
    }
} catch {
    # Port ownership is checked below before launch.
}

$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($listener) {
    $owner = $listener.OwningProcess
    throw "Port $Port is already owned by PID $owner, but GPU OCR is not healthy. Refusing duplicate launch."
}

$arguments = @(
    "-m", "uvicorn", "inference.gpuocr.runtime:app",
    "--host", $BindHost,
    "--port", "$Port",
    "--workers", "1"
)

if ($Foreground) {
    Push-Location $repoRoot
    try {
        & $python @arguments
        if ($LASTEXITCODE -ne 0) { throw "GPU OCR failed with exit code $LASTEXITCODE" }
    } finally {
        Pop-Location
    }
    exit 0
}

$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$processId = $process.Id
Set-Content -LiteralPath $pidFile -Value $processId -Encoding ascii
Write-Output "GPU OCR starting (PID $processId); waiting for CUDA provider verification."

$deadline = (Get-Date).AddSeconds($ReadyTimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    if ($process.HasExited) {
        $tail = if (Test-Path -LiteralPath $stderr) { Get-Content -LiteralPath $stderr -Tail 40 } else { @() }
        $joinedTail = $tail -join [Environment]::NewLine
        throw "GPU OCR exited before readiness. Log tail: $joinedTail"
    }
    try {
        $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 5
        if ($health.state -eq "ok" -and $health.effective_device -eq "cuda") {
            $listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
                Select-Object -First 1
            if ($listener) {
                $processId = [int]$listener.OwningProcess
                Set-Content -LiteralPath $pidFile -Value $processId -Encoding ascii
            }
            $providers = $health.execution_providers.det -join ","
            Write-Output "GPU OCR ready (PID $processId) providers=$providers"
            Write-Output "Health: $healthUrl"
            exit 0
        }
    } catch {
        # Startup is still loading models or verifying providers.
    }
    Start-Sleep -Seconds 2
}

throw "GPU OCR did not become ready within $ReadyTimeoutSeconds seconds. See $stderr"
