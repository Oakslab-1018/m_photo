[CmdletBinding()]
param([string]$BindHost = "127.0.0.1", [int]$Port = 8001, [switch]$Foreground)
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot "inference\streetclip\.venv\Scripts\python.exe"
$storage = Join-Path $repoRoot "inference\streetclip\storage"
$pidFile = Join-Path $storage "server.pid"
# Make the shared root catalog_db.py importable by the -m package launch.
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$repoRoot;$env:PYTHONPATH" } else { $repoRoot }
if (-not (Test-Path -LiteralPath $python)) { throw "Run scripts/setup-streetclip.ps1 first." }
New-Item -ItemType Directory -Path $storage -Force | Out-Null
if (Test-Path -LiteralPath $pidFile) {
    $previous = [int](Get-Content -LiteralPath $pidFile -Raw).Trim()
    $running = Get-CimInstance Win32_Process -Filter "ProcessId = $previous" -ErrorAction SilentlyContinue
    if ($running -and $running.CommandLine -like "*inference.streetclip*" -and $running.CommandLine -like "*$repoRoot*") {
        Write-Output "StreetCLIP already running (PID $previous)."; exit 0
    }
}
$arguments = @("-m", "inference.streetclip", "--host", $BindHost, "--port", "$Port")
if ($Foreground) {
    Push-Location $repoRoot
    try { & $python @arguments; if ($LASTEXITCODE -ne 0) { throw "StreetCLIP failed ($LASTEXITCODE)." } }
    finally { Pop-Location }
} else {
    $process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $storage "server.stdout.log") -RedirectStandardError (Join-Path $storage "server.stderr.log")
    Set-Content -LiteralPath $pidFile -Value $process.Id -Encoding ascii
    Write-Output "StreetCLIP starting (PID $($process.Id)); health http://127.0.0.1:$Port/health."
    Write-Output "Logs: inference/streetclip/storage/server.stderr.log"
}
