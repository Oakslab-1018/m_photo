[CmdletBinding()]
param()
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pidFile = Join-Path $repoRoot "inference\streetclip\storage\server.pid"
if (-not (Test-Path -LiteralPath $pidFile)) { Write-Output "No saved StreetCLIP PID."; exit 0 }
$serverPid = [int](Get-Content -LiteralPath $pidFile -Raw).Trim()
$process = Get-CimInstance Win32_Process -Filter "ProcessId = $serverPid" -ErrorAction SilentlyContinue
if ($process) {
    if ($process.CommandLine -notlike "*inference.streetclip*" -or $process.CommandLine -notlike "*$repoRoot*") { throw "PID ownership mismatch; refusing to stop." }
    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $serverPid"
    foreach ($child in $children) {
        if ($child.CommandLine -like "*inference.streetclip*") { Stop-Process -Id $child.ProcessId -ErrorAction SilentlyContinue }
    }
    Stop-Process -Id $serverPid -ErrorAction SilentlyContinue
    Write-Output "Stopped StreetCLIP PID $serverPid."
}
