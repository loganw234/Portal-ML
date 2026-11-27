# Launch Portal 1 with portal_ml_plugin loaded and portal_ml_5chambers.bsp ready for IPC.
param(
    [string]$PortalDir = "C:\Program Files (x86)\Steam\steamapps\common\Portal",
    [switch]$TextMode,
    [switch]$ShowConsole
)

$hl2 = Join-Path $PortalDir "hl2.exe"
if (-not (Test-Path $hl2)) {
    Write-Error "hl2.exe not found at $hl2"
    exit 1
}

$argsList = @(
    "-game", "portal",
    "-insecure",
    "-novid",
    "-nojoy",
    "+sv_cheats", "1",
    "+map", "portal_ml_5chambers"
)

if ($ShowConsole) {
    $argsList += "-console"
}

if ($TextMode) {
    $argsList += "-textmode"
} else {
    $argsList += @("-windowed", "-w", "800", "-h", "600")
}

Write-Host "Launching Portal 1 with Portal-ML plugin..."
Write-Host "Command: `"$hl2`" $($argsList -join ' ')"
Start-Process -FilePath $hl2 -ArgumentList $argsList -WorkingDirectory $PortalDir
