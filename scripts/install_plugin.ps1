<#
.SYNOPSIS
    Install Portal-ML's C++ server plugin (.dll + .vdf) and 7-chamber testbed map (.bsp) into Portal 1.
.DESCRIPTION
    Auto-detects your Steam Portal 1 installation (via Windows Registry, Steam libraryfolders.vdf,
    or default Program Files path) and copies portal_ml_plugin.dll, portal_ml_plugin.vdf, and
    portal_ml_5chambers.bsp into <Portal>\portal\addons\ and <Portal>\portal\maps\.
.PARAMETER PortalDir
    Optional explicit path to the Portal 1 install directory (containing hl2.exe).
.PARAMETER SourceDir
    Optional folder containing unpacked release artifacts (addons/ and maps/). Defaults to
    local build outputs when run from a repository checkout, or the script's own directory
    when run from an extracted release zip.
#>
param(
    [string]$PortalDir = "",
    [string]$SourceDir = ""
)

$ErrorActionPreference = "Stop"

function Find-PortalDir {
    $candidates = New-Object System.Collections.Generic.List[string]
    try {
        $steamPath = (Get-ItemProperty -Path "HKCU:\Software\Valve\Steam" -Name "SteamPath" -ErrorAction SilentlyContinue).SteamPath
        if ($steamPath) {
            $candidates.Add((Join-Path $steamPath "steamapps\common\Portal"))
            $vdf = Join-Path $steamPath "steamapps\libraryfolders.vdf"
            if (Test-Path $vdf) {
                $content = Get-Content $vdf -Raw
                foreach ($m in [regex]::Matches($content, '"path"\s+"([^"]+)"')) {
                    $lib = $m.Groups[1].Value -replace '\\\\', '\'
                    $candidates.Add((Join-Path $lib "steamapps\common\Portal"))
                }
            }
        }
    } catch {}
    $candidates.Add("C:\Program Files (x86)\Steam\steamapps\common\Portal")
    $candidates.Add("C:\Program Files\Steam\steamapps\common\Portal")

    foreach ($c in $candidates) {
        if ($c -and (Test-Path (Join-Path $c "hl2.exe"))) {
            return $c
        }
    }
    return ""
}

if (-not $PortalDir) {
    $PortalDir = Find-PortalDir
}
if (-not $PortalDir -or -not (Test-Path (Join-Path $PortalDir "hl2.exe"))) {
    Write-Error "Could not locate Portal 1 (hl2.exe). Pass -PortalDir '<path-to-Portal>' explicitly."
    exit 1
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir

$dllCandidates = @(
    $(if ($SourceDir) { Join-Path $SourceDir "addons\portal_ml_plugin.dll" }),
    $(if ($SourceDir) { Join-Path $SourceDir "portal_ml_plugin.dll" }),
    (Join-Path $scriptDir "addons\portal_ml_plugin.dll"),
    (Join-Path $repoRoot "source_plugin\build-x86\Release\portal_ml_plugin.dll")
) | Where-Object { $_ -and (Test-Path $_) }

$vdfCandidates = @(
    $(if ($SourceDir) { Join-Path $SourceDir "addons\portal_ml_plugin.vdf" }),
    $(if ($SourceDir) { Join-Path $SourceDir "portal_ml_plugin.vdf" }),
    (Join-Path $scriptDir "addons\portal_ml_plugin.vdf"),
    (Join-Path $repoRoot "source_plugin\portal_ml_plugin.vdf")
) | Where-Object { $_ -and (Test-Path $_) }

$bspCandidates = @(
    $(if ($SourceDir) { Join-Path $SourceDir "maps\portal_ml_5chambers.bsp" }),
    $(if ($SourceDir) { Join-Path $SourceDir "portal_ml_5chambers.bsp" }),
    (Join-Path $scriptDir "maps\portal_ml_5chambers.bsp"),
    (Join-Path $repoRoot "maps\portal_ml_5chambers.bsp")
) | Where-Object { $_ -and (Test-Path $_) }

$addonsDir = Join-Path $PortalDir "portal\addons"
$mapsDir = Join-Path $PortalDir "portal\maps"
New-Item -ItemType Directory -Force -Path $addonsDir | Out-Null
New-Item -ItemType Directory -Force -Path $mapsDir | Out-Null

if (-not $dllCandidates) {
    Write-Error "portal_ml_plugin.dll not found. Download the v1.0.0 Windows release bundle or build source_plugin/ first."
    exit 1
}

$dllSrc = $dllCandidates[0]
Copy-Item -Force $dllSrc (Join-Path $addonsDir "portal_ml_plugin.dll")
Write-Host "Installed plugin DLL -> $(Join-Path $addonsDir 'portal_ml_plugin.dll')"

if ($vdfCandidates) {
    Copy-Item -Force $vdfCandidates[0] (Join-Path $addonsDir "portal_ml_plugin.vdf")
} else {
    $vdfContent = "`"Plugin`"`r`n{`r`n    `"file`" `"addons/portal_ml_plugin`"`r`n}`r`n"
    Set-Content -Path (Join-Path $addonsDir "portal_ml_plugin.vdf") -Value $vdfContent -Encoding ASCII
}
Write-Host "Installed plugin VDF -> $(Join-Path $addonsDir 'portal_ml_plugin.vdf')"

if ($bspCandidates) {
    Copy-Item -Force $bspCandidates[0] (Join-Path $mapsDir "portal_ml_5chambers.bsp")
    Write-Host "Installed 7-chamber BSP -> $(Join-Path $mapsDir 'portal_ml_5chambers.bsp')"
} else {
    Write-Warning "portal_ml_5chambers.bsp not found (you can still run on unmodified Valve campaign maps like testchmb_a_02, or compile maps/portal_ml_5chambers.vmf)."
}

Write-Host ""
Write-Host "Portal-ML installation complete! Launch Portal 1 with:"
Write-Host "  .\scripts\launch_portal.ps1 -TextMode"
Write-Host "And run the 7-chamber live verification:"
Write-Host "  python scripts/replay_live.py --backend source --chambers 1 2 3 4 5 6 7 --neural-moveto champions/moveto_mlp.json --neural-parkour champions/parkour_mlp.json --skill-mode auto --neural-aim champions/aim_mlp.json --planner rule"
