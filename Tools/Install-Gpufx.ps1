# SPDX-FileCopyrightText: 2026 ThioJoe
# SPDX-License-Identifier: GPL-3.0-only

<#
    Downloads the GPU module from the ThioJoe/kdenlive-gpu-acceleration releases and
    installs it into a Kdenlive standalone folder:
      <folder>\lib\mlt\libmltgpufx.dll
      <folder>\lib\mlt\gpufx\gpufx-settings.dll

    It takes the newest release with a build for the Kdenlive version in the folder
    name (or -KdenliveVersion), unless -Tag names a release (for example 26.08.2_0.2.0).

    It never waits for input, so it can run unattended: it exits with 0 when
    installed and 1 on any error, with the reason on the error stream.

    Example:
      .\Install-Gpufx.ps1 'C:\Kdenlive\kdenlive-26.08.1_standalone'
#>
param(
    [string]$KdenlivePath,
    [string]$KdenliveVersion,
    [string]$Tag,
    [string]$Repo = 'ThioJoe/kdenlive-gpu-acceleration'
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Fail([string]$message) {
    [Console]::Error.WriteLine("Error: $message")
    exit 1
}

try {
    if (-not $KdenlivePath) { Fail 'Pass -KdenlivePath, the Kdenlive standalone folder (the one containing bin and lib).' }
    $KdenlivePath = $KdenlivePath.Trim().Trim('"').Trim("'").TrimEnd('\')
    if (-not (Test-Path (Join-Path $KdenlivePath 'bin\kdenlive.exe'))) {
        Fail "No bin\kdenlive.exe in '$KdenlivePath'."
    }

    # The build has to match the Kdenlive version, taken from the folder name if not given
    if (-not $KdenliveVersion) {
        if ((Split-Path $KdenlivePath -Leaf) -match '^kdenlive-([\d.]+)') {
            $KdenliveVersion = $Matches[1]
        } else {
            Fail 'Cannot tell the Kdenlive version from the folder name: pass -KdenliveVersion, for example 26.08.1.'
        }
    }

    if (Get-Process kdenlive -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "$KdenlivePath\*" }) {
        Fail 'Close this Kdenlive first.'
    }

    # The build for this Kdenlive version, named after the release's version
    # (gpufx-kdenlive-26.08.2_0.2.0-windows.zip) or, before 0.2.0, after the
    # Kdenlive version alone (gpufx-kdenlive-26.08.1-windows.zip)
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    if ($Tag) {
        $api = "https://api.github.com/repos/$Repo/releases/tags/$Tag"
    } else {
        $api = "https://api.github.com/repos/$Repo/releases?per_page=100"
    }
    try {
        $releases = Invoke-RestMethod -UseBasicParsing -Uri $api
    } catch {
        Fail "Cannot get the releases of $Repo from $api ($($_.Exception.Message))."
    }
    $pattern = '^gpufx-kdenlive-' + [regex]::Escape($KdenliveVersion) + '(_\d+(\.\d+)*)?-windows\.zip$'
    $asset = $null
    # Releases come newest first
    foreach ($release in $releases) {
        $asset = $release.assets | Where-Object { $_.name -match $pattern } | Select-Object -First 1
        if ($asset) { break }
    }
    if (-not $asset) {
        if ($Tag) { Fail "Release $Tag has no build for Kdenlive $KdenliveVersion on Windows." }
        Fail "No release has a build for Kdenlive $KdenliveVersion on Windows yet."
    }
    $file = $asset.name
    $url = $asset.browser_download_url

    $work = Join-Path $env:TEMP "gpufx-install-$PID"
    New-Item -ItemType Directory -Force $work | Out-Null
    $zip = Join-Path $work $file
    Write-Output "Downloading $url"
    try {
        Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $zip
    } catch {
        Fail "Download failed ($($_.Exception.Message))."
    }
    Expand-Archive -Force $zip (Join-Path $work 'build')

    $source = Get-ChildItem (Join-Path $work 'build') -Directory -Filter 'kdenlive-*_standalone' | Select-Object -First 1
    if (-not $source -or -not (Test-Path (Join-Path $source.FullName 'lib\mlt\libmltgpufx.dll'))) {
        Fail 'The download does not contain lib\mlt\libmltgpufx.dll.'
    }

    $mlt = Join-Path $KdenlivePath 'lib\mlt'
    New-Item -ItemType Directory -Force (Join-Path $mlt 'gpufx') | Out-Null
    Copy-Item -Force (Join-Path $source.FullName 'lib\mlt\libmltgpufx.dll') $mlt
    Copy-Item -Force (Join-Path $source.FullName 'lib\mlt\gpufx\gpufx-settings.dll') (Join-Path $mlt 'gpufx')
    Remove-Item -Recurse -Force $work

    Write-Output "Installed $mlt\libmltgpufx.dll and $mlt\gpufx\gpufx-settings.dll"
    exit 0
}
catch {
    Fail $_.Exception.Message
}
