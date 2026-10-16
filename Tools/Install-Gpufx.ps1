# SPDX-FileCopyrightText: 2026 ThioJoe
# SPDX-License-Identifier: GPL-3.0-only

<#
    Downloads the GPU module from the ThioJoe/kdenlive-gpu-acceleration releases and
    installs it into a Kdenlive standalone folder:
      <folder>\lib\mlt\libmltgpufx.dll
      <folder>\lib\mlt\gpufx\gpufx-settings.dll

    It takes the latest release unless -Tag names one (for example v0.1.0), and the
    build for the Kdenlive version in the folder name unless -KdenliveVersion is given.

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

    $file = "gpufx-kdenlive-$KdenliveVersion-windows.zip"
    if ($Tag) {
        $url = "https://github.com/$Repo/releases/download/$Tag/$file"
    } else {
        $url = "https://github.com/$Repo/releases/latest/download/$file"
    }

    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $work = Join-Path $env:TEMP "gpufx-install-$PID"
    New-Item -ItemType Directory -Force $work | Out-Null
    $zip = Join-Path $work $file
    Write-Output "Downloading $url"
    try {
        Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $zip
    } catch {
        Fail "Download failed ($($_.Exception.Message)). The release may have no build for Kdenlive $KdenliveVersion."
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
