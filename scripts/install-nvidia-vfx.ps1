<#
.SYNOPSIS
Downloads the NVIDIA runtime for chiaki-ng's NVIDIA VFG frame generation and
RTX Video Super Resolution and installs it into a "vfx" folder.

.DESCRIPTION
chiaki-ng does not redistribute these NVIDIA binaries. This script fetches
NVIDIA's own nvidia-vfx wheel (Video Effects SDK runtime, a zip archive) from
pypi.nvidia.com, checks its SHA-256, and copies out only what chiaki-ng uses:

  NVVideoEffects.dll, NVCVImage.dll, nvVFXVideoFrameGeneration.dll,
  cudart64_12.dll   - NVIDIA VFG (Video Frame Generation)
  nvngx_vsr.dll     - RTX Video Super Resolution 1.8.3 (all modes)

plus NVIDIA's license files. By running it you accept NVIDIA's license terms
for this software (see the "licenses" folder it creates).

.PARAMETER Destination
Folder to install into. Default: %LOCALAPPDATA%\Chiaki\Chiaki\vfx, where
chiaki-ng looks for it (next to chiaki.exe, a "vfx" folder works as well).

.PARAMETER NoPause
Do not wait for a key press at the end.
#>
param(
    [string]$Destination = (Join-Path $env:LOCALAPPDATA 'Chiaki\Chiaki\vfx'),
    [switch]$NoPause
)

$ErrorActionPreference = 'Stop'

$WheelName = 'nvidia_vfx-0.2.0.0-cp312-abi3-win_amd64.whl'
$WheelUrl = "https://pypi.nvidia.com/nvidia-vfx/$WheelName"
$WheelSha256 = 'cebaa12411f7e22b39293205c5c4d07b8073d14ca1e0e7727ee4375207a36445'
$Dlls = @('NVVideoEffects.dll', 'NVCVImage.dll', 'nvVFXVideoFrameGeneration.dll', 'cudart64_12.dll', 'nvngx_vsr.dll')

function Finish([int]$code) {
    if (-not $NoPause) {
        Write-Host ''
        Read-Host 'Press Enter to close' | Out-Null
    }
    exit $code
}

$work = Join-Path ([IO.Path]::GetTempPath()) ("chiaki-vfx-" + [Guid]::NewGuid().ToString('N'))
try {
    Write-Host "chiaki-ng: installing the NVIDIA VFG / RTX VSR runtime into"
    Write-Host "  $Destination"
    Write-Host "Downloading $WheelName (about 420 MB) from pypi.nvidia.com ..."
    New-Item -ItemType Directory -Force -Path $work | Out-Null
    $wheel = Join-Path $work $WheelName

    # curl.exe ships with Windows 10 1803+, is much faster than Invoke-WebRequest
    # and shows progress; fall back to Invoke-WebRequest without it.
    $curl = Join-Path $env:SystemRoot 'System32\curl.exe'
    if (Test-Path $curl) {
        & $curl -L --fail --retry 3 --progress-bar -o $wheel $WheelUrl
        if ($LASTEXITCODE -ne 0) { throw "download failed (curl exit code $LASTEXITCODE)" }
    } else {
        $ProgressPreference = 'SilentlyContinue'
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -UseBasicParsing -Uri $WheelUrl -OutFile $wheel
    }

    Write-Host 'Checking SHA-256 ...'
    $hash = (Get-FileHash -Algorithm SHA256 -Path $wheel).Hash.ToLowerInvariant()
    if ($hash -ne $WheelSha256) { throw "SHA-256 mismatch: got $hash, expected $WheelSha256" }

    Write-Host 'Extracting ...'
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $staging = Join-Path $work 'vfx'
    New-Item -ItemType Directory -Force -Path (Join-Path $staging 'licenses') | Out-Null
    $zip = [IO.Compression.ZipFile]::OpenRead($wheel)
    try {
        foreach ($dll in $Dlls) {
            $entry = $zip.GetEntry("nvvfx/libs/$dll")
            if (-not $entry) { throw "$dll is missing from the package" }
            [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, (Join-Path $staging $dll), $true)
        }
        foreach ($entry in $zip.Entries) {
            if ($entry.FullName -like '*.dist-info/licenses/*' -and $entry.Name) {
                [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, (Join-Path $staging "licenses\$($entry.Name)"), $true)
            }
        }
    } finally {
        $zip.Dispose()
    }
    Set-Content -Path (Join-Path $staging 'SOURCE.txt') -Encoding ASCII -Value @(
        "Extracted unmodified from NVIDIA's $WheelName",
        "($WheelUrl, SHA-256 $WheelSha256)",
        'by chiaki-ng install-nvidia-vfx.ps1. NVIDIA license terms: see licenses\.')

    Write-Host 'Installing ...'
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    try {
        Copy-Item -Path (Join-Path $staging '*') -Destination $Destination -Recurse -Force
    } catch {
        throw "could not write to $Destination (close chiaki-ng if it is running): $($_.Exception.Message)"
    }

    Write-Host ''
    Write-Host 'Done. Restart chiaki-ng, then pick "NVIDIA VFG" under Frame Generation.' -ForegroundColor Green
    $result = 0
} catch {
    Write-Host ''
    Write-Host "Failed: $($_.Exception.Message)" -ForegroundColor Red
    $result = 1
} finally {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $work
}
Finish $result
