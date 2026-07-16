[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RunDir,

    [string]$Python = "C:\Users\skywalker\.conda\envs\temporal2seq-gpu\python.exe",

    [string]$Config = "configs/causaltad/thumos_pes_q2_persist_fixed.py"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$runPath = [System.IO.Path]::GetFullPath($RunDir)
$configPath = Join-Path $repo $Config
$dataRoot = "C:\data\run01\sczc063\yuzibo\thumos14"
$expectedHashes = [ordered]@{
    "features\pes_siglip2_stride8\manifest.json" = "bca3528b82858cd47a2f9581158dd192dfa65975d2031d009a1a68ec637f6796"
    "annotations\thumos_14_anno.json" = "ee526d55aa4315a8adc68c501d0331f96a56ce16fa960f1d2ea182b9381ab9ad"
    "manifests\full_petal_q2\thumos_fit_core_160.txt" = "36ebd89b583d259989b0260ca7314a308a2ba98ab3bc7bcbf79cc8ec40b628bc"
    "manifests\full_petal_q2\thumos_development_split.json" = "847893621aa44cd665429dcdd6b41f4516f1034de2be555ea9b66d5a5384020f"
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Missing frozen Python executable: $Python"
}
if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "Missing frozen Q2 config: $configPath"
}
if ($Config -ne "configs/causaltad/thumos_pes_q2_persist_fixed.py") {
    throw "Local capacity audit requires the frozen fixed Q2 config"
}
if (Test-Path -LiteralPath $runPath) {
    throw "Refusing to overwrite local capacity run: $runPath"
}

$status = & git -C $repo status --porcelain
if ($LASTEXITCODE -ne 0 -or $status) {
    throw "Local capacity audit requires a clean checkout"
}

foreach ($entry in $expectedHashes.GetEnumerator()) {
    $path = Join-Path $dataRoot $entry.Key
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing frozen local data file: $path"
    }
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $entry.Value) {
        throw "Frozen local data hash mismatch: $path"
    }
}

$parent = Split-Path -Parent $runPath
New-Item -ItemType Directory -Force -Path $parent | Out-Null
New-Item -ItemType Directory -Path $runPath | Out-Null

$previousCudaVisible = $env:CUDA_VISIBLE_DEVICES
$previousOmpThreads = $env:OMP_NUM_THREADS
$previousMklThreads = $env:MKL_NUM_THREADS
try {
    $env:CUDA_VISIBLE_DEVICES = "-1"
    $env:OMP_NUM_THREADS = "8"
    $env:MKL_NUM_THREADS = "8"

    Push-Location $repo
    try {
        & $Python tools\build_q2_capacity_checkpoints.py $Config `
            --output (Join-Path $runPath "checkpoints") `
            --seeds 705 706 707
        if ($LASTEXITCODE -ne 0) {
            throw "Q2 capacity checkpoint generation failed"
        }

        & $Python tools\audit_q2_capacity_lifecycle.py $Config `
            --checkpoint-bundle (Join-Path $runPath "checkpoints") `
            --output (Join-Path $runPath "evidence") `
            --cpu-threads 8 `
            --wall-time-budget-seconds 21000 `
            --cpu-hour-cap 48
        if ($LASTEXITCODE -ne 0) {
            throw "Q2 capacity audit failed"
        }
    }
    finally {
        Pop-Location
    }
}
finally {
    $env:CUDA_VISIBLE_DEVICES = $previousCudaVisible
    $env:OMP_NUM_THREADS = $previousOmpThreads
    $env:MKL_NUM_THREADS = $previousMklThreads
}

Write-Output "Q2_CAPACITY_LOCAL_RUN=$runPath"
