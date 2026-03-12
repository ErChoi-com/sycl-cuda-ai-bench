param(
    [string]$BuildDir = "build",
    [string]$Configuration = "Release",
    [switch]$Quick,
    [switch]$NoValidate
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

New-Item -ItemType Directory -Force -Path "$root/results" | Out-Null

if (!(Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}

$quickFlag = if ($Quick) { "--quick" } else { "" }
$validateFlag = if ($NoValidate) { "--no-validate" } else { "" }

Write-Host "Configuring CMake..."
cmake -S . -B $BuildDir

Write-Host "Building..."
cmake --build $BuildDir --config Release -j

function Resolve-BinaryPath {
    param([string]$baseDir, [string]$exeName, [string]$config)

    $direct = Join-Path $baseDir $exeName
    if (Test-Path $direct) { return $direct }

    $cfgPath = Join-Path (Join-Path $baseDir $config) $exeName
    if (Test-Path $cfgPath) { return $cfgPath }

    return $direct
}

$cudaExe = Resolve-BinaryPath -baseDir $BuildDir -exeName "benchmark_cuda.exe" -config $Configuration
$syclExe = Resolve-BinaryPath -baseDir $BuildDir -exeName "benchmark_sycl.exe" -config $Configuration
$cpuExe = Resolve-BinaryPath -baseDir $BuildDir -exeName "benchmark_cpu.exe" -config $Configuration

if (Test-Path $cpuExe) {
    Write-Host "Running CPU validation benchmark..."
    & $cpuExe $quickFlag $validateFlag --csv "results/cpu_latest.csv"
} else {
    Write-Warning "CPU binary not found at $cpuExe"
}

if (Test-Path $cudaExe) {
    Write-Host "Running CUDA benchmark..."
    & $cudaExe $quickFlag $validateFlag --csv "results/cuda_latest.csv"
} else {
    Write-Warning "CUDA binary not found at $cudaExe"
}

if (Test-Path $syclExe) {
    Write-Host "Running SYCL benchmark..."
    & $syclExe $quickFlag $validateFlag --csv "results/sycl_latest.csv"
} else {
    Write-Warning "SYCL binary not found at $syclExe"
}

if ((Test-Path "results/cuda_latest.csv") -and (Test-Path "results/sycl_latest.csv")) {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        Write-Host "Merging CUDA and SYCL results..."
        & python scripts/merge_results.py --cuda "results/cuda_latest.csv" --sycl "results/sycl_latest.csv" --out "results/merged_latest.csv"
    } else {
        Write-Warning "Python not found, skipping merged comparison CSV."
    }
}

Write-Host "Done. Results saved in results/"
