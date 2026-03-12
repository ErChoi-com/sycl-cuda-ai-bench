param(
    [string]$BuildDir = "build",
    [switch]$Quick
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force -Path "$root/results" | Out-Null

$cudaExe = Join-Path $BuildDir "benchmark_cuda.exe"
if (!(Test-Path $cudaExe)) {
    throw "Missing $cudaExe. Build benchmark_cuda first."
}

$quickFlag = if ($Quick) { "--quick" } else { "" }

Write-Host "Collecting Nsight Systems trace..."
nsys profile --force-overwrite true --trace cuda,nvtx,osrt --output "results/nsys_cuda" --sample=none --stats=true -- `
    $cudaExe $quickFlag --no-validate --csv "results/cuda_profiled.csv"

Write-Host "Collecting Nsight Compute for kernels of interest..."
ncu --set full --target-processes all --export "results/ncu_cuda" --force-overwrite -- `
    $cudaExe $quickFlag --no-validate --csv "results/cuda_profiled_ncu.csv"

Write-Host "NVIDIA profiling artifacts saved to results/"
