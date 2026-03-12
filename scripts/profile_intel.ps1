param(
    [string]$BuildDir = "build",
    [switch]$Quick
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force -Path "$root/results" | Out-Null

$syclExe = Join-Path $BuildDir "benchmark_sycl.exe"
if (!(Test-Path $syclExe)) {
    throw "Missing $syclExe. Build benchmark_sycl first."
}

$quickFlag = if ($Quick) { "--quick" } else { "" }

Write-Host "Collecting Intel VTune GPU hotspots..."
vtune -collect gpu-hotspots -result-dir "results/vtune_gpu_hotspots" -- `
    $syclExe $quickFlag --no-validate --csv "results/sycl_profiled.csv"

Write-Host "Collecting oneAPI Advisor roofline data..."
advisor --collect=roofline --project-dir="results/advisor_roofline" -- `
    $syclExe $quickFlag --no-validate --csv "results/sycl_profiled_advisor.csv"

Write-Host "Intel profiling artifacts saved to results/"
