#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def load_rows(path: Path):
    rows = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["kernel"]] = row
    return rows


def to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_ratio(num: float, den: float) -> str:
    if den <= 0.0:
        return ""
    return f"{num / den:.4f}"


def main():
    parser = argparse.ArgumentParser(description="Merge CUDA and SYCL benchmark CSV files and compute speedups.")
    parser.add_argument("--cuda", required=True, help="Path to CUDA CSV")
    parser.add_argument("--sycl", required=True, help="Path to SYCL CSV")
    parser.add_argument("--out", required=True, help="Output merged CSV path")
    args = parser.parse_args()

    cuda_path = Path(args.cuda)
    sycl_path = Path(args.sycl)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cuda_rows = load_rows(cuda_path)
    sycl_rows = load_rows(sycl_path)
    kernels = sorted(set(cuda_rows.keys()) | set(sycl_rows.keys()))

    header = [
        "kernel",
        "cuda_avg_ms",
        "sycl_avg_ms",
        "speedup_cuda_over_sycl",
        "cuda_tflops",
        "sycl_tflops",
        "cuda_gbs",
        "sycl_gbs",
        "cuda_valid",
        "sycl_valid",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for kernel in kernels:
            c = cuda_rows.get(kernel, {})
            s = sycl_rows.get(kernel, {})
            c_ms = to_float(c.get("avg_ms", "0"))
            s_ms = to_float(s.get("avg_ms", "0"))

            writer.writerow(
                [
                    kernel,
                    c.get("avg_ms", ""),
                    s.get("avg_ms", ""),
                    format_ratio(s_ms, c_ms),
                    c.get("tflops", ""),
                    s.get("tflops", ""),
                    c.get("gbs", ""),
                    s.get("gbs", ""),
                    c.get("valid", ""),
                    s.get("valid", ""),
                ]
            )

    print(f"Merged comparison CSV written to {out_path}")


if __name__ == "__main__":
    main()
