"""Dual-architecture benchmark driver for NVIDIA A100 (sm_80) and Blackwell G4 (sm_120).

Benchmarks packing and seeded CUDA solvers across puzzle sizes and replica chains,
measuring latency, proposal throughput, and certified square packing quality.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

# Ensure repository paths are accessible
REPO_ROOT = Path(__file__).resolve().parents[1]
CUDA_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(CUDA_DIR) not in sys.path:
    sys.path.insert(0, str(CUDA_DIR))

import numpy as np
from demo import certified_candidate
from squares.core import certify, grid
from squares.optimizer import tilted_five

ARCH_NVCC_FLAGS: dict[str, list[str]] = {
    "sm_80": ["-gencode", "arch=compute_80,code=sm_80"],
    "sm_120": ["-gencode", "arch=compute_120,code=sm_120"],
}


def detect_device(device_idx: int = 0) -> dict:
    """Detect active CUDA GPU device via torch.cuda.get_device_name() and get_device_capability()."""
    try:
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            idx = max(0, min(device_idx, count - 1)) if count > 0 else 0
            name = torch.cuda.get_device_name(idx)
            capability = torch.cuda.get_device_capability(idx)
            major, minor = capability
            if major == 12:
                arch = "sm_120"
            elif major == 8 and minor == 0:
                arch = "sm_80"
            elif major == 8 and minor == 9:
                arch = "sm_89"
            elif major == 9 and minor == 0:
                arch = "sm_90"
            else:
                arch = f"sm_{major}{minor}"
            return {
                "available": True,
                "device_idx": idx,
                "device_count": count,
                "name": name,
                "capability": [major, minor],
                "detected_arch": arch,
            }
    except Exception as exc:
        return {
            "available": False,
            "device_idx": device_idx,
            "device_count": 0,
            "name": f"Detection unavailable ({exc})",
            "capability": [0, 0],
            "detected_arch": "sm_80",
        }
    return {
        "available": False,
        "device_idx": device_idx,
        "device_count": 0,
        "name": "No CUDA device available",
        "capability": [0, 0],
        "detected_arch": "sm_80",
    }


def detect_all_devices() -> list[dict]:
    """Enumerate all available CUDA devices."""
    devices = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(detect_device(i))
    except Exception:
        pass
    if not devices:
        devices.append(detect_device(0))
    return devices


def compile_solver(
    solver: str,
    arch: str,
    out_path: Path,
    nvcc: str = "nvcc",
) -> tuple[bool, str]:
    """Compile CUDA solver with nvcc targeting specified architecture flags."""
    src = CUDA_DIR / f"{solver}.cu"
    if not src.exists():
        return False, f"Source file {src} not found"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flags = ARCH_NVCC_FLAGS.get(arch, [f"-arch={arch}"])
    cmd = [
        nvcc,
        "-O3",
        "-std=c++17",
        f"-I{CUDA_DIR}",
    ] + flags + [str(src), "-o", str(out_path)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, res.stdout + res.stderr
    except subprocess.CalledProcessError as err:
        return False, f"nvcc compilation failed: {err.stderr}\nCommand: {' '.join(cmd)}"
    except FileNotFoundError:
        return False, f"nvcc compiler not found at '{nvcc}'"


def resolve_binary(
    solver: str,
    arch: str,
    bin_dir: Path,
    compile_if_missing: bool = True,
    force_compile: bool = False,
    dry_run: bool = False,
) -> tuple[Path | None, str]:
    """Locate or compile binary for requested solver and architecture."""
    arch_bin = bin_dir / f"{solver}_{arch}"
    default_bin = bin_dir / solver

    if force_compile and not dry_run:
        ok, msg = compile_solver(solver, arch, arch_bin)
        return (arch_bin, "compiled") if ok else (None, msg)

    if arch_bin.is_file() and os.access(arch_bin, os.X_OK):
        return arch_bin, "arch_binary"
    if default_bin.is_file() and os.access(default_bin, os.X_OK):
        return default_bin, "default_binary"

    if compile_if_missing and not dry_run:
        ok, msg = compile_solver(solver, arch, arch_bin)
        if ok:
            return arch_bin, "compiled_on_demand"
        ok_def, msg_def = compile_solver(solver, arch, default_bin)
        if ok_def:
            return default_bin, "compiled_default"
        return None, f"Failed compilation: {msg} | {msg_def}"

    if dry_run:
        return arch_bin, "dry_run"

    return None, f"Executable not found for {solver} ({arch}) in {bin_dir}"


def prepare_seed(n: int, work_dir: Path) -> tuple[Path, float]:
    """Create seed text file and return path and baseline certified L."""
    seed_file = work_dir / f"seed{n}.txt"
    rows = tilted_five() if n == 5 else grid(n)
    side = float(certify(n, rows))
    if not seed_file.exists():
        work_dir.mkdir(parents=True, exist_ok=True)
        dilation = 1.00002
        lines = [f"{n} {side * dilation:.17g}\n"]
        for r in rows:
            x_str = format(float(r["x"]) * dilation, ".17g")
            y_str = format(float(r["y"]) * dilation, ".17g")
            a_str = format(float(r["angle"]), ".17g")
            lines.append(f"{x_str} {y_str} {a_str}\n")
        seed_file.write_text("".join(lines))
    return seed_file, side


def run_benchmark_trial(
    executable: Path,
    seed_file: Path,
    n: int,
    epochs: int,
    steps: int,
    chains: int,
    seed: int,
    raw_path: Path,
    baseline_L: float,
    solver: str,
    arch: str,
    device_info: dict,
    dry_run: bool = False,
) -> dict:
    """Execute a single solver benchmark pass and collect latency, throughput, and certified L."""
    t0 = time.perf_counter()
    if dry_run:
        sim_sec = 0.001 * n + 0.0002 * chains * (1.0 if arch == "sm_120" else 1.8)
        sim_props = int(chains * epochs * steps * 0.95)
        sim_throughput = sim_props / sim_sec if sim_sec > 0 else 0.0
        return {
            "solver": solver,
            "arch": arch,
            "device": device_info.get("name", "Simulated"),
            "device_capability": device_info.get("capability", [0, 0]),
            "n": n,
            "chains": chains,
            "epochs": epochs,
            "steps": steps,
            "latency_seconds": round(sim_sec, 6),
            "proposals": sim_props,
            "throughput_proposals_per_sec": round(sim_throughput, 1),
            "best_certified_L": round(baseline_L, 12),
            "baseline_L": round(baseline_L, 12),
            "dilation": 1.000002,
            "status": "certified (simulated)",
        }

    cmd = [
        str(executable),
        str(seed_file),
        str(epochs),
        str(steps),
        str(chains),
        str(seed),
        str(raw_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        wall_time = time.perf_counter() - t0
        raw = json.loads(raw_path.read_text())
        device_sec = float(raw.get("seconds", wall_time))
        proposals = int(raw.get("proposals", chains * epochs * steps))
        throughput = proposals / device_sec if device_sec > 0 else 0.0

        best_L = None
        dilation = None
        status = "rejected"
        try:
            _, certified_L, dil = certified_candidate(raw)
            best_L = float(certified_L)
            dilation = dil
            status = "certified"
        except Exception as exc:
            status = f"rejected ({exc})"

        return {
            "solver": solver,
            "arch": arch,
            "device": raw.get("device", device_info.get("name", "NVIDIA GPU")),
            "device_capability": device_info.get("capability", [0, 0]),
            "n": n,
            "chains": chains,
            "epochs": epochs,
            "steps": steps,
            "latency_seconds": round(device_sec, 6),
            "proposals": proposals,
            "throughput_proposals_per_sec": round(throughput, 1),
            "best_certified_L": best_L,
            "baseline_L": round(baseline_L, 12),
            "dilation": dilation,
            "status": status,
        }
    except Exception as exc:
        return {
            "solver": solver,
            "arch": arch,
            "device": device_info.get("name", "Error"),
            "device_capability": device_info.get("capability", [0, 0]),
            "n": n,
            "chains": chains,
            "epochs": epochs,
            "steps": steps,
            "latency_seconds": round(time.perf_counter() - t0, 6),
            "proposals": 0,
            "throughput_proposals_per_sec": 0.0,
            "best_certified_L": None,
            "baseline_L": round(baseline_L, 12),
            "dilation": None,
            "status": f"error ({exc})",
        }


def format_markdown_table(
    headers: list[str],
    rows: list[list[str]],
    alignments: list[str] | None = None,
) -> str:
    """Format table with GitHub-flavored Markdown."""
    if not alignments:
        alignments = [":---"] * len(headers)
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(val)))
            else:
                col_widths.append(len(str(val)))

    def pad(text: str, width: int, align: str) -> str:
        if align == ":---:":
            return text.center(width)
        if align == "---:":
            return text.rjust(width)
        return text.ljust(width)

    header_line = "| " + " | ".join(pad(headers[i], col_widths[i], alignments[i]) for i in range(len(headers))) + " |"
    sep_line = "| " + " | ".join(
        (":" + "-" * max(1, col_widths[i] - 2) + ":") if alignments[i] == ":---:" else
        ("-" * max(1, col_widths[i] - 1) + ":") if alignments[i] == "---:" else
        (":" + "-" * max(1, col_widths[i] - 1))
        for i in range(len(headers))
    ) + " |"
    data_lines = [
        "| " + " | ".join(pad(str(row[i]), col_widths[i], alignments[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line] + data_lines)


def generate_markdown_reports(benchmarks: list[dict]) -> str:
    """Build comparative Markdown tables for benchmark results."""
    sections = []

    # 1. Primary Full Benchmark Table
    headers = [
        "Solver",
        "Arch",
        "Device",
        "n",
        "Chains",
        "Latency (s)",
        "Throughput (prop/s)",
        "Best Certified L",
        "Baseline L",
        "Status",
    ]
    alignments = [":---", ":---", ":---", ":---:", ":---:", "---:", "---:", "---:", "---:", ":---"]
    rows = []
    for b in benchmarks:
        cert_l_str = f"{b['best_certified_L']:.12g}" if b["best_certified_L"] is not None else "N/A"
        base_l_str = f"{b['baseline_L']:.12g}" if b["baseline_L"] is not None else "N/A"
        th_str = f"{b['throughput_proposals_per_sec']:,.1f}"
        lat_str = f"{b['latency_seconds']:.4f}"
        rows.append([
            b["solver"],
            b["arch"],
            b.get("device", "N/A"),
            str(b["n"]),
            str(b["chains"]),
            lat_str,
            th_str,
            cert_l_str,
            base_l_str,
            b["status"],
        ])
    sections.append("### Detailed Benchmark Results\n\n" + format_markdown_table(headers, rows, alignments))

    # 2. Dual Architecture Comparison (sm_80 vs sm_120)
    archs = {b["arch"] for b in benchmarks}
    if "sm_80" in archs and "sm_120" in archs:
        comp_headers = [
            "Solver",
            "n",
            "Chains",
            "sm_80 Latency (s)",
            "sm_120 Latency (s)",
            "sm_80 Throughput (prop/s)",
            "sm_120 Throughput (prop/s)",
            "Speedup (sm_120 / sm_80)",
            "sm_80 Best L",
            "sm_120 Best L",
        ]
        comp_align = [":---", ":---:", ":---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]
        lookup = {(b["solver"], b["n"], b["chains"], b["arch"]): b for b in benchmarks}
        keys = sorted({(b["solver"], b["n"], b["chains"]) for b in benchmarks})
        comp_rows = []
        for solver, n, chains in keys:
            r80 = lookup.get((solver, n, chains, "sm_80"))
            r120 = lookup.get((solver, n, chains, "sm_120"))
            if r80 and r120:
                t80 = r80["throughput_proposals_per_sec"]
                t120 = r120["throughput_proposals_per_sec"]
                speedup_str = f"{t120 / t80:.2f}x" if t80 > 0 else "N/A"
                l80_str = f"{r80['best_certified_L']:.12g}" if r80["best_certified_L"] is not None else "N/A"
                l120_str = f"{r120['best_certified_L']:.12g}" if r120["best_certified_L"] is not None else "N/A"
                comp_rows.append([
                    solver,
                    str(n),
                    str(chains),
                    f"{r80['latency_seconds']:.4f}",
                    f"{r120['latency_seconds']:.4f}",
                    f"{t80:,.1f}",
                    f"{t120:,.1f}",
                    speedup_str,
                    l80_str,
                    l120_str,
                ])
        if comp_rows:
            sections.append(
                "### Dual-Architecture Comparison: A100 (sm_80) vs Blackwell G4 (sm_120)\n\n"
                + format_markdown_table(comp_headers, comp_rows, comp_align)
            )

    # 3. Solver Comparison (packing vs seeded)
    solvers = {b["solver"] for b in benchmarks}
    if "packing" in solvers and "seeded" in solvers:
        s_headers = [
            "Arch",
            "n",
            "Chains",
            "packing Latency (s)",
            "seeded Latency (s)",
            "packing Throughput",
            "seeded Throughput",
            "packing Best L",
            "seeded Best L",
        ]
        s_align = [":---", ":---:", ":---:", "---:", "---:", "---:", "---:", "---:", "---:"]
        s_lookup = {(b["arch"], b["n"], b["chains"], b["solver"]): b for b in benchmarks}
        s_keys = sorted({(b["arch"], b["n"], b["chains"]) for b in benchmarks})
        s_rows = []
        for arch, n, chains in s_keys:
            rp = s_lookup.get((arch, n, chains, "packing"))
            rs = s_lookup.get((arch, n, chains, "seeded"))
            if rp and rs:
                lp_str = f"{rp['best_certified_L']:.12g}" if rp["best_certified_L"] is not None else "N/A"
                ls_str = f"{rs['best_certified_L']:.12g}" if rs["best_certified_L"] is not None else "N/A"
                s_rows.append([
                    arch,
                    str(n),
                    str(chains),
                    f"{rp['latency_seconds']:.4f}",
                    f"{rs['latency_seconds']:.4f}",
                    f"{rp['throughput_proposals_per_sec']:,.1f}",
                    f"{rs['throughput_proposals_per_sec']:,.1f}",
                    lp_str,
                    ls_str,
                ])
        if s_rows:
            sections.append(
                "### Solver Comparison: packing vs seeded\n\n"
                + format_markdown_table(s_headers, s_rows, s_align)
            )

    return "\n\n".join(sections)


def run_benchmarks(
    solvers: list[str],
    architectures: list[str],
    sizes: list[int],
    chains_list: list[int],
    device_idx: int = 0,
    epochs: int = 20,
    steps: int = 32,
    seed: int = 42,
    work_dir: Path = Path("outputs/bench"),
    bin_dir: Path = Path("cuda"),
    force_compile: bool = False,
    dry_run: bool = False,
) -> list[dict]:
    """Execute all benchmark combinations."""
    dev_info = detect_device(device_idx)
    work_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for arch in architectures:
        for solver in solvers:
            exe, status_msg = resolve_binary(
                solver,
                arch,
                bin_dir,
                compile_if_missing=True,
                force_compile=force_compile,
                dry_run=dry_run,
            )
            if exe is None and not dry_run:
                print(f"[WARN] Skipping {solver} ({arch}): {status_msg}", file=sys.stderr)
                continue

            for n in sizes:
                seed_file, baseline_L = prepare_seed(n, work_dir)
                for chains in chains_list:
                    raw_out = work_dir / f"raw_{solver}_{arch}_n{n}_c{chains}.json"
                    record = run_benchmark_trial(
                        executable=exe or (bin_dir / f"{solver}_{arch}"),
                        seed_file=seed_file,
                        n=n,
                        epochs=epochs,
                        steps=steps,
                        chains=chains,
                        seed=seed,
                        raw_path=raw_out,
                        baseline_L=baseline_L,
                        solver=solver,
                        arch=arch,
                        device_info=dev_info,
                        dry_run=dry_run,
                    )
                    results.append(record)
    return results


def main() -> None:
    p = argparse.ArgumentParser(
        description="Dual-architecture benchmark driver for G4 (sm_120) and A100 (sm_80)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--solver",
        choices=["packing", "seeded", "langevin", "all", "both"],
        default="both",
        help="CUDA solver to benchmark",
    )
    p.add_argument(
        "--sizes",
        type=str,
        default="5,10,11,17,26,50,100",
        help="Comma-separated puzzle sizes n",
    )
    p.add_argument(
        "--chains",
        type=str,
        default="64,256",
        help="Comma-separated replica chain counts",
    )
    p.add_argument(
        "--arch",
        choices=["sm_80", "sm_120", "both", "auto"],
        default="auto",
        help="Target NVCC GPU architecture (sm_80=A100, sm_120=Blackwell G4)",
    )
    p.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index for benchmarking",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Search epochs per benchmark trial",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=32,
        help="Search steps per epoch",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for GPU solvers",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="Recompile solvers with NVCC architecture flags before benchmarking",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate execution without requiring GPU hardware or nvcc",
    )
    p.add_argument(
        "--bin-dir",
        type=Path,
        default=CUDA_DIR,
        help="Directory containing or targeting solver binaries",
    )
    p.add_argument(
        "--work-dir",
        type=Path,
        default=Path("outputs/bench"),
        help="Working directory for seeds and trial outputs",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Export detailed latency, throughput, and certified L to JSON",
    )
    args = p.parse_args()

    dev_info = detect_device(args.device)

    # Determine architectures
    if args.arch == "both":
        architectures = ["sm_80", "sm_120"]
    elif args.arch == "auto":
        detected = dev_info.get("detected_arch", "sm_80")
        architectures = [detected if detected in ("sm_80", "sm_120") else "sm_80"]
    else:
        architectures = [args.arch]

    # Determine solvers
    if args.solver == "all":
        solvers = ["packing", "seeded", "langevin"]
    elif args.solver == "both":
        solvers = ["packing", "seeded"]
    else:
        solvers = [args.solver]

    # Parse sizes and chains
    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    chains_list = [int(x.strip()) for x in args.chains.split(",") if x.strip()]

    benchmarks = run_benchmarks(
        solvers=solvers,
        architectures=architectures,
        sizes=sizes,
        chains_list=chains_list,
        device_idx=args.device,
        epochs=args.epochs,
        steps=args.steps,
        seed=args.seed,
        work_dir=args.work_dir,
        bin_dir=args.bin_dir,
        force_compile=args.compile,
        dry_run=args.dry_run,
    )

    md_report = generate_markdown_reports(benchmarks)
    print(md_report)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "device": dev_info,
            "architectures": architectures,
            "solvers": solvers,
            "sizes": sizes,
            "chains": chains_list,
            "epochs": args.epochs,
            "steps": args.steps,
            "benchmarks": benchmarks,
        }
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
