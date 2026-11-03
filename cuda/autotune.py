"""Finite launch/ladder/move experiments, certified quality before throughput.

Supports multi-GPU benchmarking, device capability detection, and dual-architecture
targeting for sm_80 (A100) and sm_120 (Blackwell G4).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Ensure repository paths are accessible
REPO_ROOT = Path(__file__).resolve().parents[1]
CUDA_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(CUDA_DIR) not in sys.path:
    sys.path.insert(0, str(CUDA_DIR))

from demo import certified_candidate
from polish import polish
from squares.core import certify, write_submission

ARCH_NVCC_FLAGS = {
    "sm_80": ["-gencode", "arch=compute_80,code=sm_80"],
    "sm_120": ["-gencode", "arch=compute_120,code=sm_120"],
}


def detect_devices() -> list[dict]:
    """Detect available CUDA devices for multi-GPU benchmarking."""
    devices = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                cap = torch.cuda.get_device_capability(i)
                major, minor = cap
                arch = "sm_120" if major == 12 else "sm_80" if (major, minor) == (8, 0) else f"sm_{major}{minor}"
                devices.append({
                    "device_idx": i,
                    "name": name,
                    "capability": [major, minor],
                    "arch": arch,
                })
    except Exception:
        pass
    return devices


def prepare(
    directory: Path,
    devices: list[int] | None = None,
    solvers: list[str] | None = None,
    arch: str | None = None,
    threads: tuple[int, ...] = (64, 128, 256),
    chains: tuple[int, ...] = (256, 1024),
    sizes: tuple[int, ...] = (11,),
) -> None:
    """Reproduce finite tuning matrix, optionally distributed across multiple GPUs."""
    directory.mkdir(parents=True, exist_ok=True)
    active_solvers = solvers or ["packing"]

    # When no devices or special options are requested, preserve the exact legacy 12-launch matrix
    if devices is None and active_solvers == ["packing"] and arch is None and sizes == (11,):
        script = ["#!/bin/sh", "set -eu", 'cd "$(dirname "$0")"', "mkdir -p tuning"]
        for t in threads:
            executable = "./packing" if t == 64 else f"./packing{t}"
            for c in chains:
                for name, config in [
                    ("cold", "PACK_TMIN=1e-9 PACK_TMAX=.001 PACK_MOVE=.003 PACK_ROTATION=.35 PACK_SHRINK=.0005"),
                    ("warm", "PACK_TMIN=1e-7 PACK_TMAX=.03 PACK_MOVE=.02 PACK_ROTATION=.8 PACK_SHRINK=.001"),
                ]:
                    script.append(
                        f"{config} {executable} seed11.txt 4000 64 {c} 49 tuning/tune-t{t}-c{c}-{name}.json"
                    )
        script.append("tar -czf tuning.tar.gz tuning")
        (directory / "tune-run.sh").write_text("\n".join(script) + "\n")
        return

    # Multi-GPU / Multi-architecture tuning configuration
    target_devices = devices if devices else [0]
    dev_count = len(target_devices)
    dev_scripts: dict[int, list[str]] = {
        d: ["#!/bin/sh", "set -eu", 'cd "$(dirname "$0")"', "mkdir -p tuning"]
        for d in target_devices
    }

    launch_idx = 0
    configs = [
        ("cold", "PACK_TMIN=1e-9 PACK_TMAX=.001 PACK_MOVE=.003 PACK_ROTATION=.35 PACK_SHRINK=.0005"),
        ("warm", "PACK_TMIN=1e-7 PACK_TMAX=.03 PACK_MOVE=.02 PACK_ROTATION=.8 PACK_SHRINK=.001"),
    ]

    for solver in active_solvers:
        for t in threads:
            base_exe = f"./{solver}" if t == 64 else f"./{solver}{t}"
            if arch and arch in ARCH_NVCC_FLAGS:
                base_exe = f"{base_exe}_{arch}" if Path(f"{base_exe}_{arch}").exists() else base_exe

            for c in chains:
                for n in sizes:
                    for name, cfg in configs:
                        assigned_dev = target_devices[launch_idx % dev_count]
                        out_name = f"tuning/tune-{solver}-gpu{assigned_dev}-t{t}-c{c}-n{n}-{name}.json"
                        cmd = f"CUDA_VISIBLE_DEVICES={assigned_dev} {cfg} {base_exe} seed{n}.txt 4000 64 {c} 49 {out_name}"
                        dev_scripts[assigned_dev].append(cmd)
                        launch_idx += 1

    orchestrator = [
        "#!/bin/sh",
        "set -eu",
        'cd "$(dirname "$0")"',
        "mkdir -p tuning",
    ]

    for d in target_devices:
        dev_file = f"tune-run-gpu{d}.sh"
        dev_scripts[d].append(f"# Finished runs for GPU {d}")
        (directory / dev_file).write_text("\n".join(dev_scripts[d]) + "\n")
        orchestrator.append(f"sh {dev_file} &")

    orchestrator.append("wait")
    orchestrator.append("tar -czf tuning.tar.gz tuning")
    (directory / "tune-run.sh").write_text("\n".join(orchestrator) + "\n")


def format_markdown_table(headers: list[str], rows: list[list[str]], alignments: list[str] | None = None) -> str:
    """Format clean GitHub-flavored Markdown table."""
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


def summarize(
    directory: Path,
    markdown: bool = False,
    json_out: Path | None = None,
) -> list[dict]:
    """Parse raw tuning results, certify candidates, and report comparisons."""
    reports = []
    bests = {}

    for path in sorted(directory.glob("tune*.json")):
        raw = json.loads(path.read_text())
        sec = raw.get("seconds", 1.0)
        chain_steps = raw.get("chain_steps", 0)
        proposals = raw.get("proposals", 0)

        r = dict(
            source=path.name,
            n=raw["n"],
            seconds=sec,
            chain_steps_per_second=chain_steps / sec if sec > 0 else 0.0,
            proposals_per_second=proposals / sec if sec > 0 else 0.0,
            device=raw.get("device", "N/A"),
            chains=raw.get("chains", "N/A"),
            config=path.stem,
        )

        try:
            rows, L, dilation = certified_candidate(raw)
            r.update(certified_L=L, dilation=dilation)
            polished, p = polish(raw["q"], maxiter=600)
            r["polished_L"] = p["L"]
            r["full_precision_L"] = certify(raw["n"], polished)
            if raw["n"] not in bests or p["L"] < float(certify(raw["n"], bests[raw["n"]])):
                bests[raw["n"]] = polished
        except ValueError as e:
            r["rejected"] = str(e)
        reports.append(r)

    reports.sort(key=lambda r: (r.get("polished_L", float("inf")), -r["chain_steps_per_second"]))

    (directory / "report.json").write_text(json.dumps(reports, indent=2) + "\n")
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(reports, indent=2) + "\n")

    if bests:
        write_submission(directory / "certified.csv", bests)

    # Print JSON summary
    print(json.dumps(reports, indent=2), flush=True)

    # Format Markdown comparative table
    headers = [
        "Config",
        "Device",
        "n",
        "Chains",
        "Seconds",
        "Proposals/s",
        "Chain Steps/s",
        "Certified L",
        "Polished L",
        "Status",
    ]
    alignments = [":---", ":---", ":---:", ":---:", "---:", "---:", "---:", "---:", "---:", ":---"]
    rows = []
    for r in reports:
        cert_l_str = f"{r['certified_L']:.12g}" if "certified_L" in r else "N/A"
        pol_l_str = f"{r['polished_L']:.12g}" if "polished_L" in r else "N/A"
        status = "rejected" if "rejected" in r else "certified"
        rows.append([
            r.get("config", r.get("source", "")),
            str(r.get("device", "N/A")),
            str(r.get("n", "")),
            str(r.get("chains", "N/A")),
            f"{r.get('seconds', 0.0):.4f}",
            f"{r.get('proposals_per_second', 0.0):,.1f}",
            f"{r.get('chain_steps_per_second', 0.0):,.1f}",
            cert_l_str,
            pol_l_str,
            status,
        ])

    md_table = "### Autotuning Comparative Benchmark Summary\n\n" + format_markdown_table(headers, rows, alignments)
    if markdown:
        print("\n" + md_table, flush=True)

    return reports


def main() -> None:
    p = argparse.ArgumentParser(
        description="Finite autotuning and multi-GPU benchmarking matrix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("directory", type=Path, help="Directory for tuning runs or summary")
    p.add_argument("--prepare", action="store_true", help="Generate tuning matrix runner scripts")
    p.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Comma-separated GPU device IDs (e.g. '0,1') or 'all' for multi-GPU benchmarking",
    )
    p.add_argument(
        "--solver",
        choices=["packing", "seeded", "both"],
        default="packing",
        help="Solver to benchmark in autotune matrix",
    )
    p.add_argument(
        "--arch",
        choices=["sm_80", "sm_120", "both", "auto"],
        default=None,
        help="Target NVCC GPU architecture (sm_80 for A100, sm_120 for Blackwell G4)",
    )
    p.add_argument(
        "--markdown",
        action="store_true",
        help="Print summary in clean Markdown table format",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Export summary JSON report to specific path",
    )
    a = p.parse_args()

    if a.prepare:
        devices = None
        if a.devices:
            if a.devices.lower() == "all":
                detected = detect_devices()
                devices = [d["device_idx"] for d in detected] if detected else [0]
            else:
                devices = [int(x.strip()) for x in a.devices.split(",") if x.strip()]
        solvers = ["packing", "seeded"] if a.solver == "both" else [a.solver]
        prepare(a.directory, devices=devices, solvers=solvers, arch=a.arch)
    else:
        summarize(a.directory, markdown=a.markdown, json_out=a.json_out)


if __name__ == "__main__":
    main()
