#!/usr/bin/env python3
"""
rocprofv3 Profiler for SGLang on AMD MI350X
Attaches to the running SGLang process and collects hardware PMCs.

Usage:
    ./rocprofv3_profile.py [--duration 30] [--output /path/to/output] [--pmcs basic|full|custom]

PMCs collected (basic set - low overhead):
    GPU_UTIL, OccupancyPercent, MeanOccupancyPerCU, MeanOccupancyPerActiveCU,
    MfmaFlops, MfmaFlopsBF16, MfmaUtil, MemWrites32B, MemUnitStalled,
    GRBM_CPC_BUSY, GRBM_CPF_BUSY, GRBM_TC_BUSY, CU_NUM, SIMD_NUM, SE_NUM

PMCs collected (full set - higher overhead, more detail):
    All basic + SPI_RA_* stalls, CPC/CPF busy/idle/stall, SALUBusy, BANDWIDTH_EA
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

ROCPROFV3 = "/opt/rocm/bin/rocprofv3"

PMCS_BASIC = [
    "GPU_UTIL",
    "OccupancyPercent",
    "MeanOccupancyPerCU",
    "MeanOccupancyPerActiveCU",
    "MfmaFlops",
    "MfmaFlopsBF16",
    "MfmaUtil",
    "MemWrites32B",
    "MemUnitStalled",
    "GRBM_CPC_BUSY",
    "GRBM_CPF_BUSY",
    "GRBM_TC_BUSY",
    "CU_NUM",
    "SIMD_NUM",
    "SE_NUM",
]

PMCS_FULL = PMCS_BASIC + [
    "CPC_CPC_STAT_BUSY",
    "CPC_CPC_STAT_IDLE",
    "CPC_CPC_STAT_STALL",
    "CPF_CPF_STAT_BUSY",
    "CPF_CPF_STAT_IDLE",
    "CPF_CPF_STAT_STALL",
    "SALUBusy",
    "BANDWIDTH_EA",
    "SPI_RA_RES_STALL_CSN",
    "SPI_RA_TMP_STALL_CSN",
    "SPI_RA_WVLIM_STALL_CSN",
    "SPI_RA_TGLIM_CU_FULL_CSN",
    "SPI_RA_SGPR_SIMD_FULL_CSN",
    "SPI_RA_VGPR_SIMD_FULL_CSN",
    "SPI_RA_WAVE_SIMD_FULL_CSN",
    "SPI_CS0_BUSY",
    "SPI_CS1_BUSY",
    "SPI_CS2_BUSY",
    "SPI_CS3_BUSY",
    "SPI_CSN_BUSY",
    "SPI_CSN_WAVE",
    "SPI_CSN_NUM_THREADGROUPS",
    "CPC_CPC_TCIU_BUSY",
    "CPC_CPC_UTCL2IU_BUSY",
    "CPC_CPC_UTCL2IU_STALL",
    "CPF_CPF_TCIU_BUSY",
    "CPF_CPF_TCIU_STALL",
    "GRBM_SPI_BUSY",
    "GRBM_TA_BUSY",
    "GRBM_UTCL2_BUSY",
    "GRBM_EA_BUSY",
    "GRBM_GUI_ACTIVE",
    "GRBM_CP_BUSY",
    "FETCH_SIZE",
    "LdsUtil",
    "MAX_WAVE_SIZE",
]

def find_sglang_pid():
    result = subprocess.run(
        ["amd-smi", "process", "--json"],
        capture_output=True, text=True, timeout=10
    )
    try:
        data = json.loads(result.stdout)
        for pid_str, info in data.get("system", {}).items():
            parts = info.split(", ")
            if len(parts) >= 2 and "sglang" in parts[0].lower():
                pid = pid_str.replace("PID", "")
                vram = float(parts[2]) if parts[2] != "0" else 0
                if vram > 1e9:
                    return int(pid), parts[0], vram
    except (json.JSONDecodeError, KeyError, IndexError):
        pass
    return None, None, 0

def run_profile(pid, duration, pmcs, output_dir):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"rocprof_{ts}")

    cmd = [
        ROCPROFV3,
        "--attach", str(pid),
        "--pmc"] + list(pmcs) + [
        "--output-directory", output_dir,
        "--output-file", output_path,
        "--output-format", "json",
        "--summary",
        "--summary-units", "msec",
        "-P", f"0:{duration}:1",
    ]

    print(f"Attaching rocprofv3 to PID {pid} for {duration}s...")
    print(f"Collecting {len(pmcs)} PMCs: {', '.join(pmcs[:5])}...")
    print(f"Output: {output_path}.json")
    print(f"Command: {' '.join(cmd)}")
    print()

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 60)
    elapsed = time.time() - start

    print(f"Profiling completed in {elapsed:.1f}s")
    print(f"Return code: {result.returncode}")

    if result.stderr:
        print(f"\nStderr:\n{result.stderr[:2000]}")

    output_file = f"{output_path}.json"
    if os.path.exists(output_file):
        print(f"\nOutput file: {output_file}")
        file_size = os.path.getsize(output_file)
        print(f"File size: {file_size / 1024 / 1024:.1f} MB")

        try:
            with open(output_file) as f:
                data = json.load(f)
            print_summary(data)
        except json.JSONDecodeError:
            print("Could not parse output JSON")
    else:
        print(f"Output file not found at {output_file}")
        print("Listing output directory:")
        for f in os.listdir(output_dir):
            if ts in f:
                print(f"  {f} ({os.path.getsize(os.path.join(output_dir, f)) / 1024:.1f} KB)")

    return output_file

def print_summary(data):
    print("\n" + "=" * 60)
    print("ROCprofv3 Summary")
    print("=" * 60)

    if isinstance(data, dict):
        for key, val in data.items():
            if isinstance(val, dict):
                print(f"\n[{key}]")
                for k, v in val.items():
                    if isinstance(v, (int, float)):
                        print(f"  {k}: {v}")
                    elif isinstance(v, str) and len(v) < 100:
                        print(f"  {k}: {v}")
            elif isinstance(val, (int, float)):
                print(f"  {key}: {val}")
            elif isinstance(val, str) and len(val) < 200:
                print(f"  {key}: {val}")

def list_available_pmcs():
    result = subprocess.run(
        ["/opt/rocm/bin/rocprofv3-avail", "list"],
        capture_output=True, text=True, timeout=10
    )
    lines = result.stdout.strip().split("\n")
    pmc_lines = []
    in_pmc = False
    for line in lines:
        if "PMC" in line and ":" in line:
            in_pmc = True
            continue
        if in_pmc and line.strip():
            parts = line.split()
            for p in parts:
                if p.startswith("CPC") or p.startswith("CPF") or p.startswith("GRBM") or \
                   p.startswith("SPI") or p.startswith("GPU") or p.startswith("CU_") or \
                   p.startswith("Mem") or p.startswith("Mfma") or p.startswith("Occupancy") or \
                   p.startswith("SALU") or p.startswith("BANDWIDTH") or p.startswith("Fetch") or \
                   p.startswith("Lds") or p.startswith("MAX") or p.startswith("Mean") or \
                   p.startswith("SE_") or p.startswith("SIMD") or p.startswith("TA_") or \
                   p.startswith("PA_") or p.startswith("SQ_"):
                    pmc_lines.append(p)
    print(f"Available relevant PMCs ({len(pmc_lines)}):")
    for p in sorted(pmc_lines):
        print(f"  {p}")

def main():
    parser = argparse.ArgumentParser(description="rocprofv3 Profiler for SGLang on AMD MI350X")
    parser.add_argument("--duration", type=int, default=30, help="Profiling duration in seconds (default: 30)")
    parser.add_argument("--output", type=str, default="/tmp/rocprof_output", help="Output directory")
    parser.add_argument("--pmcs", type=str, choices=["basic", "full", "list"], default="basic",
                        help="PMC set: basic (low overhead), full (detailed), list (show available)")
    parser.add_argument("--pid", type=int, default=0, help="Target PID (auto-detect SGLang if 0)")
    parser.add_argument("--custom-pmcs", type=str, default="", help="Comma-separated custom PMC list")
    args = parser.parse_args()

    if args.pmcs == "list":
        list_available_pmcs()
        return

    if args.custom_pmcs:
        pmcs = [p.strip() for p in args.custom_pmcs.split(",")]
    elif args.pmcs == "full":
        pmcs = PMCS_FULL
    else:
        pmcs = PMCS_BASIC

    pid = args.pid
    if not pid:
        pid, name, vram = find_sglang_pid()
        if not pid:
            print("ERROR: Could not find SGLang process. Use --pid to specify.")
            sys.exit(1)
        print(f"Found SGLang process: {name} (PID {pid}, VRAM: {vram/1e9:.1f} GB)")

    os.makedirs(args.output, exist_ok=True)
    run_profile(pid, args.duration, pmcs, args.output)

if __name__ == "__main__":
    main()
