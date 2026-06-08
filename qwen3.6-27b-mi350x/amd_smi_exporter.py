#!/usr/bin/env python3
"""
amd-smi Prometheus Exporter
Scrapes amd-smi metric --json and exposes metrics on /metrics endpoint.
Designed for MI300X/MI350X with multiple XCCs.
"""

import json
import subprocess
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9300

def run_amd_smi(args):
    try:
        result = subprocess.run(
            ["amd-smi"] + args + ["--json"],
            capture_output=True, text=True, timeout=10
        )
        return json.loads(result.stdout)
    except Exception as e:
        print(f"Error running amd-smi: {e}", file=sys.stderr)
        return None

def extract_float(val, default=0.0):
    if val is None or val == "N/A":
        return default
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, dict):
        return extract_float(val.get("value"), default)
    try:
        return float(str(val).replace('"', ''))
    except (ValueError, TypeError):
        return default

def format_metrics(data):
    lines = []
    if not data or "gpu_data" not in data:
        return "# error\namd_smi_scrape_error 1\n"

    for gpu in data["gpu_data"]:
        gid = gpu.get("gpu", 0)
        labels = f'gpu="{gid}"'

        # GPU Usage
        usage = gpu.get("usage", {})
        gfx_activity = extract_float(usage.get("gfx_activity"))
        umc_activity = extract_float(usage.get("umc_activity"))
        lines.append(f'# HELP amd_gpu_gfx_activity_percent GPU graphics activity percentage')
        lines.append(f'# TYPE amd_gpu_gfx_activity_percent gauge')
        lines.append(f'amd_gpu_gfx_activity_percent{{{labels}}} {gfx_activity}')

        lines.append(f'# HELP amd_gpu_umc_activity_percent GPU memory controller activity percentage')
        lines.append(f'# TYPE amd_gpu_umc_activity_percent gauge')
        lines.append(f'amd_gpu_umc_activity_percent{{{labels}}} {umc_activity}')

        # GFX busy per XCC
        gfx_busy = usage.get("gfx_busy_inst", {})
        for xcc, engines in gfx_busy.items():
            if isinstance(engines, list):
                for i, eng in enumerate(engines):
                    val = extract_float(eng)
                    xcc_labels = f'gpu="{gid}",xcc="{xcc}",engine="{i}"'
                    lines.append(f'# HELP amd_gpu_gfx_busy_inst_percent Per-engine GFX busy percentage')
                    lines.append(f'# TYPE amd_gpu_gfx_busy_inst_percent gauge')
                    lines.append(f'amd_gpu_gfx_busy_inst_percent{{{xcc_labels}}} {val}')

        # Power
        power = gpu.get("power", {})
        socket_power = extract_float(power.get("socket_power"))
        lines.append(f'# HELP amd_gpu_socket_power_watts Current socket power in watts')
        lines.append(f'# TYPE amd_gpu_socket_power_watts gauge')
        lines.append(f'amd_gpu_socket_power_watts{{{labels}}} {socket_power}')

        lines.append(f'# HELP amd_gpu_power_management_enabled Power management enabled (1/0)')
        lines.append(f'# TYPE amd_gpu_power_management_enabled gauge')
        lines.append(f'amd_gpu_power_management_enabled{{{labels}}} {"1" if power.get("power_management") == "ENABLED" else "0"}')

        # Clocks - per GFX engine
        clocks = gpu.get("clock", {})
        for key, clk_data in clocks.items():
            if isinstance(clk_data, dict) and "clk" in clk_data:
                clk_val = extract_float(clk_data.get("clk"))
                min_clk = extract_float(clk_data.get("min_clk"))
                max_clk = extract_float(clk_data.get("max_clk"))
                clk_labels = f'gpu="{gid}",clock="{key}"'

                lines.append(f'# HELP amd_gpu_clock_mhz Current clock frequency in MHz')
                lines.append(f'# TYPE amd_gpu_clock_mhz gauge')
                lines.append(f'amd_gpu_clock_mhz{{{clk_labels}}} {clk_val}')

                lines.append(f'# HELP amd_gpu_clock_min_mhz Minimum clock frequency in MHz')
                lines.append(f'# TYPE amd_gpu_clock_min_mhz gauge')
                lines.append(f'amd_gpu_clock_min_mhz{{{clk_labels}}} {min_clk}')

                lines.append(f'# HELP amd_gpu_clock_max_mhz Maximum clock frequency in MHz')
                lines.append(f'# TYPE amd_gpu_clock_max_mhz gauge')
                lines.append(f'amd_gpu_clock_max_mhz{{{clk_labels}}} {max_clk}')

        # Temperature
        temp = gpu.get("temperature", {})
        hotspot = extract_float(temp.get("hotspot"))
        mem_temp = extract_float(temp.get("mem"))
        lines.append(f'# HELP amd_gpu_temperature_celsius GPU temperature in celsius')
        lines.append(f'# TYPE amd_gpu_temperature_celsius gauge')
        lines.append(f'amd_gpu_temperature_celsius{{{labels},sensor="hotspot"}} {hotspot}')
        lines.append(f'amd_gpu_temperature_celsius{{{labels},sensor="memory"}} {mem_temp}')

        # GPU Board temperatures
        gpu_board = gpu.get("gpu_board", {})
        board_temp = gpu_board.get("temperature", {})
        for sensor_name, sensor_val in board_temp.items():
            val = extract_float(sensor_val)
            sensor_labels = f'gpu="{gid}",sensor="{sensor_name}"'
            lines.append(f'# HELP amd_gpu_board_temperature_celsius GPU board component temperature')
            lines.append(f'# TYPE amd_gpu_board_temperature_celsius gauge')
            lines.append(f'amd_gpu_board_temperature_celsius{{{sensor_labels}}} {val}')

        # PCIe
        pcie = gpu.get("pcie", {})
        pcie_width = extract_float(pcie.get("width"))
        pcie_speed = extract_float(pcie.get("speed"))
        pcie_bandwidth = extract_float(pcie.get("bandwidth"))
        pcie_replay = extract_float(pcie.get("replay_count"))

        lines.append(f'# HELP amd_gpu_pcie_width Current PCIe lane width')
        lines.append(f'# TYPE amd_gpu_pcie_width gauge')
        lines.append(f'amd_gpu_pcie_width{{{labels}}} {pcie_width}')

        lines.append(f'# HELP amd_gpu_pcie_speed_gts Current PCIe speed in GT/s')
        lines.append(f'# TYPE amd_gpu_pcie_speed_gts gauge')
        lines.append(f'amd_gpu_pcie_speed_gts{{{labels}}} {pcie_speed}')

        lines.append(f'# HELP amd_gpu_pcie_bandwidth_mbps Current PCIe bandwidth in Mb/s')
        lines.append(f'# TYPE amd_gpu_pcie_bandwidth_mbps gauge')
        lines.append(f'amd_gpu_pcie_bandwidth_mbps{{{labels}}} {pcie_bandwidth}')

        lines.append(f'# HELP amd_gpu_pcie_replay_count PCIe replay counter')
        lines.append(f'# TYPE amd_gpu_pcie_replay_count counter')
        lines.append(f'amd_gpu_pcie_replay_count{{{labels}}} {pcie_replay}')

        # ECC
        ecc = gpu.get("ecc", {})
        ecc_correctable = extract_float(ecc.get("total_correctable_count"))
        ecc_uncorrectable = extract_float(ecc.get("total_uncorrectable_count"))
        ecc_deferred = extract_float(ecc.get("total_deferred_count"))

        lines.append(f'# HELP amd_gpu_ecc_correctable_count Total correctable ECC errors')
        lines.append(f'# TYPE amd_gpu_ecc_correctable_count counter')
        lines.append(f'amd_gpu_ecc_correctable_count{{{labels}}} {ecc_correctable}')

        lines.append(f'# HELP amd_gpu_ecc_uncorrectable_count Total uncorrectable ECC errors')
        lines.append(f'# TYPE amd_gpu_ecc_uncorrectable_count counter')
        lines.append(f'amd_gpu_ecc_uncorrectable_count{{{labels}}} {ecc_uncorrectable}')

        lines.append(f'# HELP amd_gpu_ecc_deferred_count Total deferred ECC errors')
        lines.append(f'# TYPE amd_gpu_ecc_deferred_count counter')
        lines.append(f'amd_gpu_ecc_deferred_count{{{labels}}} {ecc_deferred}')

        # ECC per block
        ecc_blocks = gpu.get("ecc_blocks", {})
        for block_name, block_data in ecc_blocks.items():
            if isinstance(block_data, dict):
                block_labels = f'gpu="{gid}",block="{block_name}"'
                corr = extract_float(block_data.get("correctable_count"))
                uncorr = extract_float(block_data.get("uncorrectable_count"))
                deferred = extract_float(block_data.get("deferred_count"))

                lines.append(f'# HELP amd_gpu_ecc_block_correctable_count ECC correctable errors per block')
                lines.append(f'# TYPE amd_gpu_ecc_block_correctable_count counter')
                lines.append(f'amd_gpu_ecc_block_correctable_count{{{block_labels}}} {corr}')

                lines.append(f'# HELP amd_gpu_ecc_block_uncorrectable_count ECC uncorrectable errors per block')
                lines.append(f'# TYPE amd_gpu_ecc_block_uncorrectable_count counter')
                lines.append(f'amd_gpu_ecc_block_uncorrectable_count{{{block_labels}}} {uncorr}')

                lines.append(f'# HELP amd_gpu_ecc_block_deferred_count ECC deferred errors per block')
                lines.append(f'# TYPE amd_gpu_ecc_block_deferred_count counter')
                lines.append(f'amd_gpu_ecc_block_deferred_count{{{block_labels}}} {deferred}')

        # Voltage
        voltage = gpu.get("voltage", {})
        for vkey, vdata in voltage.items():
            val = extract_float(vdata)
            vlabels = f'gpu="{gid}",voltage="{vkey}"'
            lines.append(f'# HELP amd_gpu_voltage_mv GPU voltage in millivolts')
            lines.append(f'# TYPE amd_gpu_voltage_mv gauge')
            lines.append(f'amd_gpu_voltage_mv{{{vlabels}}} {val}')

        # Memory Usage
        mem = gpu.get("mem_usage", {})
        total_vram = extract_float(mem.get("total_vram"))
        used_vram = extract_float(mem.get("used_vram"))
        free_vram = extract_float(mem.get("free_vram"))
        total_gtt = extract_float(mem.get("total_gtt"))
        used_gtt = extract_float(mem.get("used_gtt"))
        free_gtt = extract_float(mem.get("free_gtt"))

        lines.append(f'# HELP amd_gpu_vram_bytes_total Total VRAM in bytes')
        lines.append(f'# TYPE amd_gpu_vram_bytes_total gauge')
        lines.append(f'amd_gpu_vram_bytes_total{{{labels}}} {total_vram * 1024 * 1024}')

        lines.append(f'# HELP amd_gpu_vram_bytes_used Used VRAM in bytes')
        lines.append(f'# TYPE amd_gpu_vram_bytes_used gauge')
        lines.append(f'amd_gpu_vram_bytes_used{{{labels}}} {used_vram * 1024 * 1024}')

        lines.append(f'# HELP amd_gpu_vram_bytes_free Free VRAM in bytes')
        lines.append(f'# TYPE amd_gpu_vram_bytes_free gauge')
        lines.append(f'amd_gpu_vram_bytes_free{{{labels}}} {free_vram * 1024 * 1024}')

        lines.append(f'# HELP amd_gpu_vram_usage_ratio VRAM usage ratio (0-1)')
        lines.append(f'# TYPE amd_gpu_vram_usage_ratio gauge')
        lines.append(f'amd_gpu_vram_usage_ratio{{{labels}}} {used_vram / total_vram if total_vram > 0 else 0}')

        lines.append(f'# HELP amd_gpu_gtt_bytes_total Total GTT aperture in bytes')
        lines.append(f'# TYPE amd_gpu_gtt_bytes_total gauge')
        lines.append(f'amd_gpu_gtt_bytes_total{{{labels}}} {total_gtt * 1024 * 1024}')

        lines.append(f'# HELP amd_gpu_gtt_bytes_used Used GTT aperture in bytes')
        lines.append(f'# TYPE amd_gpu_gtt_bytes_used gauge')
        lines.append(f'amd_gpu_gtt_bytes_used{{{labels}}} {used_gtt * 1024 * 1024}')

    return "\n".join(lines) + "\n"

class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/metrics":
            data = run_amd_smi(["metric"])
            body = format_metrics(data)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.end_headers()
            self.wfile.write(body.encode())
        elif parsed.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK\n")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), MetricsHandler)
    print(f"amd-smi exporter listening on :{PORT}/metrics", flush=True)
    server.serve_forever()
