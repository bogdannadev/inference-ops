#!/usr/bin/env python3
"""Write a compose override that turns r1 into one trial arm.

Usage (from qwen3.6-27b-2-A100/):
  python3 tuning/trial/make_arm.py <arm> [--set name=value]... [--add name]...
          [--drop name]... [--patch-dir DIR] [--l3-dir DIR]

Flag names are given without the leading "--" (argparse would read them as
its own options), e.g. --set max-total-tokens=65536.

The arm is r1's resolved production command (docker compose config) with
flags set, added or dropped. It always gets its own HiCache L3 directory: the
production directory is shared with r0, and a trial (other weights, other KV
scales, synthetic prompts) must never write into r0's cache. The SGLang API key
is put back as ${SGLANG_API_KEY} so no secret lands in the file.

Apply:   COMPOSE_FILE=docker-compose.yml:tuning/trial/<arm>.yml \
           NO_REGISTER=1 ./deploy/roll-replica.sh r1
Restore: NO_REGISTER=0 ./deploy/roll-replica.sh r1   (plain compose = production)
"""
import argparse, json, os, subprocess, sys

SVC = "qwen36-27b-r1"
here = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("arm")
ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
ap.add_argument("--add", action="append", default=[], metavar="NAME")
ap.add_argument("--drop", action="append", default=[], metavar="NAME")
ap.add_argument("--patch-dir", default="./deploy/enginepatch")
ap.add_argument("--l3-dir", default=None)
a = ap.parse_args()

cfg = json.loads(subprocess.check_output(
    ["docker", "compose", "-f", "docker-compose.yml", "config", "--format", "json"]))
cmd = list(cfg["services"][SVC]["command"])

def find(flag):
    return cmd.index(flag) if flag in cmd else -1

def takes_value(i):
    return i + 1 < len(cmd) and not cmd[i + 1].startswith("--")

a.drop = ["--" + n for n in a.drop]
a.add = ["--" + n for n in a.add]
a.set = [("--" + kv.split("=", 1)[0], kv.split("=", 1)[1]) for kv in a.set]

for flag in a.drop:
    i = find(flag)
    if i < 0:
        sys.exit(f"--drop {flag}: not in the production command")
    del cmd[i:i + (2 if takes_value(i) else 1)]
for flag, value in a.set:
    i = find(flag)
    if i >= 0 and takes_value(i):
        cmd[i + 1] = value
    elif i >= 0:
        cmd.insert(i + 1, value)
    else:
        cmd += [flag, value]
for flag in a.add:
    if find(flag) < 0:
        cmd.append(flag)

i = find("--api-key")
if i >= 0:
    cmd[i + 1] = "${SGLANG_API_KEY}"

l3 = a.l3_dir or f"/dev/shm/qwen36-hicache-l3-trial/{a.arm}"
os.makedirs(l3, exist_ok=True)
override = {"services": {SVC: {
    "command": cmd,
    "volumes": [f"{a.patch_dir}:/opt/enginepatch:ro", f"{l3}:/tmp/hicache"],
}}}
out = os.path.join(here, f"{a.arm}.yml")
with open(out, "w") as f:
    json.dump(override, f, indent=2)   # JSON is valid YAML
print(out)
