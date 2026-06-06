# Qwen3.6-27B-FP8 on MI350X — Deployment, Metrics & Kernel-Profiling Manual

A complete, ordered guide for serving `Qwen/Qwen3.6-27B-FP8` on a single AMD
Instinct MI350X (DigitalOcean GPU Droplet, ROCm 7.2.0) with SGLang, a full
telemetry ecosystem, and a separate kernel-profiling deployment profile.

The manual is split into three parts that map onto three *tiers of
observability*. Read the conceptual note below before you start — it explains
why the profiling work is a separate deployment rather than an extra metric, and
keeps you from going down the wrong path later.

---

## 0. Conceptual orientation (read first)

You are building two things that look similar but are fundamentally different
classes of instrument, and a third tier you reach for only on demand.

1. **Serving** — the SGLang inference server itself, tuned for throughput and
   latency: HIP graphs on, speculative decoding on, concurrency high.

2. **Telemetry tier** (always-on, cheap) — the AMD Device Metrics Exporter →
   Prometheus → Grafana stack. It samples *board-level aggregates* (GPU-busy %,
   HBM usage/bandwidth, power, clocks, temperature, PCIe, RAS errors) every few
   seconds. It answers *"is the system healthy, and is anything in the macro
   envelope sabotaging my measurement?"* It does **not** and **cannot** expose
   compute-unit occupancy, LDS, or L1/L2 cache behaviour — those are hardware
   performance counters read per-kernel through a profiler, an entirely
   separate path from the `amd-smi` management interface the exporter reads.

3. **Profiling tier** (on-demand, invasive) — `rocprofv3` /
   `rocprof-compute` (ROCm Compute Profiler, formerly Omniperf) /
   `rocprof-systems` (formerly Omnitrace), plus SGLang's built-in Torch
   Profiler. This is where CU occupancy, LDS, L1 (vL1D), and L2 (TCC) cache
   metrics actually come from. Because counter collection replays each kernel
   multiple times, you never point it at live production traffic — you run it
   against a bounded one-shot, in a deployment configured specifically to make
   kernels visible.

The single most important consequence: **everything that makes the serving
config fast makes it opaque to a kernel profiler.** HIP graphs hide individual
dispatches; speculative decoding interleaves draft/verify kernels; concurrency
overlaps dozens of requests on one timeline. The profiling deployment undoes all
three. That is the whole reason Part C exists as its own config rather than a
flag you toggle.

### Hardware context

- **GPU:** 1× AMD Instinct MI350X, CDNA4 / `gfx950`, 288 GB HBM3E, 8 TB/s.
  Per-CU 32 KB L1, 4 MB shared L2, 256 MB Infinity Cache across 8 XCDs.
- **Host:** DigitalOcean GPU Droplet, Atlanta (ATL1), ROCm 7.2.0.
- **Model:** `Qwen/Qwen3.6-27B-FP8` — dense 27B, hybrid Gated Delta Network
  attention, block-FP8 (block size 128), ~31 GB, native 262K context, MTP
  (NEXTN) supported. Apache 2.0.
- **Why TP=1:** a 31 GB model on a 288 GB card fits with ~250 GB free; single
  GPU avoids all-reduce entirely. Shard only if throughput demands it.

---

# PART A — Serving + Telemetry Ecosystem

## A1. Directory scaffold

The compose bind-mounts specific files and folders. They must exist *before*
`docker compose up`, otherwise Docker silently creates a **directory** where a
file was expected (the classic bind-mount footgun) and the dependent service
fails cryptically. Run this from inside your working directory.

```bash
mkdir -p logs \
         prometheus \
         grafana/provisioning/datasources \
         grafana/provisioning/dashboards/json \
         profiling \
         sglang_cache_qwen36_27b \
         triton_cache_qwen36_27b
```

Resulting layout:

```
.
├── docker-compose.yml
├── docker-compose.profiling.yml
├── .env
├── logs/
├── profiling/                              # profiler output lands here
├── prometheus/
│   └── prometheus.yml
├── grafana/
│   └── provisioning/
│       ├── datasources/prometheus.yml
│       └── dashboards/
│           ├── provider.yml
│           └── json/                        # AMD dashboard JSONs
├── sglang_cache_qwen36_27b/
└── triton_cache_qwen36_27b/
```

## A2. The serving compose file

Save as `docker-compose.yml`. Note the **rocm720** image tag (your host is
7.2.0 — the container ROCm userspace must match the host amdgpu kernel driver).
Verify the exact tag against <https://hub.docker.com/r/lmsysorg/sglang-rocm/tags>
before pulling; the version prefix moves, but the `-rocm720-mi35x` suffix is the
constant that guarantees gfx950 + ROCm 7.2 compatibility.

```yaml
# =============================================================================
# QWEN3.6-27B-FP8 — SGLANG ROCm TP=1 (MI350X / CDNA4, full observability stack)
# =============================================================================
# Hardware:  1× AMD Instinct MI350X (288 GB HBM3E, CDNA4 / gfx950)
# Model:     Qwen/Qwen3.6-27B-FP8 (~31 GB, block-FP8 g128, dense hybrid GDN)
# Stack:     SGLang (rocm720-mi35x) + AMD Device Metrics Exporter
#            + Prometheus + Grafana
#
# ASSUMPTIONS (flip if wrong):
#   - Single GPU, TP=1     model fits with ~250 GB free; no all-reduce path
#   - ROCm 7.2 host        rocm720 image tag
#   - Text-only serving    vision tower loads but is never exercised
#   - FP8 KV cache OFF     288 GB makes it pointless; bf16 KV preserves quality
#   - torch.compile OFF    stabilise first, then test as a single variable
# =============================================================================

services:

  # ---------------------------------------------------------------------------
  # Inference server.
  # SYS_PTRACE + unconfined seccomp are deliberate: they let you attach the
  # ROCm profilers inside the running container without recreating the service.
  # (Hardware-counter collection needs SYS_ADMIN too — added in the profiling
  # overlay, Part C, not here.)
  # ---------------------------------------------------------------------------
  qwen36-27b-fp8-sglang:
    image: lmsysorg/sglang-rocm:v0.5.12.post1-rocm720-mi35x-20260603
    container_name: qwen36-27b-fp8-sglang
    ipc: host
    shm_size: "16g"

    devices:
      - /dev/kfd
      - /dev/dri
    group_add:
      - video
      - render
    security_opt:
      - seccomp=unconfined
    cap_add:
      - SYS_PTRACE

    ports:
      - "8002:8002"

    volumes:
      - ${HF_HOME:-~/.cache/huggingface}:/root/.cache/huggingface
      - ./logs:/app/logs
      - /etc/localtime:/etc/localtime:ro
      - ./sglang_cache_qwen36_27b:/root/.cache/torch
      - ./triton_cache_qwen36_27b:/root/.triton

    environment:
      - HF_TOKEN=${HF_TOKEN:-}
      - PYTHONUNBUFFERED=1
      - SGLANG_API_KEY=${SGLANG_API_KEY:-}
      # Pin to the single MI350X; extend/drop for multi-GPU TP.
      - HIP_VISIBLE_DEVICES=0
      # AMD-optimised GEMM/attention kernels.
      - SGLANG_USE_AITER=1
      # Specv2 draft/verify overlap — what makes NEXTN MTP pay off on ROCm.
      - SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
      - PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

    entrypoint: python3 -m sglang.launch_server
    command: >
      --model-path Qwen/Qwen3.6-27B-FP8
      --port 8002
      --host 0.0.0.0
      --served-model-name qwen36-27b-fp8
      --trust-remote-code
      --reasoning-parser qwen3
      --tool-call-parser qwen3_coder
      --tp 1
      --context-length 262144
      --mem-fraction-static 0.9
      --chunked-prefill-size 16384
      --max-running-requests 32
      --page-size 64
      --attention-backend triton
      --mamba-backend triton
      --mamba-ssm-dtype bfloat16
      --speculative-algo NEXTN
      --speculative-num-steps 3
      --speculative-eagle-topk 1
      --speculative-num-draft-tokens 4
      --schedule-policy lpm
      --grammar-backend xgrammar
      --allow-auto-truncate
      --enable-metrics
      --enable-cache-report
      --log-requests
      --log-requests-level 2
      --watchdog-timeout 1200
      --api-key ${SGLANG_API_KEY}

    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8002/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 1200s

    logging:
      driver: "json-file"
      options:
        max-size: "50m"
        max-file: "5"

    restart: unless-stopped

  # ---------------------------------------------------------------------------
  # AMD GPU telemetry -> Prometheus format. The DCGM-exporter analogue.
  # privileged may be needed for full RAS/CPER depth — start with SYS_ADMIN.
  # ---------------------------------------------------------------------------
  device-metrics-exporter:
    image: rocm/device-metrics-exporter:v1.5.0
    container_name: amd-device-metrics-exporter
    devices:
      - /dev/kfd
      - /dev/dri
    group_add:
      - video
      - render
    cap_add:
      - SYS_ADMIN
    ports:
      - "5000:5000"
    restart: unless-stopped

  # ---------------------------------------------------------------------------
  # Scrapes the inference plane and the hardware plane onto one timeline.
  # ---------------------------------------------------------------------------
  prometheus:
    image: prom/prometheus:latest    # pin to a release tag for production
    container_name: prometheus
    ports:
      - "127.0.0.1:9090:9090"        # localhost-only; reach via SSH tunnel
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - --config.file=/etc/prometheus/prometheus.yml
      - --storage.tsdb.retention.time=30d
    depends_on:
      - qwen36-27b-fp8-sglang
      - device-metrics-exporter
    restart: unless-stopped

  # ---------------------------------------------------------------------------
  # Visualisation. Loads the AMD dashboards provisioned in grafana/.../json/.
  # ---------------------------------------------------------------------------
  grafana:
    image: grafana/grafana:latest    # pin for production
    container_name: grafana
    ports:
      - "127.0.0.1:3000:3000"        # localhost-only; reach via SSH tunnel
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:

networks:
  default:
    name: qwen36-27b-mi350x-network
```

> **Security note.** Prometheus (9090) and Grafana (3000) are bound to
> `127.0.0.1` above so they are not exposed on the droplet's public IP. The
> exporter (5000) and SGLang (8002) are reachable on the Docker network for
> internal scraping/serving; if you need SGLang externally, front it with the
> DigitalOcean cloud firewall and rely on its `--api-key`. Reach the dashboards
> over an SSH tunnel (Step A8).

## A3. Prometheus scrape config

Save as `prometheus/prometheus.yml`. The hardware job scrapes faster (5 s) for
tighter correlation against kernel-level events; the inference job stays at the
15 s default.

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  # Inference plane: TTFT, TPOT, throughput, KV pressure, spec-decode accept len.
  - job_name: sglang
    metrics_path: /metrics
    static_configs:
      - targets: ["qwen36-27b-fp8-sglang:8002"]
    # If /metrics sits behind the API key and the target shows DOWN with 401,
    # uncomment and supply the key:
    # authorization:
    #   type: Bearer
    #   credentials: "YOUR_SGLANG_API_KEY"

  # Hardware plane: finer interval for kernel-event correlation.
  - job_name: amd_gpu
    scrape_interval: 5s
    static_configs:
      - targets: ["device-metrics-exporter:5000"]

  - job_name: prometheus
    static_configs:
      - targets: ["localhost:9090"]
```

## A4. Grafana provisioning

Datasource — save as `grafana/provisioning/datasources/prometheus.yml`:

```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

Dashboard provider — save as `grafana/provisioning/dashboards/provider.yml`.
It tells Grafana to load any dashboard JSON found in the `json/` subfolder at
boot:

```yaml
apiVersion: 1
providers:
  - name: amd-gpu
    type: file
    disableDeletion: false
    editable: true
    options:
      # path inside the container; matches the read-only mount in compose
      path: /etc/grafana/provisioning/dashboards/json
```

Fetch AMD's prebuilt dashboards into that folder (shallow clone is robust
against filename changes):

```bash
git clone --depth 1 https://github.com/ROCm/device-metrics-exporter /tmp/dme
cp /tmp/dme/grafana/*.json grafana/provisioning/dashboards/json/
rm -rf /tmp/dme
```

## A5. Secrets file

Generate a real API key, then write `.env` (compose reads it automatically).
`HF_HOME` must point at a disk with ~40 GB free — check `df -h` first; on a
droplet, prefer an attached data volume over the boot disk.

```bash
# generate a strong key for the line below
openssl rand -hex 32

cat > .env <<'EOF'
HF_TOKEN=hf_your_token_here
SGLANG_API_KEY=paste_openssl_output
GRAFANA_PASSWORD=choose_something
HF_HOME=/root/.cache/huggingface
EOF
chmod 600 .env
```

## A6. Pre-flight verification

Fail cheaply and early. Each check below rules out a whole class of later
confusion.

```bash
# 1. Host sees the card, you are in the right groups, device nodes exist.
amd-smi list                       # must show the MI350X
groups                             # expect: ... video ... render ...
ls -l /dev/kfd /dev/dri/renderD*

#    If video/render are missing, add and re-login (won't apply to this shell):
#    sudo usermod -aG video,render $USER

# 2. Prove passthrough + tag + version THROUGH Docker with the real image.
TAG=lmsysorg/sglang-rocm:v0.5.12.post1-rocm720-mi35x-20260603
docker run --rm --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render "$TAG" amd-smi list
docker run --rm --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render "$TAG" \
  python3 -c "import sglang; print(sglang.__version__)"   # confirm real version
```

## A7. Pre-stage the model

Downloading 31 GB while the healthcheck clock ticks is needless stress. Fetch it
first into the same host path the compose mounts, so first boot is a disk read.

```bash
export HF_HOME=/root/.cache/huggingface
export HF_TOKEN=hf_your_token_here
huggingface-cli download Qwen/Qwen3.6-27B-FP8
```

## A8. Staged bring-up

Two layers of staging. First the cheap observability trio, then SGLang — and for
SGLang's *first* boot, **without speculative decoding**, so that if anything
breaks you can isolate the model from the newest, least-proven path.

```bash
# 1. Observability trio (instant). Confirm GPU metrics flow.
docker compose up -d device-metrics-exporter prometheus grafana
docker compose ps
curl -s localhost:5000/metrics | head        # GPU telemetry, raw text

# 2. FIRST SGLang boot WITHOUT spec-decode:
#    comment out the four --speculative-* lines in docker-compose.yml, then:
docker compose up -d qwen36-27b-fp8-sglang
docker compose logs -f qwen36-27b-fp8-sglang  # watch load + HIP graph capture
```

Validate base inference once the healthcheck passes:

```bash
source .env
curl -s localhost:8002/health
curl -s -H "Authorization: Bearer $SGLANG_API_KEY" localhost:8002/v1/models
curl -s -H "Authorization: Bearer $SGLANG_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"qwen36-27b-fp8","messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":16}' \
  localhost:8002/v1/chat/completions
```

Re-enable speculative decoding as a *separate* change (uncomment the four flags),
recreate the service, and confirm it still boots:

```bash
docker compose up -d qwen36-27b-fp8-sglang    # recreates with the new command
docker compose logs -f qwen36-27b-fp8-sglang
```

Confirm both telemetry planes report, and that spec-decode is *measurable*:

```bash
# Inference plane — the accept-length gauge is your spec-decode tuning signal.
curl -s -H "Authorization: Bearer $SGLANG_API_KEY" localhost:8002/metrics | grep -i accept
# Hardware plane.
curl -s localhost:5000/metrics | grep -i gpu
```

Reach the dashboards safely from your workstation via SSH tunnel:

```bash
ssh -L 3000:localhost:3000 -L 9090:localhost:9090 user@your-droplet
# then browse  http://localhost:3000   (admin / GRAFANA_PASSWORD)
#              http://localhost:9090/targets   (sglang + amd_gpu should be UP)
```

At this point **Part A is complete**: the model serves with speculative
decoding, and both telemetry planes feed Grafana.

---

# PART B — Using the Telemetry Tier

This tier is for operations and for *guarding* profiling runs, not for kernel
analysis. Two reads matter most:

- **Throttle detection.** A decode-throughput dip that lines up with a junction-
  temperature spike or a clock drop is thermal throttling, not a kernel problem.
  Always check this before trusting a profiling number — a capture taken during
  a throttle event is worthless.
- **RAS / AFID errors.** An uncorrected ECC event mid-benchmark invalidates the
  run. The exporter exposes these as labelled counters; alert on them.

Key inference-plane series to watch in Grafana (from SGLang `/metrics`):
time-to-first-token, time-per-output-token, running/waiting request counts, KV
pool usage, and the speculative-decode acceptance length. Remember your prior
caveat: per-response `cached_tokens` reads 0 under NEXTN (SGLang #20451) — trust
the `/metrics` gauge, not the per-response field.

---

# PART C — Kernel-Profiling Tier

This is where CU occupancy, LDS, and L1/L2 cache metrics come from. It is a
*separate deployment profile* layered over the base compose, plus on-demand
profiler runs. The serving stack and telemetry ecosystem stay untouched.

## C1. Host-side prerequisite (cannot be set from a container)

Hardware-counter and PAPI collection require the host kernel to permit it. Set
the paranoia level to ≤ 2 on the droplet (not persisted across reboot):

```bash
echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid
```

## C2. The profiling overlay

Save as `docker-compose.profiling.yml`. It reuses the base service definition
and overrides only what profiling needs. The critical additions and changes:

- **`SYS_ADMIN` capability** — required (on top of `SYS_PTRACE` + unconfined
  seccomp already in the base) for the profiler to read hardware counters.
- **`--disable-cuda-graph`** — surfaces each kernel as its own HIP dispatch so
  the profiler can attribute counters per kernel. Without this you get one
  opaque graph blob.
- **speculative decoding removed** — profile the clean base forward pass first.
- **`--max-running-requests 1`** — single-stream, deterministic attribution.
- **context trimmed** — fast, repeatable dispatches.
- **`./profiling` mounted at `/workdir`** — profiler output persists on the host.

```yaml
# =============================================================================
# docker-compose.profiling.yml  —  kernel-profiling overlay (NOT for serving)
# Usage:
#   docker compose -f docker-compose.yml -f docker-compose.profiling.yml \
#     up -d qwen36-27b-fp8-sglang
# =============================================================================
services:
  qwen36-27b-fp8-sglang:

    # SYS_ADMIN unlocks hardware-counter reads (the rest is in the base file).
    cap_add:
      - SYS_PTRACE
      - SYS_ADMIN

    # Persist profiler output + give the Torch Profiler a trace directory.
    volumes:
      - ./profiling:/workdir
    environment:
      - SGLANG_TORCH_PROFILER_DIR=/workdir
      - SGLANG_PROFILE_WITH_STACK=False   # avoids a known torch.profiler hang

    # Clean, single-stream, graph-free capture command.
    command: >
      --model-path Qwen/Qwen3.6-27B-FP8
      --port 8002
      --host 0.0.0.0
      --served-model-name qwen36-27b-fp8-profiling
      --trust-remote-code
      --reasoning-parser qwen3
      --tp 1
      --context-length 32768
      --mem-fraction-static 0.9
      --max-running-requests 1
      --page-size 64
      --attention-backend triton
      --mamba-backend triton
      --mamba-ssm-dtype bfloat16
      --disable-cuda-graph
      --grammar-backend xgrammar
      --watchdog-timeout 1200
      --api-key ${SGLANG_API_KEY}
```

## C3. Tool availability check

The inference image ships `rocprofv3` (part of the ROCm base), but the Python
analysis tool `rocprof-compute` may be absent from a slim runtime image. Check:

```bash
docker compose -f docker-compose.yml -f docker-compose.profiling.yml \
  exec qwen36-27b-fp8-sglang bash -lc 'which rocprofv3 rocprof-compute || true'
```

If `rocprof-compute` is missing, you have two options: `pip install` it inside
the container, or — cleaner — collect raw counters with `rocprofv3` (present)
and run the *analysis* in a separate `rocm/dev-ubuntu` container that has the
full toolkit, pointed at the same `./profiling` output directory. The counters
are gathered by `rocprofv3`; `rocprof-compute` only interprets them, so the two
steps can run in different containers.

## C4. Path 1 — Timeline trace (where does time go?)

Use SGLang's built-in Torch Profiler to see the kernel call stack, overlap, and
relative durations across a forward pass. This runs *in the server* and is the
fastest way to decide *which* kernel deserves the microscope. Bring up the
profiling deployment, then drive it:

```bash
# Launch with the overlay.
docker compose -f docker-compose.yml -f docker-compose.profiling.yml \
  up -d qwen36-27b-fp8-sglang
docker compose logs -f qwen36-27b-fp8-sglang   # wait for ready

# Profile a fixed number of decode steps. Output lands in ./profiling on host.
source .env
curl -X POST -H "Content-Type: application/json" \
  -H "Authorization: Bearer $SGLANG_API_KEY" \
  "http://localhost:8002/start_profile" \
  -d '{"num_steps": 10}'

# Send one deterministic request to generate the steps being profiled.
curl -s -H "Authorization: Bearer $SGLANG_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"qwen36-27b-fp8-profiling","messages":[{"role":"user","content":"Write a haiku about caches."}],"max_tokens":64,"temperature":0}' \
  "http://localhost:8002/v1/chat/completions"

# Profiling auto-stops after num_steps and writes a .trace.json.gz to ./profiling.
# (Stopping can take seconds-to-minutes; that is expected.)
```

Open the resulting `./profiling/*.trace.json.gz` in <https://ui.perfetto.dev/>.
Identify the dominant kernels — likely a GDN/Mamba Triton kernel or an AITER
GEMM. Those are your targets for Path 2.

## C5. Path 2 — Hardware counters (why is this kernel slow?)

For occupancy, LDS, and cache metrics you need a **bounded one-shot**, not the
live server, because counter collection replays kernels. SGLang's
`bench_one_batch` runs a single prefill+decode batch and then exits — perfect to
wrap in a profiler. Run it inside the profiling container (note
`--disable-cuda-graph` again, mirroring the overlay):

```bash
# Open a shell in the profiling container.
docker compose -f docker-compose.yml -f docker-compose.profiling.yml \
  exec qwen36-27b-fp8-sglang bash

# --- inside the container ---------------------------------------------------

# (a) Raw counter collection with rocprofv3.
#     TCC_* = L2 cache, TCP_* = L1 (vL1D), SQ_* = wavefront/occupancy/LDS.
#     Counters are limited per pass; rocprofv3 splits them across passes.
rocprofv3 --pmc GRBM_GUI_ACTIVE SQ_WAVES SQ_INSTS_VALU \
          TCP_TCP_TA_DATA_STALL_CYCLES TCC_HIT TCC_MISS \
          --kernel-trace --output-format csv -d /workdir/rpv3_run -- \
  python3 -m sglang.bench_one_batch \
    --model-path Qwen/Qwen3.6-27B-FP8 \
    --tp 1 --batch-size 1 --input-len 1024 --output-len 32 \
    --attention-backend triton --mamba-backend triton \
    --page-size 64 --disable-cuda-graph --trust-remote-code

# (b) Full roofline + occupancy/cache report with rocprof-compute (if present).
#     'profile' collects (multi-pass replay); 'analyze' interprets.
rocprof-compute profile -n qwen_decode -- \
  python3 -m sglang.bench_one_batch \
    --model-path Qwen/Qwen3.6-27B-FP8 \
    --tp 1 --batch-size 1 --input-len 1024 --output-len 32 \
    --attention-backend triton --mamba-backend triton \
    --page-size 64 --disable-cuda-graph --trust-remote-code

# List the metric blocks available for this architecture, then analyse:
rocprof-compute analyze --list-metrics gfx950 --path workloads/qwen_decode/
rocprof-compute analyze --path workloads/qwen_decode/
```

The `rocprof-compute analyze` report gives you, per kernel: wavefront
**occupancy** against the VGPR/SGPR/LDS limiters, **LDS** utilisation and
bank-conflict stalls (`ALUStalledByLDS`), **vL1D (L1)** and **TCC (L2)** hit
rates, VALU/MFMA/SALU utilisation, and a roofline placing the kernel against the
gfx950 peak (HBM 8 TB/s; 4 MB L2; 256 MB Infinity Cache).

## C6. Trust-but-verify on new silicon

The gfx950 *counters* are defined in `rocprof-compute`, but some *derived
panels* and roofline curves can lag on a fresh architecture — a raw counter may
read correctly while a higher-level summary is still being tuned. Before
trusting numbers on an unfamiliar kernel, profile something whose behaviour you
already understand (a plain large GEMM) and sanity-check that its L2 hit rate and
occupancy land where the architecture predicts. If they do, trust the harder
cases; if they don't, suspect the tool's derived layer, not your kernel.

## C7. Tear down the profiling deployment

The profiling overlay is single-stream and graph-free — do **not** leave it
serving production traffic. When done, restore the serving config:

```bash
docker compose up -d qwen36-27b-fp8-sglang   # base file only -> back to serving
```

---

# Appendix — Why each non-obvious flag is set

| Flag / setting | Tier | Why |
|---|---|---|
| `--attention-backend triton` | serving + profiling | GDN hybrid attention runs best on Triton on ROCm; GDN layers always use Triton kernels internally regardless. flashinfer is CUDA-only. |
| `--mamba-backend triton` | serving + profiling | Consistent with the GDN/Mamba Triton path. |
| `SGLANG_USE_AITER=1` | serving | AMD's tuned GEMM/attention kernels; accelerates even with Triton attention. |
| `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1` | serving | Specv2 draft/verify overlap — what makes NEXTN MTP pay off on ROCm (enabled for AMD via SGLang #17450). |
| `--page-size 64` | both | Required by the hybrid Mamba KV layout. |
| `--speculative-* NEXTN` | serving only | MTP head ships in the weights; works on MI350X via Specv2. Removed for profiling to isolate the base forward pass. |
| no `--quantization` | both | block-FP8 is auto-detected from the model config (unlike MoE `moe_wna16`). |
| FP8 KV cache OFF | serving | CDNA4 supports it, but 288 GB makes it unnecessary; only ~25% of layers use transformer KV; bf16 preserves quality. |
| `--disable-cuda-graph` | profiling only | Surfaces individual kernel dispatches to the profiler; graphs would hide them. |
| `--max-running-requests 1` | profiling only | Deterministic, single-stream attribution for counters. |
| `cap_add: SYS_PTRACE` | both | Lets profilers attach to the running process. |
| `cap_add: SYS_ADMIN` | profiling only | Required for hardware performance-counter reads. |
| `perf_event_paranoid ≤ 2` | host | Kernel permission for PAPI/counter collection; cannot be set from a container. |

## Three-tier summary

| Tier | Tool | Frequency | Answers |
|---|---|---|---|
| Telemetry | Device Metrics Exporter → Prometheus → Grafana | continuous (5–15 s) | Is the system healthy? Throttling? RAS errors? |
| Timeline | SGLang Torch Profiler (`/start_profile`) → Perfetto | on-demand burst | Where does time go? Which kernel dominates? |
| Kernel | `rocprofv3` / `rocprof-compute` on `bench_one_batch` | on-demand, multi-pass | Occupancy, LDS, L1/L2 cache, roofline — *why* a kernel is slow. |