# Inference Ops — Reference Deployment Configs

Production Docker Compose configurations for LLM inference, embeddings, vector search, and full observability stacks.

## What's Inside

### Embeddings

- **BGE-M3** (`bge-m3/`, `bge-m3-cpu/`) — Multilingual embeddings (768-dim, port 8000), CPU and GPU variants

### Vector Database

- **Milvus** (`milvus/`, `milvus-gpu/`) — Vector database, CPU and GPU variants (port 19530)

### Qwen3 Series — NVIDIA GPUs

Small-to-medium models for single-GPU or multi-GPU NVIDIA deployments.

| Directory | Model | Engine | Notes |
|---|---|---|---|
| `qwen3-0.6b/` | Qwen3-0.6B | SGLang | Lightweight, edge-friendly |
| `qwen3-1.7b/` | Qwen3-1.7B | SGLang | Balanced production config |
| `qwen3-4b/` | Qwen3-4B | SGLang | Mid-range single GPU |
| `qwen3-8b/` | Qwen3-8B | vLLM / SGLang | Multi-GPU, YaRN extended context |
| `qwen3.5-4b/` | Qwen3.5-4B | SGLang | Latest 4B variant |
| `qwen3.5-9B/` | Qwen3.5-9B | SGLang / llama.cpp | Multi-GPU SGLang + CPU fallback |
| `qwen3-Embedding-0.6B/` | Qwen3 Embedding 0.6B | SGLang / vLLM | Embedding model |
| `qwen3-Embedding-8b/` | Qwen3 Embedding 8B | SGLang | Large embedding model |
| `qwen-image-2512/` | Qwen2.5-VL | llama.cpp (SDCpp) | Vision-language model |

### Qwen3.5 MoE — Consumer GPU

| Directory | Model | Engine | Hardware |
|---|---|---|---|
| `qwen3.5-35B-A3B/` | Qwen3.5-35B-A3B (UD-Q4_K_XL) | llama.cpp | RTX A4000 16 GB |

### Qwen3.6 MoE — Multi-GPU NVIDIA

| Directory | Model | Engine | Hardware |
|---|---|---|---|
| `qwen3.6-35B-A3B/` | Qwen3.6-35B-A3B-FP8 | vLLM nightly | 4× RTX A4000 (64 GB total VRAM) |

### Qwen3.6-27B — AMD Instinct (MI300X / MI350X)

Large dense hybrid (GDN + attention) models on AMD CDNA3/CDNA4. All configs include full observability (Prometheus + AMD Device Metrics Exporter + Grafana).

| Directory | Precision | Engine | Hardware | Notes |
|---|---|---|---|---|
| `qwen3.6-27b-mi300x/` | BF16 | SGLang (rocm/sgl-dev) | MI300X (192 GB HBM3, CDNA3) | HiCache, AITER kernels, Triton attention. Full metrics stack with Grafana dashboards. |
| `qwen3.6-27b-fp8-mi300x/` | FP8 | SGLang (rocm/sgl-dev) | MI300X (192 GB HBM3, CDNA3) | FP8 weights, HiCache enabled |
| `qwen3.6-27b-mi350x/` | BF16 | SGLang (rocm/sgl-dev) | MI350X (288 GB HBM3E, CDNA4) | NEXTN speculative decoding, full metrics + rocprofv3 profiling |
| `qwen3.6-27b-FP8-mi350x/` | FP8 | SGLang (lmsysorg/sglang-rocm) | MI350X (288 GB HBM3E, CDNA4) | FP8 + NEXTN spec decode + Triton kernel cache. Full 3-tier observability: telemetry, timeline, kernel profiling. |
| `qwen3.6-27B-awq/` | AWQ | SGLang | NVIDIA GPU | AWQ-int4 quantized |

## Quick Start — Basic RAG Pipeline

```bash
export HF_TOKEN=hf_xxx

docker compose -f milvus/docker-compose.yml up -d
docker compose -f bge-m3/docker-compose.yml up -d
docker compose -f qwen3-8b/docker-compose.yml up -d

# Verify
curl http://localhost:19530/healthz  # Milvus
curl http://localhost:8000/health     # BGE-M3
curl http://localhost:8002/v1/models  # Qwen3-8B
```

## Quick Start — Qwen3.6-27B on MI300X (with metrics)

```bash
export HF_TOKEN=hf_xxx
cd qwen3.6-27b-mi300x

# 1. Bring up observability tier first
docker compose -f docker-compose.yml -f compose.metrics.yml up -d prometheus amd-metrics-exporter grafana

# 2. Then bring up the inference server
docker compose -f docker-compose.yml -f compose.metrics.yml up -d

# Endpoints
# SGLang:     http://<host>:8002
# Prometheus: http://<host>:9090
# Grafana:    http://<host>:3000
# AMD GPU:    http://<host>:5000/metrics
```

See `qwen3.6-27b-mi300x/README.md` for full setup, dashboards, and tuning notes.

## Quick Start — Qwen3.6-27B-FP8 on MI350X (with profiling)

```bash
export HF_TOKEN=hf_xxx
cd qwen3.6-27b-FP8-mi350x

# Serving + telemetry
docker compose up -d

# Kernel profiling overlay (on-demand, NOT for production)
docker compose -f docker-compose.yml -f docker-compose.profiling.yml up -d qwen36-27b-fp8-sglang
```

See `qwen3.6-27b-FP8-mi350x/AGENTS.md` for the complete 3-tier observability guide.

## Prerequisites

- **NVIDIA deployments:** NVIDIA GPU + Docker NVIDIA runtime
- **AMD deployments:** AMD Instinct MI300X/MI350X + ROCm-enabled kernel + `/dev/kfd` + `/dev/dri` accessible
- **HuggingFace token** (for gated model access)
- **Sufficient VRAM / HBM** (check individual configs)

## Architecture

Documents → BGE-M3 (embeddings) → Milvus (search) → Qwen (generation)

Each service is independent; compose them as needed.

## Observability

The AMD MI300X/MI350X deployments ship with a 3-tier observability stack:

| Tier | Tool | Frequency | Answers |
|---|---|---|---|
| Telemetry | AMD Device Metrics Exporter → Prometheus → Grafana | continuous (5–15 s) | System health, throttling, RAS errors |
| Timeline | SGLang Torch Profiler → Perfetto | on-demand burst | Where does time go? Which kernel dominates? |
| Kernel | `rocprofv3` / `rocprof-compute` | on-demand, multi-pass | Occupancy, LDS, L1/L2 cache, roofline |

## License

Milvus (Apache 2.0), vLLM/SGLang (Apache 2.0), Qwen3/Qwen3.5/Qwen3.6 (Apache 2.0), BGE-M3 (MIT).
