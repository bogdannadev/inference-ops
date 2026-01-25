# RAG Infrastructure Reference Configs

Production Docker Compose configurations for a complete RAG pipeline.

## What's Inside

- **BGE-M3** (`bge-m3/`) — Multilingual embeddings (768-dim, port 8000)
- **Milvus** (`milvus/`, `milvus-gpu/`) — Vector database, CPU and GPU variants (port 19530)
- **Qwen3 Models** (`qwen3-*/`) — LLMs (0.6B to 8B), OpenAI-compatible API
  - `qwen3-1.7b/` — Balanced production config
  - `qwen3-8b/` — vLLM (multi-GPU, extended context window through YaRN)
  - `qwen3-8b/docker-compose.sglang.yaml` — SGLang (multi-GPU, RAG-optimised with prefix caching, extended context window through YaRN)

## Quick Start

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

## Prerequisites

- NVIDIA GPU + Docker NVIDIA runtime
- HuggingFace token (for model access)
- Sufficient VRAM (check individual configs)

## Architecture

Documents → BGE-M3 (embeddings) → Milvus (search) → Qwen3 (generation)

Each service is independent; compose them as needed.

## License

Milvus (Apache 2.0), vLLM/SGLang (Apache 2.0), Qwen3 (Apache 2.0), BGE-M3 (MIT).
