# Semantic LLM Router

Routes queries to the cheapest LLM that can handle them — by embedding, comparing, and caching.

## Why?

LLM APIs are expensive. A simple "hello" doesn't need GPT-4.

This router:
- **Embeds** the query
- **Routes** to the right model (Flash for simple, Pro for complex)
- **Caches** similar queries (free repeats)
- **Monitors** latency and cost per stage

## Results

Benchmark (Flash, 3 queries):
- **p50**: 1733 ms
- **p95**: 3234 ms
- **Cost**: $0.0016 per query

**Latency breakdown**:
- Embedding: 395ms (41%)
- Generation: 564ms (59%)
- Routing: 3ms (0.3%)

## Trade-off

| Query Type | Model | Why |
|---|---|---|
| Simple | Flash | Fast + cheap |
| Complex | Pro | Better quality |
| Repeated | Cache | Free |

Dynamic routing: if Pro > 2s, fallback to Flash.

## Setup

```bash
pip install -r requirements.txt
python -m src.database