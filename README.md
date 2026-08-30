# RAG backend — phase 1 (FastAPI + LangGraph + Qdrant)

Functional-parity rebuild of the original Streamlit app's retrieval logic,
restructured into a real API with a branching LangGraph pipeline instead of
a flat LCEL chain.

## What's here vs what's still a placeholder

| Component | Status |
|---|---|
| FastAPI (`/documents`, `/query`) | Implemented |
| LangGraph (`retrieve -> check_sufficiency -> generate/fallback`) | Implemented |
| Qdrant (child chunk vectors) | Implemented |
| Parent chunk store | **sqlite placeholder** — becomes Postgres in phase 4 |
| Document status registry | **sqlite placeholder** — becomes Postgres in phase 4 |
| Multi-provider LLM (openai / anthropic / google_genai) | Implemented via config, no reranker/cache/streaming yet |
| BGE Reranker v2 | Not yet — hook point is commented in `graph/nodes.py::retrieve_node` |
| Redis cache | Not yet (phase 4) |
| Next.js frontend | Not yet (phase 5) |
| LangSmith / evals / guardrails | Not yet (phases 6-7) |

## Run it

```bash
cp .env.example .env
# fill in OPENAI_API_KEY (embeddings always use OpenAI for now) and
# whichever LLM_PROVIDER key you're using

docker compose up -d qdrant

pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Try it

```bash
# Upload a PDF
curl -X POST http://localhost:8000/documents \
  -F "file=@apple_10k.pdf"
# -> {"document_id": "...", "status": "processing"}

# Poll status
curl http://localhost:8000/documents/<document_id>/status

# Query once status is "ready"
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main risk factors described in Item 1A?", "document_id": "<document_id>"}'
```

## Next steps (see the architecture plan doc)

1. Add BGE Reranker v2 as a node between `retrieve` and `check_sufficiency`.
2. Move `parent_store.py` / `document_registry.py` from sqlite to Postgres.
3. Add Redis for response/embedding caching.
4. Add SSE streaming to `/query`.
5. Build the Next.js frontend against this API.
"# RAG-Financial-Eval" 
