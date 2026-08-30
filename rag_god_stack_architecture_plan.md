# RAG "God Stack" — Architecture & Migration Plan

## 0. Where you're starting from

Your current app (`app.py`, `ingestion.py`, `rag_engine.py`) is a single-process Streamlit prototype:
- One user at a time, no auth, no persistence beyond the session
- Chroma runs in-process, `InMemoryStore` for parents — everything is lost on restart
- One LLM provider, no fallback, no caching, no retry logic
- No evaluation loop except manually clicking "Run RAGAS Evaluation"
- No observability beyond print statements and Streamlit's own error banner

This is fine for validating the retrieval logic (which works — you proved that with the apple_10k.pdf test). It is not fine for anything with concurrent users, real documents at scale, or anyone who needs the system to stay up.

The stack below turns each of those weaknesses into a dedicated, swappable service.

---

## 1. Layer-by-layer breakdown

### Frontend — Next.js
Replaces the Streamlit UI. Responsibilities:
- Document upload UI, indexing status/progress (via polling or SSE from FastAPI)
- Chat-style query interface with streaming token output
- Auth (NextAuth or Clerk) gating who can upload/query
- Renders citations/source chunks as clickable references back to page numbers

Why it matters here: Streamlit can't do multi-user sessions, streaming responses feel bolted-on, and there's no real component model for things like a source-chunk sidebar. Next.js gives you SSR, streaming via the App Router, and a normal deployment story (Vercel or containerized).

### Backend — FastAPI
The single entry point for the frontend. Responsibilities:
- REST/SSE endpoints: `POST /documents` (ingest), `POST /query` (streamed), `GET /documents/{id}/status`
- Auth/session validation, rate limiting, request validation (Pydantic — which you're already halfway using via LangChain)
- Talks to LangGraph for anything involving retrieval or generation; talks directly to Postgres for anything CRUD (document metadata, chat history, user accounts)
- Background task queue (or a lightweight worker via `BackgroundTasks`/Celery/RQ) for ingestion, since parsing + chunking + embedding a large PDF shouldn't block a request

This replaces `app.py`'s direct function calls with a proper API boundary — the frontend never touches LangChain objects directly.

### Orchestration — LangGraph
Replaces the flat LCEL chain in `rag_engine.py` with an explicit graph. This is the biggest structural change. Where your current chain is:

```
retriever -> format_docs -> prompt -> llm -> parser
```

a LangGraph version becomes a state machine with real branches, e.g.:

```
query -> [rewrite query?] -> retrieve -> rerank -> [enough context?]
  -> yes: generate -> [guardrail check] -> return
  -> no: return INSUFFICIENT_DATA (or trigger a broader search)
```

Why this matters: your current `INSUFFICIENT_DATA` behavior is just a prompt instruction the LLM might ignore. A graph lets you enforce it programmatically — check retrieval score thresholds before ever calling the LLM, add a query-rewriting node for bad user phrasing, add a self-correction loop if the guardrail step flags a hallucination, etc. It also gives you resumable, inspectable state for debugging (which pairs directly with LangSmith below).

### Embeddings — OpenAI `text-embedding-3-large` or BGE-M3
Two real options, not equivalent:
- **`text-embedding-3-large`**: managed, no infra, strong general-purpose retrieval quality, cost per token, data leaves your infrastructure.
- **BGE-M3**: open-weight, self-hostable, supports dense + sparse + multi-vector retrieval in one model (useful for hybrid search), no per-call cost once hosted, but you own the GPU/inference infra.

For financial/compliance documents specifically (your actual use case), self-hosting via BGE-M3 is worth serious consideration if data residency or cost-at-scale matters — a 10-K should probably not leave your VPC by default in a lot of compliance contexts. This is a genuine tradeoff to decide explicitly, not default on.

### Vector database — Qdrant (your pick)
Replaces Chroma's in-process store. Qdrant gives you:
- A real server process (Docker/K8s), persistent by default, not tied to your app's lifecycle
- Payload filtering (filter by document ID, page range, doc type) alongside vector search — useful since you'll have many indexed filings, not one
- Native support for hybrid (dense + sparse) search if you go the BGE-M3 route
- Horizontal scaling story if retrieval volume grows

Your Parent-Child pattern still applies: child embeddings go into Qdrant, but the parent store should move from `InMemoryStore` to Postgres (or Redis) — anything in-process memory dies with the container.

### Reranker — BGE Reranker v2
Added stage between retrieval and generation: Qdrant returns top-k (e.g. 20) candidates by vector similarity, the reranker cross-encodes query+candidate pairs and reorders by actual relevance, and you keep only the top-n (e.g. 5) for the LLM. This directly fixes the failure mode you hit earlier in this conversation — a semantically-close-but-wrong chunk being retrieved. Cross-encoders are more accurate than pure vector similarity precisely because they see the query and document together instead of comparing two independent embeddings.

### LLM — GPT-5 / Claude / Gemini (multi-provider)
Don't hardcode one provider (your current `gemini-1.5-flash` string literally stopped working mid-project — you saw that firsthand). Options:
- Abstract behind LangGraph's model-agnostic chat interface, config-driven per environment
- Or add real routing logic: cheap/fast model for query rewriting, stronger model for final compliance-grade generation, automatic fallback if a provider errors or is deprecated
- Pin exact model version strings in config (not code), and add a scheduled check or alerting so a retirement doesn't surface as a silent prod outage — you'd have found gemini-1.5-flash's shutdown from a dashboard instead of a stack trace

### Cache — Redis
Multiple uses, not just "make it faster":
- Cache embeddings for repeated/near-duplicate queries
- Cache full LLM responses for identical queries within a TTL (real cost savings on repeated compliance questions across a filing)
- Session/chat history for in-flight conversations before they're persisted to Postgres
- Rate limiting counters at the FastAPI layer

### Relational database — PostgreSQL
Everything that isn't a vector: document metadata (filename, upload date, indexing status, page count), user accounts, chat/query history with citations, audit logs (who asked what, when, what was retrieved, what was answered) — genuinely important for a *compliance* tool where you may need to prove what the system told someone and why. `pgvector` is also an option if you want a secondary/backup vector store or want to keep small indexes co-located with metadata, though Qdrant remains primary for this stack.

### Observability — LangSmith
Traces every LangGraph run: which node executed, what was retrieved, what the reranker did to the ordering, exact prompt sent to the LLM, latency per step, token cost per step. This is what makes the graph in section 1.3 debuggable in production instead of a black box — without it, "why did the compliance answer say INSUFFICIENT_DATA" is a guessing game.

### Evals — Ragas + DeepEval
You already have a Ragas skeleton (`eval_ragas.py`) — this generalizes it:
- **Ragas**: faithfulness, context precision/recall, answer relevance — RAG-specific metrics, run as a CI gate on every prompt/retrieval change
- **DeepEval**: broader LLM-app testing (unit-test-style assertions, custom metrics, regression suites) — pairs well with a proper test suite instead of manually typing queries into a text area

### Guardrails — Guardrails AI + Presidio
Two different jobs:
- **Guardrails AI**: structural/content validation on LLM output — enforce your citation format, block responses that don't cite a page number, catch schema violations if you move to structured output
- **Presidio**: PII detection/redaction — relevant specifically because financial filings and legal contracts can contain names, SSNs, account numbers; scan both ingested documents and generated answers before they're returned or logged

---

## 2. Request lifecycles

**Ingestion**: `Next.js upload → FastAPI (POST /documents) → background task → parse PDF → chunk (child/parent) → embed children → Qdrant upsert → parent chunks to Postgres → status row updated → frontend polls/gets notified`

**Query**: `Next.js query → FastAPI (POST /query, SSE) → LangGraph: embed query → Qdrant search (top-k) → BGE rerank (top-n) → assemble context → guardrail pre-check → LLM generate (streamed) → guardrail post-check (PII/citation format) → Postgres audit log write → SSE stream back to Next.js`

Every arrow in the query path is a LangSmith span.

---

## 3. Suggested migration phases

1. **FastAPI + LangGraph skeleton** (your stated starting point) — stand up the API shell and port your existing Chroma/OpenAI logic into a LangGraph graph with 2-3 nodes (retrieve, generate, fallback). Get functional parity with today's app before adding new components.
2. **Qdrant swap-in** — replace Chroma, move parent store from `InMemoryStore` to Postgres. Validate retrieval quality is at least equal before moving on.
3. **Reranker** — add BGE Reranker v2 as a graph node between retrieve and generate. This is usually the single highest-leverage quality improvement.
4. **Redis + Postgres persistence** — chat history, caching, audit logging. Needed before real users touch it.
5. **Next.js frontend** — once the API is stable, build the real UI against it instead of Streamlit.
6. **LangSmith + evals in CI** — instrument before you scale traffic, not after something goes wrong.
7. **Guardrails/Presidio + multi-provider LLM routing** — hardening pass once the core loop is proven.

---

## 4. Open decisions to make explicitly (not defaults)

- Self-hosted BGE-M3 vs OpenAI embeddings — data residency vs operational simplicity
- Qdrant deployment target — self-managed Docker/K8s vs Qdrant Cloud
- Sync request/response vs SSE streaming for `/query` — affects both FastAPI and Next.js design
- How aggressively to cache LLM responses given compliance answers may need to reflect document updates (cache invalidation on re-ingestion)
