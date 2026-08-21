# Ask My Repo — repo-2-graph

Natural-language Q&A over any GitHub repository. Point it at a repo, and it builds a rich dependency graph (Neo4j) + vector search index (Qdrant) so you can ask structural and behavioural questions in plain English.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      API (FastAPI)                       │
│  POST /api/parse  POST /api/chat  GET /api/graph/{id}    │
└────┬────────────────────┬────────────────────────────────┘
     │                    │
     ▼                    ▼
┌──────────┐      ┌──────────────┐
│ Indexing │      │    Query     │
│ Pipeline │      │  Pipeline    │
│ (async)  │      │ (LangGraph)  │
└──────────┘      └──────────────┘
     │                    │
     └────────┬───────────┘
              ▼
     ┌──────────────────┐
     │  Neo4j + Qdrant  │
     └──────────────────┘
```

---

## Indexing Pipeline

Triggered by `POST /api/parse` and orchestrated in `src/backend/map/mapper.py:map_repository()`. Runs asynchronously with progress reporting.

### Stage 1 — Fetch

`src/backend/chunking/repo_parser.py:clone_repo()` clones the target repo into `src/data/<repo_id>`. Only `.py` files and `README.md` are read. Ignores virtual environments, caches, lockfiles, etc.

### Stage 2 — Parse (AST)

`repo_parser.py:parse_file()` uses Python's `ast` module to extract per file:

- **Imports** — module-level and `from` imports
- **Classes** — name, qualified name, base classes, line ranges
- **Functions & methods** — name, qualified name, parent class, line ranges
- **Call graph** — which function calls which (BFS, no nested descent)
- **Entry points** — HTTP decorators (`@app.get`, etc.), `if __name__ == '__main__'`, CLI decorators, task decorators, app runner calls (`uvicorn.run`, etc.)
- **Flagged candidates** — conventional entry filenames (`main.py`, `app.py`, `server.py`...) that _might_ be entry points but need LLM review

### Stage 3 — Graph Building

`src/backend/chunking/chunk_builder.py:ChunkBuilder.build()`:

1. Builds a **name index** (symbol → filepath) and **module index** (Python module path → filepath)
2. Resolves each file's imports against these indexes to create a **NetworkX directed graph** where nodes are files and edges are `IMPORTS` relationships
3. Nodes are colour-coded by top-level directory

`ChunkBuilder.push_to_neo4j()` writes the full graph to Neo4j:

```
Repo ──CONTAINS──► File
File ──DEFINES_CLASS──► Class
File ──DEFINES_FUNCTION──► Function
Class ──HAS_METHOD──► Function
Class ──INHERITS_FROM──► Class
Class ──INHERITS_EXTERNAL──► ExternalSymbol
Function ──CALLS──► Function
Function ──INSTANTIATES──► Class
Function ──CALLS_EXTERNAL──► ExternalSymbol
File ──IMPORTS──► File
File ──IMPORTS_SYMBOL──► Import
```

AST-detected entry points (`http_endpoint`, `main_block`, `app_runner`, `cli_entry`, `task_entry`) are annotated on Function/File nodes.

### Stage 4 — Entry Point LLM Review

Files flagged in Stage 2 (conventional filenames with no clear AST entry point) are sent to an LLM with their function names, decorators, and imports. The LLM classifies each as `http_endpoint`, `main_block`, `app_runner`, `cli_entry`, `task_entry`, or not an entry point. Results are merged back into Neo4j via `ChunkBuilder.apply_entry_points()`.

### Stage 5 — Vector Indexing

`src/backend/services/vector_db.py:VectorStore.build()` chunks code by:

- **Class boundaries** — with configurable overlap (default 5 lines)
- **Function/method boundaries** — including decorator lines
- **Module-level code** — remaining lines grouped in segments

Chunk size capped at 100 lines / 12,000 characters. Each chunk is metadata-tagged with its file path, class name, function name, and line range.

`VectorStore.push()` batches chunks (batch size: 4) to Qdrant with:
- **Dense embeddings**: `jinaai/jina-embeddings-v2-base-code`
- **Sparse embeddings**: `Qdrant/bm25`
- UUIDs deterministically derived from path + function + line range

### Stage 6 — Visualization

`ChunkBuilder.show()` generates an interactive HTML dependency graph via PyVis (saved to `notebook/`). Served via `GET /api/graph/{repo_id}`.

### Duplicate Detection

Before indexing, the pipeline checks Qdrant and Neo4j for existing data (`VectorStore.collection_exists()`, `ChunkBuilder.repo_exists()`). If both exist, the pipeline short-circuits to reusing existing indexes. Partial rebuilds (e.g., graph exists but vectors don't) are also handled.

### Progress Reporting

Pipeline stages with approximate progress percentages:

| Stage | Progress | Description |
|---|---|---|
| `fetching` | 8% | Clone repository |
| `graph_building` | 22% | AST parse + NetworkX build |
| `graph_saving` | 42-45% | Push to Neo4j + LLM entry review |
| `vector_building` | 55% | Chunk + embed |
| `vector_saving` | 58-90% | Push to Qdrant |
| `assistant_ready` | 95% | Finalising |

---

## Query Pipeline

Defined in `src/backend/chat_engine/engine.py:ChatWorkflow` as a LangGraph `StateGraph`. Each node is a callable that reads/writes `AgentState`.

### Graph Structure

```
Router ──graph_only──► Graph ──meaningful?──► Synthesizer
  │                       │                      │
  │                       └──no data──► Vector ──┘
  │
  ├──architecture──► Architect ──► Synthesizer
  │                      │
  │                      └──(with vector enrichment)
  │
  └──hybrid──► QueryRewriter ──► Graph ──► Vector ──► Synthesizer
```

### Node: Router

Classifies the user query into one of three paths:

- **`graph_only`** — purely structural: imports, dependencies, definitions, call graphs
- **`architecture`** — system-wide: request flows, entry point call chains, dependency maps, class interactions, system overview
- **`hybrid`** — everything else: behaviour, logic, variable values, feature implementation

Uses `with_structured_output(RouterDecision)` for structured classification.

### Node: QueryRewriter

Resolves pronouns and references (`it`, `that`, `the function`) using the last 8 turns of conversation history. Produces a self-contained rewritten query.

### Node: Architect

Handles system-level queries using predefined Cypher templates:

| Template | Description |
|---|---|
| `entry_call_chain` | Entry point → downstream call chains (up to 6 hops) |
| `request_flow` | Entry point → sink functions (call chain, depth-sorted) |
| `dependency_map` | File → transitive import chains (up to 4 hops) |
| `class_interaction` | Cross-class method calls and instantiations |
| `system_overview` | Per-file function, class, and entry point inventory |

Subtype is selected via keyword matching with LLM fallback.

Architect results can be enriched with vector search on critical-path files (up to 4 files, top-4 chunks per file reranked by cosine similarity).

### Node: Graph

Executes structural queries against Neo4j:

1. **Template matching** — regex-based pattern matching for common question types (direct imports, reverse lookup, leaf nodes, most dependencies, transitive dependencies, call graph, class hierarchy, file structure)
2. **LLM Cypher generation** — if no template matches, generates a Cypher query via `with_structured_output(CypherQuery)` with a schema-aware system prompt

Results are cached in `_graph_cache` keyed by query string.

### Node: Vector

Searches Qdrant for code chunks relevant to the query. If graph results are available, search is filtered to files extracted from the graph results. Uses `rerank()` which sorts by Qdrant's pre-computed score (or falls back to SentenceTransformer cosine similarity).

### Node: Synthesizer

Formats graph and vector results into a structured context block (`[Graph Relationships]` + `[Code Chunks]`) and calls the LLM to produce a final answer with confidence score and source attribution.

### State Handling

`AgentState` is a `TypedDict` containing: `user_query`, `rewritten_query`, `router_decision`, `reason`, `graph_result`, `vector_result`, `context`, `final_answer`, `user_history`, `current_agent`, and metadata fields.

History is persisted server-side per session (last 16 turns), capped per session in `session_histories`.

---

## LLM Fallback Pipeline

`src/backend/services/llm_fallback.py:FallbackChatModel` wraps `ChatOpenAI` with automatic failover to Groq's `llama-3.3-70b-versatile`:

- On `openai.OpenAIError` → retries via Groq with the same messages
- Implements `with_structured_output(fallback=True)` for Pydantic-parsed responses
- Tracks fallback state via `_fallback_used` flag (reported in API response)

---

## Cleanup Pipeline

`src/backend/services/repo_activity.py:RepoActivityTracker` runs a daemon thread that:

- Records activity timestamps per repo (`record_activity()`)
- Checks every 5 minutes (`CLEANUP_CHECK_INTERVAL_MINUTES`) for repos inactive > 3 hours (`REPO_INACTIVITY_TIMEOUT_HOURS`)
- On inactivity: wipes session caches → deletes Neo4j graph → deletes Qdrant collection → removes HTML visualisation
- Failed cleanups are retried with exponential backoff

Manual cleanup at `POST /api/cleanup/manual/{repo_id}`.

---

## Evaluation Pipeline

`src/evaluation/eval.py` runs automated Q&A evaluation using RAGAS:

1. Loads 20 questions with ground truth from `questions.json`
2. Processes them in batches of 3 with a 12-second cooldown between batches
3. Evaluates against: `faithfulness`, `answer_relevancy`, `answer_correctness`, `context_precision`, `context_recall`
4. Uses `gpt-4o-mini` as the evaluator LLM

---

## Project Structure

```
src/
├── backend/
│   ├── api.py                     # FastAPI server
│   ├── job_status.py              # Async job tracking
│   ├── agent_state/state.py       # LangGraph AgentState
│   ├── chat_engine/engine.py      # LangGraph query workflow
│   ├── chunking/
│   │   ├── repo_parser.py         # Git clone + AST analysis
│   │   └── chunk_builder.py       # NetworkX graph + Neo4j push
│   ├── map/mapper.py              # Indexing pipeline orchestrator
│   └── services/
│       ├── vector_db.py           # Qdrant chunking/embedding/search
│       ├── query_engine.py        # Cypher + architect templates
│       ├── llm_fallback.py        # OpenAI → Groq fallback
│       └── repo_activity.py       # Inactivity cleanup daemon
├── evaluation/
│   ├── eval.py                    # RAGAS evaluation
│   └── questions.json             # 20 eval questions
├── data/                          # Cloned repos (gitignored)
├── models/                        # Local model cache (gitignored)
└── frontend/                      # React UI (Vite + Tailwind)
```

---

## Configuration

All via `.env` (see `.env.example`):

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | LLM for Q&A + query generation |
| `GROQ_API_KEY` | No | LLM fallback |
| `NEO4J_URI` | Yes | Neo4j connection string |
| `NEO4J_USER` | Yes | Neo4j username |
| `NEO4J_PASS` | Yes | Neo4j password |
| `QDRANT_END_POINT` | Yes | Qdrant cluster URL |
| `QDRANT_API_KEY` | Yes | Qdrant API key |
| `REPO_INACTIVITY_TIMEOUT_HOURS` | No | Cleanup timeout (default: 3) |
| `CLEANUP_CHECK_INTERVAL_MINUTES` | No | Cleanup check interval (default: 5) |

---

## Quick Start

```bash
# Backend
uv sync
uv run uvicorn src.backend.api:app --reload

# Frontend
cd src/frontend
npm install
npm run dev
```

Point the UI at `http://localhost:5173`, enter a GitHub repo URL, and start asking questions.
