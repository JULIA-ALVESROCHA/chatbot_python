# GeoLUME: RAG Chatbot for the Brazilian Geography Olympiad

GeoLUME is the virtual assistant of the Brazilian Geography Olympiad (OBG). It answers questions from students, teachers and schools using only the official OBG documents: the regulations, official notices and published materials. If the answer is not in those documents, it says so instead of guessing.

The backend is a FastAPI service built on a retrieval augmented generation (RAG) pipeline. The frontend is a self contained chat widget that drops into the OBG WordPress site.

## Features

- **Grounded answers.** Every answer is generated from retrieved chunks of the official corpus and returned with its sources.
- **Hybrid retrieval.** FAISS vector search combined with BM25 lexical search over the same chunks.
- **Query rewriting.** Follow up questions ("and for the in person phase?") are rewritten into self contained queries using the session history.
- **Bilingual.** Portuguese and English. English queries are translated to Portuguese before embedding, since the corpus is in Portuguese.
- **Live calendar.** Dates (registration, phases, results, certificates) come from a published Google Sheet, with a local YAML fallback and a cache.
- **Optional reranker.** A multilingual cross encoder can reorder the retrieved chunks.
- **Session history.** Short conversation memory per `session_id`, with an endpoint to clear it.
- **Index manifest.** Each index build records corpus hashes, package versions and the git revision, and the API checks it on startup.
- **Evaluation harness.** Atomic precision, recall, F1 and FactScore against a gold test set, plus retrieval and hallucination regression suites.

## How it works

```
question
  │
  ├─ meta question about the bot? ──► fixed answer, no retrieval
  │
  ├─ 1. rewrite      make the question self contained (uses chat history)
  ├─ 2. retrieve     FAISS + BM25, cosine thresholds, per page diversity
  ├─ 3. rerank       optional cross encoder
  └─ 4. generate     gpt-4o-mini answers from the chunks + calendar block
                      │
                      ▼
               { answer, sources }
```

The orchestration lives in `src/rag_pipeline/pipeline.py`.

## Tech stack

| Layer | Tools |
|---|---|
| API | FastAPI, Uvicorn, Pydantic Settings |
| RAG | LangChain, FAISS, rank-bm25 |
| Models | OpenAI `text-embedding-3-large` and `gpt-4o-mini`; optional `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| PDF loading | PyMuPDF |
| Frontend | Vanilla JS widget for WordPress (no build step) |
| Deploy | Railway (Python 3.11.9) |

## Project structure

```
chatbot_python/
├── src/
│   ├── app/                  FastAPI app: main.py, routes, settings, schemas
│   ├── rag_pipeline/
│   │   ├── pipeline.py       Rewrite → Retrieve → Rerank → Generate
│   │   ├── rewrite/          Query rewriting prompts and service
│   │   ├── retrieval/        Loader, item aware splitter, FAISS + BM25, manifest
│   │   ├── reranker/         Optional cross encoder
│   │   └── generator/        Answer service, prompt templates, calendar
│   ├── infra/                Answer cache and session history
│   └── utils/                Text cleanup, language detection, metrics
├── scripts/                  build_index.py, run_dev.py, component tests
├── data/
│   ├── raw/                  Source PDFs (not versioned)
│   ├── processed/            FAISS index and manifest
│   └── data.yml              Local calendar fallback
├── frontend/geolume-chat.js  WordPress chat widget
├── eval/                     Evaluation harness, test sets, reports
├── tests/                    QA and retrieval quality suites
└── run_all_tests.sh          Runs the full test battery
```

## Getting started

### 1. Requirements

- Python 3.11
- An OpenAI API key

### 2. Install

```bash
git clone https://github.com/JULIA-ALVESROCHA/chatbot_python.git
cd chatbot_python

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install rank-bm25 pymupdf    # hybrid search and PDF loading
```

### 3. Configure

```bash
cp .env.example .env
```

Open `.env` and set `OPENAI_API_KEY`. Every other value has a working default and is documented in `.env.example`. To print the settings currently in effect:

```bash
python -m src.app.core.config
```

### 4. Build the index

Put the official OBG PDFs in `data/raw/`, then run:

```bash
python -m scripts.build_index
```

This loads the PDFs, splits them into chunks that keep numbered regulation items whole, embeds them, saves the FAISS index to `data/processed/faiss_index/` and writes a manifest. Rebuild whenever the PDFs or the chunking settings change.

### 5. Run the API

```bash
python scripts/run_dev.py
# or
uvicorn src.app.main:app --reload
```

The API runs at `http://127.0.0.1:8000`. Interactive docs are at `/docs`.

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/ready` | Readiness check |
| `POST` | `/api/v1/chat` | Ask a question |
| `DELETE` | `/api/v1/chat/history/{session_id}` | Clear a session's history |

**Request**

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Quem pode participar da OBG?", "language": "pt", "session_id": "abc123"}'
```

`language` accepts `pt`, `en` or `auto`. `session_id` is optional and enables follow up questions.

**Response**

```json
{
  "answer": "...",
  "sources": [ ... ]
}
```

## Frontend widget

`frontend/geolume-chat.js` is a dependency free chat widget. To install it on WordPress, paste the file into a WPCode JavaScript snippet or load it with a script tag in the footer.

Override the defaults by defining `window.GeoLumeConfig` before the script:

```html
<script>
  window.GeoLumeConfig = {
    apiUrl: "https://your-api.example.com/api/v1/chat",
    position: "left"
  };
</script>
```

Remember to add the site's origin to `ALLOWED_ORIGINS` in `.env`.

## Configuration highlights

| Variable | Default | What it does |
|---|---|---|
| `GENERATION_MODEL` | `gpt-4o-mini` | Model that writes the answer |
| `EMBEDDING_MODEL` | `text-embedding-3-large` | Model used for the index and queries |
| `MAX_RETRIEVE` | `6` | Chunks sent to the generator |
| `RETRIEVAL_COSINE_THRESHOLD` | `0.25` | Minimum similarity to keep a chunk |
| `MAX_CHUNKS_PER_PAGE` | `4` | Caps chunks taken from one page |
| `USE_RERANKER` | `false` | Turns on the cross encoder |
| `TRANSLATE_QUERY_TO_PT` | `true` | Embeds English questions in Portuguese |
| `CALENDAR_URL` | published sheet | CSV source for OBG dates |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | CORS origins, comma separated |

See `.env.example` for the full list and the reasoning behind each threshold.

## Testing and evaluation

Run the whole battery and archive the results in a timestamped folder:

```bash
bash run_all_tests.sh           # everything, including the LLM judged eval (costs tokens)
bash run_all_tests.sh --fast    # skips the LLM judged eval
```

The battery covers QA regression, hallucination tests, retrieval quality, index diagnostics and pytest suites.

The answer quality harness is `eval/lumie_eval.py`. It decomposes answers into atomic facts and scores them with `gpt-4o-mini` as judge:

- **Precision, recall, F1** against gold answers
- **FactScore** for grounding against the official corpus

```bash
cd eval
python lumie_eval.py --dataset testset_questions.jsonl --corpus knowledge_source.txt \
  --seeds 13 21 42 --model gpt-4o-mini --out results.json --report results_report.md
```

The latest run (200 test units, 3 seeds) is in `report_final.md`. It reached a FactScore of 75.7 and an F1 of 33.8. Precision is well above recall, so the main gap is missing facts, which points to retrieval coverage. The report breaks results down by intent and language.

## Author

Júlia Rocha · Computer Science, UNIFAL-MG
