#!/usr/bin/env bash
# setup_env.sh - cria .env.example (versionado) e .env (local, ignorado)
#
# Uso:
#   bash setup_env.sh
#   # depois edite .env e coloque a OPENAI_API_KEY
#
# Rode a partir da raiz do projeto (chatbot_obg-main).

set -euo pipefail

cat > .env.example <<'EOF'
# .env.example -- COMITE ESTE ARQUIVO. Nunca comite o .env.
#
# Todo clone novo comeca com:  cp .env.example .env
# Assim nenhuma maquina cai em default silencioso do codigo.
#
# Conferir o que esta valendo:  python -m src.app.core.config

# ---------------------------------------------------------------- obrigatorio
OPENAI_API_KEY=sk-...

# ---------------------------------------------------------------- indice
FAISS_INDEX_PATH=data/processed/faiss_index
EMBEDDING_MODEL=text-embedding-3-large
DISTANCE_STRATEGY=EUCLIDEAN_DISTANCE

# ---------------------------------------------------------------- chunking
# So afeta scripts/build_index.py. Mudou aqui, precisa reconstruir o indice.
CHUNK_SIZE=500
CHUNK_OVERLAP=100

# ---------------------------------------------------------------- retrieval
RETRIEVAL_FETCH_K=20
MAX_RETRIEVE=6
MAX_RERANK=4

# Cosseno real, via cos = 1 - (1-s)^2 a partir do score do LangChain.
# Referencias medidas neste indice:
#   0.744  melhor hit PT ("quem pode participar da OBG?")
#   0.592  "como recupero minha senha?"
#   0.510  <- o que o antigo 0.3 significava de verdade
#   0.398  "o link do token expirou"  (estava sendo descartado)
RETRIEVAL_COSINE_THRESHOLD=0.25
SUPPORT_COSINE_THRESHOLD=0.15

# Nunca mandar contexto vazio para o gerador.
MIN_CHUNKS=2

# Respostas certas costumam ser chunks contiguos da mesma pagina.
# Era 2, o que truncava respostas de varias clausulas.
MAX_CHUNKS_PER_PAGE=4

BM25_TOP_ACCEPT=5

USE_RERANKER=false
RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
RERANKER_REVISION=1427fd652930e4ba29e8149678df786c240d8825

# Embeddar a pergunta em portugues mesmo quando o usuario escreve em ingles.
TRANSLATE_QUERY_TO_PT=true

# ---------------------------------------------------------------- geracao
GENERATION_MODEL=gpt-4o-mini
GENERATION_TEMPERATURE=0.1

# ---------------------------------------------------------------- historico
HISTORY_BACKEND=memory
SESSION_EXPIRY_HOURS=24

# ---------------------------------------------------------------- app
ALLOWED_ORIGINS=http://localhost:3000
LOG_LEVEL=INFO

# ---------------------------------------------------------------- obsoleto
# Nao lido apos o patch do vectorstore. Apagar depois de migrar.
RETRIEVAL_SCORE_THRESHOLD=0.3
EOF

echo "criado .env.example"

if [ -f .env ]; then
  echo ".env ja existe - nao foi sobrescrito"
else
  cp .env.example .env
  # Se a chave ja estiver exportada no shell, aproveita.
  if [ -n "${OPENAI_API_KEY:-}" ]; then
    tmp=$(mktemp)
    sed "s|^OPENAI_API_KEY=.*|OPENAI_API_KEY=${OPENAI_API_KEY}|" .env > "$tmp"
    mv "$tmp" .env
    echo "criado .env com a OPENAI_API_KEY do ambiente"
  else
    echo "criado .env - EDITE e coloque a OPENAI_API_KEY"
  fi
fi

# .gitignore
for entry in ".env" ".venv/" "venv/" ".lumie_cache.json" \
             "data/processed/.chat_history.json" "__pycache__/"; do
  if ! grep -qxF "$entry" .gitignore 2>/dev/null; then
    echo "$entry" >> .gitignore
    echo "  + .gitignore: $entry"
  fi
done

echo
echo "pronto. proximos passos:"
echo "  1. editar .env com a chave real (se ainda nao estiver)"
echo "  2. rm -rf .venv          # virtualenv vazio, sobra"
echo "  3. python -m src.app.core.config"