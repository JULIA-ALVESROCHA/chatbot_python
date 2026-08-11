#!/usr/bin/env bash
# run_all_tests.sh — roda a bateria e arquiva em v1_test/
#
#   bash run_all_tests.sh              # tudo, inclusive lumie_eval (custa tokens)
#   bash run_all_tests.sh --fast       # pula lumie_eval
#   bash run_all_tests.sh --out v2_test
#
# Nada aborta a bateria: cada etapa roda isolada e o código de saída vai
# para o MANIFEST.md.

set -uo pipefail

OUT="v1_test"; FAST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --fast) FAST=1; shift ;;
    --out)  OUT="$2"; shift 2 ;;
    *) echo "opção desconhecida: $1"; exit 1 ;;
  esac
done

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"   # resolve o "No module named src"

STAMP=$(date +%Y%m%d_%H%M%S)
DIR="${OUT}/${STAMP}"
mkdir -p "$DIR/logs" "$DIR/artefatos" "$DIR/estado"
MANIFEST="$DIR/MANIFEST.md"

log()  { echo -e "\033[1;36m==>\033[0m $*"; }
warn() { echo -e "\033[1;33m!!\033[0m $*"; }

echo "# Bateria — $STAMP" > "$MANIFEST"

# ------------------------------------------------------------------ estado
log "Estado do sistema"
git rev-parse --short HEAD > "$DIR/estado/git.txt" 2>&1
git status --porcelain >> "$DIR/estado/git.txt" 2>&1
pip freeze > "$DIR/estado/pip_freeze.txt" 2>&1
python -V > "$DIR/estado/python.txt" 2>&1
python -m src.app.core.config > "$DIR/estado/config.txt" 2>&1 || warn "config"
python -m src.rag_pipeline.generator.calendar > "$DIR/estado/calendario.txt" 2>&1 || true
ls -la data/processed/ > "$DIR/estado/processed.txt" 2>&1 || true

FALHAS=0
{ echo ""; echo "| etapa | status | tempo |"; echo "|---|---|---|"; } >> "$MANIFEST"

roda() {
  local nome="$1"; shift
  local alvo="${2:-}"
  # se o segundo argumento é um caminho .py e não existe, pula
  if [[ "$alvo" == *.py && ! -f "$alvo" ]]; then
    echo "| $nome | — | pulado (arquivo ausente) |" >> "$MANIFEST"
    warn "pulado: $nome ($alvo)"
    return
  fi
  log "$nome"
  local ini=$SECONDS
  "$@" > "$DIR/logs/${nome}.log" 2>&1
  local rc=$?
  local dur=$((SECONDS - ini))
  if [ $rc -eq 0 ]; then
    echo "| $nome | \`$rc\` OK | ${dur}s |" >> "$MANIFEST"
  else
    echo "| $nome | \`$rc\` FALHOU | ${dur}s |" >> "$MANIFEST"
    FALHAS=$((FALHAS+1)); warn "$nome -> rc=$rc  (logs/${nome}.log)"
  fi
}

# ------------------------------------------------------------------ QA quality
# runner primeiro: evaluator.py costuma ser biblioteca, não entrypoint.
roda qa_regression_runner python tests/QA_quality/regression_runner.py
roda qa_evaluator         python tests/QA_quality/evaluator.py

# ------------------------------------------------------------------ retrieval quality
roda rtv_runner    python tests/retrieval_quality/runner_retrieval.py
roda rtv_evaluator python tests/retrieval_quality/evaluator_rtv.py

# ------------------------------------------------------------------ reports
roda reports python reports/reports.py

# ------------------------------------------------------------------ diagnóstico
roda diagnose_v2         python diagnose_v2.py
roda diagnose_retrieval  python diagnose_retrieval.py
roda check_chunks        python check_chunks.py
roda check_metadata      python check_metadata.py

# ------------------------------------------------------------------ pytest
roda pytest_src   python -m pytest src/tests -v --tb=short -p no:cacheprovider
roda pytest_tests python -m pytest tests     -v --tb=short -p no:cacheprovider

# ------------------------------------------------------------------ lumie_eval
if [ "$FAST" -eq 1 ]; then
  echo "| lumie_eval | — | pulado (--fast) |" >> "$MANIFEST"
else
  # O cache fica em eval/. Sem limpar, a avaliação replica as respostas
  # da rodada anterior e parece que nenhuma mudança teve efeito.
  if [ -f eval/.lumie_cache.json ]; then
    cp eval/.lumie_cache.json "$DIR/artefatos/lumie_cache_antes.json"
    rm -f eval/.lumie_cache.json
    log "cache do eval limpo (cópia em artefatos/)"
  fi
  roda lumie_eval   python eval/lumie_eval.py
  roda regen_report python eval/regen_report.py
fi

# ------------------------------------------------------------------ artefatos
log "Coletando artefatos"
for f in eval/results.json eval/results_report.md eval/report.md \
         eval/demo_results.json eval/demo_summary.md eval/.lumie_cache.json; do
  [ -f "$f" ] && cp "$f" "$DIR/artefatos/"
done
[ -d reports ] && cp -r reports "$DIR/artefatos/reports" 2>/dev/null

if [ -f "$DIR/artefatos/results.json" ]; then
python - "$DIR/artefatos/results.json" >> "$MANIFEST" 2>/dev/null <<'PY'
import json, sys, statistics as st
d = json.load(open(sys.argv[1]))
a = d["aggregate"]["overall"]
rows = [r for s in d["per_seed"] for r in s["rows"]]
def rec(r):
    if "refused" in r: return bool(r["refused"])
    return any(k in u.lower() for u in r.get("unsupported_pred", [])
               for k in ("não encontrei","nao encontrei","contexto não","reformular"))
ref = [r for r in rows if rec(r)]; ans = [r for r in rows if not rec(r)]
def m(rs,f): return round(100*st.mean(r[f] for r in rs),1) if rs else 0.0
print("\n## lumie_eval\n")
print("| recorte | n | P | R | F1 | FactScore |")
print("|---|---|---|---|---|---|")
print(f"| tudo | {len(rows)} | {100*a['precision']['mean']:.1f} | {100*a['recall']['mean']:.1f} | {100*a['f1']['mean']:.1f} | {100*a['factscore']['mean']:.1f} |")
print(f"| recusas | {len(ref)} | {m(ref,'precision')} | {m(ref,'recall')} | {m(ref,'f1')} | {m(ref,'factscore')} |")
print(f"| respondidas | {len(ans)} | {m(ans,'precision')} | {m(ans,'recall')} | {m(ans,'f1')} | {m(ans,'factscore')} |")
print(f"\n**Taxa de recusa: {100*len(ref)/len(rows):.1f}%**")
PY
fi

{ echo ""; echo "- commit: \`$(git rev-parse --short HEAD 2>/dev/null)\`";
  echo "- etapas com falha: **$FALHAS**"; } >> "$MANIFEST"

echo ""; log "Fim. $FALHAS falha(s)."; echo "    $MANIFEST"