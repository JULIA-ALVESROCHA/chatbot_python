#!/usr/bin/env python
"""
diagnose.py — diagnóstico do índice da OBG

    python diagnose.py                 # tudo
    python diagnose.py --probe-only    # só as consultas
    python diagnose.py > v1_test/probe.txt

Sete verificações, todas com um número que dá para comparar entre rodadas:

  1. FINGERPRINT   identidade do índice (ntotal, checksum dos vetores)
  2. SANIDADE      um chunk consultado com o próprio texto -> cosseno ~1
  3. CORPUS        quantos chunks por documento
  4. METADADOS     chunk_id, page, item preenchidos? ids únicos?
  5. QUALIDADE     chunks cortados no meio da frase
  6. CONSULTAS     cosseno real por pergunta, PT e EN
  7. CONTAMINAÇÃO  material não normativo aparecendo em pergunta de regra

Sobre o cosseno: os embeddings são unit-norm, então cos = 1 - d²/2 a partir
da distância L2 crua. NÃO use similarity_search_with_relevance_scores num
índice EUCLIDEAN — ele aplica 1 - d/√2, que produz score negativo e
comprime a faixa útil.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from src.app.core.config import settings

INDEX = Path(settings.faiss_index_path)

# Documentos que definem regra. O resto é material de apoio: se aparecer no
# topo de uma pergunta normativa, o gerador responde com conteúdo errado.
NORMATIVOS = ("regulamento", "edital", "duvidas", "dúvidas", "procedimentos")

CONSULTAS = [
    ("pt", "quem pode participar da OBG?"),
    ("pt", "quantos alunos formam uma equipe?"),
    ("pt", "quando serao as provas?"),
    ("pt", "qual o valor da inscricao?"),
    ("pt", "como recupero minha senha?"),
    ("pt", "quais navegadores sao suportados?"),
    ("pt", "o link do token expirou, o que faco?"),
    ("pt", "quem pode ser professor orientador?"),
    ("pt", "como funciona a substituicao de membros da equipe?"),
    ("pt", "posso usar IA generativa nas provas?"),
    ("en", "who can participate in the OBG?"),
    ("en", "how do I recover my password?"),
    ("en", "which browsers are supported?"),
    ("en", "how many students form a team?"),
]

L = 74


def cabecalho(t):
    print("\n" + "=" * L)
    print(t)
    print("=" * L)


def l2_cos(d) -> float:
    return 1.0 - (float(d) ** 2) / 2.0


def nome_curto(meta) -> str:
    return Path(str((meta or {}).get("source", "?"))).name


def eh_normativo(meta) -> bool:
    n = nome_curto(meta).lower()
    return any(k in n for k in NORMATIVOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    if not (INDEX / "index.faiss").exists():
        sys.exit(f"Índice não encontrado em {INDEX}\n"
                 f"Rode: python -m scripts.build_index")

    emb = OpenAIEmbeddings(model=settings.embedding_model)
    vs = FAISS.load_local(str(INDEX), emb, allow_dangerous_deserialization=True)
    docs = list(vs.docstore._dict.values())

    # ---------------------------------------------------------- 1 fingerprint
    cabecalho("1. FINGERPRINT  (compare entre máquinas e entre rodadas)")
    idx = vs.index
    vecs = idx.reconstruct_n(0, idx.ntotal)
    norms = np.linalg.norm(vecs, axis=1)
    sha = hashlib.sha256(np.ascontiguousarray(vecs).tobytes()).hexdigest()[:16]

    print(f"  ntotal              {idx.ntotal}")
    print(f"  dim                 {idx.d}")
    print(f"  classe faiss        {type(idx).__name__}")
    print(f"  embedding_model     {settings.embedding_model}")
    print(f"  norma média         {norms.mean():.6f}")
    print(f"  unit-norm           {np.allclose(norms, 1.0, atol=1e-3)}")
    print(f"  vec_checksum        {sha}")
    print(f"  config fingerprint  {settings.fingerprint()}")

    if not np.allclose(norms, 1.0, atol=1e-3):
        print("  ERRO: vetores não são unit-norm — cos = 1 - d²/2 é inválido")

    # ---------------------------------------------------------- 2 sanidade
    cabecalho("2. SANIDADE  (embedder e índice concordam?)")
    texto = docs[0].page_content[:400]
    d0 = float(vs.similarity_search_with_score(texto, k=1)[0][1])
    c0 = l2_cos(d0)
    print(f"  consulta = texto de um chunk do próprio índice")
    print(f"  distância L2        {d0:.4f}")
    print(f"  cosseno             {c0:.4f}   (esperado >= 0.98)")
    print(f"  -> {'OK' if c0 >= 0.98 else 'FALHOU: índice foi criado com outro modelo'}")

    if not args.probe_only:
        # ------------------------------------------------------ 3 corpus
        cabecalho("3. CORPUS")
        por_doc = Counter(nome_curto(d.metadata) for d in docs)
        for nome, n in por_doc.most_common():
            marca = "  " if any(k in nome.lower() for k in NORMATIVOS) else "* "
            print(f"  {marca}{nome[:56]:58} {n:4} chunks")
        nao_norm = sum(n for nm, n in por_doc.items()
                       if not any(k in nm.lower() for k in NORMATIVOS))
        print(f"\n  (*) não normativo: {nao_norm} chunks "
              f"({100*nao_norm/len(docs):.0f}% do índice)")

        # ------------------------------------------------------ 4 metadados
        cabecalho("4. METADADOS")
        ids = [(d.metadata or {}).get("chunk_id") for d in docs]
        com_item = sum(1 for d in docs if (d.metadata or {}).get("item"))
        sem_page = sum(1 for d in docs if (d.metadata or {}).get("page") is None)
        print(f"  chunk_id preenchido {sum(1 for i in ids if i)}/{len(docs)}")
        print(f"  chunk_id únicos     {len(set(ids))}/{len(ids)}"
              f"{'  <- COLISÃO' if len(set(ids)) != len(ids) else ''}")
        print(f"  com número de item  {com_item}/{len(docs)} "
              f"({100*com_item/len(docs):.0f}%)")
        print(f"  sem page            {sem_page}")

        itens = Counter((d.metadata or {}).get("item") for d in docs
                        if (d.metadata or {}).get("item"))
        dup = sorted(k for k, v in itens.items() if v > 1)
        if dup:
            print(f"  itens repetidos     {len(dup)}: {', '.join(dup[:10])}")
            print("    (o regulamento repete 2.2, 2.3, 2.4, 4.3, 5.4.1, 6.4.2)")

        # ------------------------------------------------------ 5 qualidade
        cabecalho("5. QUALIDADE DO CHUNKING")
        tam = sorted(len(d.page_content) for d in docs)
        orfaos = [d for d in docs if d.page_content[:1].islower()]
        ligaduras = sum(d.page_content.count(c) for d in docs
                        for c in "\ufb00\ufb01\ufb02\ufb03\ufb04")
        print(f"  chars  min={tam[0]}  mediana={tam[len(tam)//2]}  max={tam[-1]}")
        print(f"  começando em minúscula  {len(orfaos)}/{len(docs)} "
              f"({100*len(orfaos)/len(docs):.0f}%)  <- corte no meio da frase")
        print(f"  ligaduras ﬁ/ﬂ           {ligaduras}  <- extração suja")
        if orfaos:
            print(f"\n  exemplo: {orfaos[0].page_content[:90]!r}...")

    # ---------------------------------------------------------- 6 consultas
    cabecalho(f"6. CONSULTAS  (cosseno real, k={args.k})")
    print(f"  {'':4} {'melhor':>7} {'k-ésimo':>8}  {'>=0.25':>7}  consulta")
    print("  " + "-" * (L - 4))

    resultados = defaultdict(list)
    contaminacao = []

    for lang, q in CONSULTAS:
        hits = vs.similarity_search_with_score(q, k=args.k)
        cos = [l2_cos(d) for _, d in hits]
        acima = sum(1 for c in cos if c >= settings.retrieval_cosine_threshold)
        resultados[lang].append(cos[0])

        flag = " " if cos[0] >= 0.35 else "!"
        print(f"  [{lang}]{flag}{cos[0]:7.3f} {cos[-1]:8.3f}  {acima:4}/{args.k}   {q[:38]}")

        # contaminação: doc não normativo no topo de pergunta de regra
        top = hits[0][0].metadata
        if not eh_normativo(top):
            contaminacao.append((q, nome_curto(top), cos[0]))

    for lang in ("pt", "en"):
        v = resultados[lang]
        if v:
            print(f"\n  {lang.upper()}  melhor cosseno médio {np.mean(v):.3f}  "
                  f"mín {min(v):.3f}  máx {max(v):.3f}")

    if resultados["pt"] and resultados["en"]:
        gap = np.mean(resultados["pt"]) - np.mean(resultados["en"])
        print(f"\n  penalidade EN->PT: {gap:.3f}")
        if gap > 0.15:
            print("  -> traduzir a pergunta para PT antes de embeddar "
                  "(TRANSLATE_QUERY_TO_PT=true)")

    abaixo = [q for lang, q in CONSULTAS
              if l2_cos(vs.similarity_search_with_score(q, k=1)[0][1])
              < settings.retrieval_cosine_threshold]
    print(f"\n  limiar configurado: {settings.retrieval_cosine_threshold}")
    print(f"  consultas cujo MELHOR hit fica abaixo dele: {len(abaixo)}")
    for q in abaixo:
        print(f"    - {q}")

    # ---------------------------------------------------------- 7 contaminação
    if not args.probe_only:
        cabecalho("7. CONTAMINAÇÃO DO CORPUS")
        if contaminacao:
            print("  Perguntas normativas cujo melhor hit NÃO é documento de regra:\n")
            for q, doc, c in contaminacao:
                print(f"    {c:.3f}  {doc[:44]:46} <- {q[:30]}")
            print("\n  Material de apoio competindo com o normativo. Considere")
            print("  indexar em coleção separada ou filtrar por tipo de documento.")
        else:
            print("  Nenhuma. Todo melhor hit veio de documento normativo.")

    print("\n" + "=" * L)


if __name__ == "__main__":
    main()