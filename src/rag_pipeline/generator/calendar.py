"""
src/rag_pipeline/generator/calendar.py

Calendário da OBG com atualização automática.

PROBLEMA
Prazos são prorrogados. Uma data fixa no código erra com confiança máxima
justamente no dia em que a prorrogação é anunciada — e o bot continua
dizendo "encerradas" para centenas de professores.

ARQUITETURA (precedência)
  1. REMOTO   planilha Google publicada como CSV (a comissão edita a célula)
  2. LOCAL    data/calendar.yml no repositório (se a rede falhar)
  3. CÓDIGO   DEFAULTS abaixo (se tudo falhar)

TTL ADAPTATIVO
Longe de prazos, 12h. Perto de uma fronteira, 15min. Prorrogações acontecem
exatamente nas fronteiras — é lá que o cache precisa ser curto.

INCERTEZA
Nos GRACE_DAYS após o fim de um prazo, o bot NÃO afirma categoricamente que
encerrou. Ele informa o que o calendário registra, quando foi atualizado, e
manda conferir o site. Uma prorrogação anunciada e ainda não propagada cai
exatamente nessa janela.

COMO A COMISSÃO PUBLICA A PLANILHA
  Arquivo > Compartilhar > Publicar na web > CSV > escolher a aba
  Colar a URL em CALENDAR_URL (.env)
  Colunas: chave, rotulo, inicio, fim, nota      (datas em AAAA-MM-DD)
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger("bgo_chatbot.calendar")

TZ = ZoneInfo("America/Sao_Paulo")

CALENDAR_URL = os.getenv("CALENDAR_URL", "https://docs.google.com/spreadsheets/d/e/2PACX-1vTeqs7tTMGjsFILlpynx_bNtcF2nvhGhaXJ7IAKNi_N656blnl90QUVaKLE_-TRlawCV6AOl3r3PeE3/pub?output=csv").strip()
CALENDAR_YML = Path(os.getenv("CALENDAR_YML", "data/calendar.yml"))
CACHE_FILE = Path(os.getenv("CALENDAR_CACHE", "data/processed/.calendar_cache.json"))
SITE_OFICIAL = os.getenv("OBG_SITE", "https://obg.unifal-mg.edu.br")

# TTL em segundos conforme a proximidade da fronteira mais próxima
TTL_CRITICO = 15 * 60        # <= 3 dias de alguma fronteira
TTL_PROXIMO = 60 * 60        # <= 14 dias
TTL_NORMAL = 12 * 3600       # resto do ano
DIAS_CRITICO = 3
DIAS_PROXIMO = 14

# Janela após o fim de um prazo em que o bot hedgeia em vez de afirmar
GRACE_DAYS = 7


@dataclass(frozen=True)
class Evento:
    chave: str
    rotulo: str
    inicio: Optional[date]
    fim: Optional[date]
    nota: str = ""


DEFAULTS: List[Evento] = [
    Evento("inscricoes", "Inscrições", date(2026, 3, 2), date(2026, 5, 15)),
    Evento("fase1", "1ª Fase (online)", date(2026, 6, 10), date(2026, 6, 10)),
    Evento("fase2", "2ª Fase (online)", date(2026, 7, 8), date(2026, 7, 8)),
    Evento("resultado_fase2", "Resultado da 2ª Fase",
           date(2026, 7, 29), date(2026, 7, 29)),
    Evento("fase_presencial", "Fase Presencial",
           date(2026, 9, 12), date(2026, 9, 13)),
    Evento("certificados", "Emissão de certificados",
           date(2026, 10, 1), date(2026, 12, 31)),
]


# ============================================================ estado
_lock = threading.Lock()
_eventos: List[Evento] = list(DEFAULTS)
_fonte: str = "codigo"
_buscado_em: float = 0.0
_atualizado_em: Optional[str] = None  # carimbo declarado pela fonte


def _hoje() -> date:
    return datetime.now(TZ).date()


def _parse_data(v) -> Optional[date]:
    if not v:
        return None
    if isinstance(v, date):
        return v
    s = str(v).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    logger.warning("Data ilegível no calendário: %r", s)
    return None


# ============================================================ fontes
def _do_csv(texto: str) -> List[Evento]:
    eventos = []
    for row in csv.DictReader(io.StringIO(texto)):
        row = { (k or "").strip().lower(): (v or "").strip()
                for k, v in row.items() }
        chave = row.get("chave")
        if not chave:
            continue
        eventos.append(Evento(
            chave=chave,
            rotulo=row.get("rotulo") or chave,
            inicio=_parse_data(row.get("inicio")),
            fim=_parse_data(row.get("fim")),
            nota=row.get("nota", ""),
        ))
    return eventos


def _do_remoto() -> Optional[List[Evento]]:
    if not CALENDAR_URL:
        return None
    try:
        import urllib.request
        req = urllib.request.Request(
            CALENDAR_URL, headers={"User-Agent": "GeoLUME/1.0"}
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            texto = r.read().decode("utf-8")
    except Exception as e:
        logger.warning("Calendário remoto indisponível (%s)", e)
        return None

    try:
        eventos = _do_csv(texto)
    except Exception as e:
        logger.warning("Calendário remoto ilegível (%s)", e)
        return None

    if not eventos:
        logger.warning("Calendário remoto vazio; mantendo fonte anterior")
        return None
    return eventos


def _do_yaml() -> Optional[List[Evento]]:
    if not CALENDAR_YML.exists():
        return None
    try:
        import yaml
        dados = yaml.safe_load(CALENDAR_YML.read_text(encoding="utf-8")) or {}
    except Exception as e:
        logger.warning("calendar.yml ilegível (%s)", e)
        return None

    eventos = []
    for chave, d in (dados.get("eventos") or {}).items():
        eventos.append(Evento(
            chave=chave,
            rotulo=d.get("rotulo", chave),
            inicio=_parse_data(d.get("inicio")),
            fim=_parse_data(d.get("fim")),
            nota=d.get("nota", ""),
        ))
    return eventos or None


# ============================================================ cache em disco
def _salvar_cache() -> None:
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fonte": _fonte,
            "buscado_em": _buscado_em,
            "atualizado_em": _atualizado_em,
            "eventos": [
                {**asdict(e),
                 "inicio": e.inicio.isoformat() if e.inicio else None,
                 "fim": e.fim.isoformat() if e.fim else None}
                for e in _eventos
            ],
        }
        tmp = CACHE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, CACHE_FILE)
    except OSError as e:
        logger.debug("Não foi possível gravar cache do calendário: %s", e)


def _carregar_cache() -> bool:
    """Evita refetch a cada restart (importante em free tier que dorme)."""
    global _eventos, _fonte, _buscado_em, _atualizado_em
    try:
        d = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    try:
        _eventos = [
            Evento(chave=e["chave"], rotulo=e["rotulo"],
                   inicio=_parse_data(e["inicio"]), fim=_parse_data(e["fim"]),
                   nota=e.get("nota", ""))
            for e in d["eventos"]
        ]
        _fonte = d.get("fonte", "cache")
        _buscado_em = float(d.get("buscado_em", 0))
        _atualizado_em = d.get("atualizado_em")
        return True
    except (KeyError, TypeError):
        return False


# ============================================================ TTL adaptativo
def _dias_ate_fronteira(eventos: List[Evento]) -> int:
    hoje = _hoje()
    dists = []
    for ev in eventos:
        for d in (ev.inicio, ev.fim):
            if d:
                dists.append(abs((d - hoje).days))
    return min(dists) if dists else 999


def _ttl_atual() -> int:
    d = _dias_ate_fronteira(_eventos)
    if d <= DIAS_CRITICO:
        return TTL_CRITICO
    if d <= DIAS_PROXIMO:
        return TTL_PROXIMO
    return TTL_NORMAL


def _atualizar(forcar: bool = False) -> None:
    global _eventos, _fonte, _buscado_em, _atualizado_em

    with _lock:
        if _buscado_em == 0.0 and _carregar_cache():
            pass  # cache em disco carregado

        if not forcar and (time.time() - _buscado_em) < _ttl_atual():
            return

        novos = _do_remoto()
        fonte = "remoto"
        if novos is None:
            novos = _do_yaml()
            fonte = "yaml"
        if novos is None:
            # Mantém o que já estava: nunca regride para DEFAULTS se já
            # tínhamos algo mais recente.
            if _fonte != "codigo":
                _buscado_em = time.time()  # espera o próximo TTL
                return
            novos, fonte = list(DEFAULTS), "codigo"

        _eventos = novos
        _fonte = fonte
        _buscado_em = time.time()
        _atualizado_em = datetime.now(TZ).isoformat(timespec="minutes")
        _salvar_cache()
        logger.info("Calendário atualizado via %s (%d eventos, próximo TTL %ds)",
                    fonte, len(novos), _ttl_atual())


# ============================================================ redação
def _status(ev: Evento, hoje: date) -> str:
    if ev.inicio is None or ev.fim is None:
        return "SEM DATA DEFINIDA — conferir no site oficial"

    pontual = ev.inicio == ev.fim

    if hoje < ev.inicio:
        dias = (ev.inicio - hoje).days
        if pontual:
            return f"AINDA NÃO OCORREU — {ev.inicio:%d/%m/%Y} (faltam {dias} dias)"
        return (f"AINDA NÃO ABRIU — abre {ev.inicio:%d/%m/%Y} (faltam {dias} dias), "
                f"fecha {ev.fim:%d/%m/%Y}")

    if hoje > ev.fim:
        dias = (hoje - ev.fim).days
        verbo = "REALIZADA" if pontual else "ENCERRADAS"
        # Pontual (data única) não tem início separado do fim; período tem.
        quando = f"em {ev.fim:%d/%m/%Y}" if pontual else f"de {ev.inicio:%d/%m/%Y} a {ev.fim:%d/%m/%Y}"
        if dias <= GRACE_DAYS and not pontual:
            # Janela de incerteza: prorrogação recém-anunciada pode não ter
            # chegado aqui ainda. Não afirmar categoricamente.
            return (f"CONSTAM COMO {verbo} ({quando}) (há {dias} dias). "
                    f"ATENÇÃO: prazo encerrado há pouco — se houve prorrogação, "
                    f"ela pode ainda não constar aqui. Oriente o usuário a "
                    f"confirmar em {SITE_OFICIAL}")
        return f"{verbo} {quando} (há {dias} dias)"

    if pontual:
        return f"É HOJE ({ev.inicio:%d/%m/%Y})"

    restam = (ev.fim - hoje).days
    if restam <= DIAS_CRITICO:
        return (f"ABERTAS — de {ev.inicio:%d/%m/%Y}, encerram {ev.fim:%d/%m/%Y} "
                f"(restam {restam} dias). Prazo próximo do fim; pode haver prorrogação")
    return f"ABERTAS — de {ev.inicio:%d/%m/%Y} até {ev.fim:%d/%m/%Y} (restam {restam} dias)"


def bloco_calendario() -> str:
    """Bloco autoritativo, injetado ACIMA do contexto recuperado."""
    _atualizar()
    hoje = _hoje()

    idade_h = (time.time() - _buscado_em) / 3600 if _buscado_em else 999
    proc = "atualizado agora" if idade_h < 1 else f"atualizado há {idade_h:.0f}h"

    linhas = [
        f"DATA DE HOJE: {hoje:%d/%m/%Y} (America/Sao_Paulo)",
        f"CALENDÁRIO: fonte={_fonte}, {proc}",
        "",
        "ESTADO DO CALENDÁRIO — AUTORITATIVO.",
        "Substitui QUALQUER data encontrada nos documentos recuperados.",
        "Para prazos, datas ou se algo abriu/fechou, use APENAS este bloco.",
        "Nunca calcule datas a partir da prosa dos documentos.",
        "Quando uma linha pedir confirmação no site, repasse essa ressalva",
        "ao usuário em vez de afirmar categoricamente.",
        "",
    ]
    for ev in _eventos:
        linhas.append(f"- {ev.rotulo}: {_status(ev, hoje)}")
        if ev.nota:
            # A comissão edita "nota" livremente na planilha e às vezes
            # mistura ali informação de outra unidade (valores, regras) que
            # nada tem a ver com data/prazo. Linha própria + instrução
            # inline, para não virar parte do "use APENAS este bloco" acima.
            linhas.append(
                f'  detalhe de "{ev.rotulo}" (inclua na resposta só se a '
                f"pergunta pedir especificamente esse detalhe — não é parte "
                f"do prazo/data, pode ser sobre outro assunto): {ev.nota}"
            )

    if _fonte == "codigo":
        linhas += ["", "AVISO: calendário vindo de valores padrão do código. "
                       f"Pode estar desatualizado — direcione ao {SITE_OFICIAL}."]
    return "\n".join(linhas)


def forcar_atualizacao() -> dict:
    """Endpoint administrativo: POST /admin/calendar/refresh"""
    _atualizar(forcar=True)
    return {"fonte": _fonte, "eventos": len(_eventos),
            "atualizado_em": _atualizado_em, "ttl_segundos": _ttl_atual()}


def inscricoes_abertas() -> Optional[bool]:
    """None = incerto (dentro da janela de graça). Use no banner do frontend."""
    _atualizar()
    hoje = _hoje()
    for ev in _eventos:
        if ev.chave == "inscricoes" and ev.inicio and ev.fim:
            if ev.inicio <= hoje <= ev.fim:
                return True
            if 0 < (hoje - ev.fim).days <= GRACE_DAYS:
                return None
            return False
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(bloco_calendario())
    print("\nTTL atual:", _ttl_atual(), "segundos")