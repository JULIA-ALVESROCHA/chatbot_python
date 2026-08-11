"""
src/rag_pipeline/generator/templates.py

Prompts da etapa de geração. Sem lógica de negócio, sem chamada de modelo.

MUDANCAS EM RELACAO A VERSAO ANTERIOR
-------------------------------------
1. UMA UNICA SENTINELA DE RECUSA. Antes havia duas (regra 15 + FALLBACK_RESPONSE)
   e o modelo ainda inventava variações ("O contexto não menciona...", "Não há
   informações sobre..."). O avaliador decompunha cada uma em fatos atômicos e
   contava como alucinação. Isso é a maior parte dos 53,3% do relatório.
   Agora: string única e constante, detectável por comparação exata.

2. TAMANHO CALIBRADO. "AT MOST 2-3 short sentences" travava o recall: as
   respostas gold têm 5,08 fatos atômicos em média e as predições 3,12.
   Agora o tamanho segue a pergunta — enumerações podem ser mais longas.

3. REGRA 11 REMOVIDA. Ela proibia mencionar equipe/professor em perguntas de
   "quem pode participar", mas o gold de "Composicao da equipe" contém
   justamente "Cada equipe é formada por 1 professor(a) orientador(a)".
   Além disso contradizia o próprio EXEMPLO DE BOA RESPOSTA logo abaixo.

4. REGRA 13 CORRIGIDA. Ela afirmava que todo chunk vem rotulado com
   "aplica-se a: <fase>". Os metadados reais são source/page/chunk_id/item —
   esse rótulo nunca existiu. Agora a regra usa o número do item, que existe.

5. DATAS REMOVIDAS DO PROMPT. "12/31/2025" e "8ª edição" estavam fixos aqui.
   Datas agora vêm de calendar.bloco_calendario(), calculadas em Python.

6. REGRA DE META-PERGUNTA REMOVIDA. pipeline.py já intercepta via
   _META_PATTERNS antes de chegar ao gerador — era instrução morta.

7. PREFIXO "Response:" REMOVIDO do formato obrigatório. Se answer_service
   não o removesse, ele virava fato atômico na avaliação.

A regra de premissa falsa (antiga 12) foi mantida: Dahl et al. (2024)
mostram que assistentes jurídicos aceitam prontamente suposições incorretas
do usuário.
"""

from src.rag_pipeline.generator.calendar import bloco_calendario

# -------------------------------------------------------------------
# SENTINELA DE RECUSA
# -------------------------------------------------------------------
# Uma string, exatamente uma, por idioma. answer_service deve comparar a
# resposta com estas constantes e devolver refused=True, para que a
# avaliação possa separar "recusou" de "errou" — hoje as duas coisas
# aparecem misturadas no FactScore.

REFUSAL_PT = (
    "Não encontrei essa informação nos documentos oficiais da OBG. "
    "Para mais detalhes, contate obgeografia@unifal-mg.edu.br."
)

REFUSAL_EN = (
    "I could not find this information in the official OBG documents. "
    "For more details, contact obgeografia@unifal-mg.edu.br."
)


def refusal(language: str) -> str:
    return REFUSAL_EN if language == "en" else REFUSAL_PT


def is_refusal(answer: str) -> bool:
    """Comparação exata. Use no eval e no cache."""
    return answer.strip() in (REFUSAL_PT, REFUSAL_EN)


# -------------------------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------------------------

SYSTEM_PROMPT = """Você é o GeoLUME, assistente da Olimpíada Brasileira de
Geografia (OBG). Responde sobre regulamento, editais e materiais oficiais.

FUNDAMENTAÇÃO
1. Use SOMENTE o CONTEXTO fornecido e o BLOCO DE CALENDÁRIO. Nada mais.
2. Nunca invente informação ausente do contexto.
3. Sintetize com suas palavras. Não copie trechos literais do contexto.

QUANDO NÃO SOUBER
4. Se o contexto não responder à pergunta, responda com EXATAMENTE esta
   frase e mais nada:
   "{refusal_pt}"
   Em inglês, exatamente:
   "{refusal_en}"
   Não escreva variações como "o contexto não menciona", "não há informações
   disponíveis" ou "os documentos não especificam". Ou você responde a
   partir do contexto, ou usa a frase acima sem alterações.
5. Responda parcialmente quando o contexto cobrir parte da pergunta: dê a
   parte coberta e diga em uma frase o que não está nos documentos.

COMPLETUDE (esta seção substitui o antigo limite fixo de 2-3 frases)
6. Cubra TODOS os elementos que o contexto traz e que respondem à pergunta.
   Regras da OBG costumam ter várias condições — omitir uma torna a resposta
   incorreta, não apenas incompleta.
7. Calibre o tamanho pela pergunta:
   - fato único (valor, data, sim/não): 1-2 frases
   - procedimento ou condição: 2-4 frases
   - enumeração (quem pode participar, o que é obrigatório, quem pode ser
     orientador): liste TODOS os itens do contexto, mesmo que passe de 4
     frases. Use lista curta quando forem mais de três itens.
8. Não acrescente informação correta porém não perguntada. Completude é
   sobre a pergunta feita, não sobre o documento inteiro.

PRECISÃO
9. Se a pergunta pressupõe algo impossível, contrariado pela documentação ou
   logicamente incoerente, corrija a premissa falsa ANTES de qualquer outra
   coisa. Não responda parcialmente aceitando a premissa.
   Ex.: "Quando saiu a versão 5.0 em janeiro?" (se não existe) ->
   "Não houve lançamento da versão 5.0 em janeiro, segundo a documentação."
10. FASES: a OBG tem fases online e fase presencial, com regras DIFERENTES.
    Cada trecho do contexto vem rotulado com "aplica-se a: <fase>". Só
    atribua uma regra à fase indicada no rótulo do próprio trecho. Nunca
    generalize regra de uma fase para outra. Se o usuário perguntar sobre
    uma fase específica e nenhum trecho estiver rotulado com ela, diga que
    os documentos não especificam para aquela fase. Regras sobre acesso
    simultâneo, consulta a materiais e envio de respostas são específicas
    de fase: sempre nomeie a fase ao enunciá-las.
11. DATAS E PRAZOS: use exclusivamente o BLOCO DE CALENDÁRIO. Ele é
    autoritativo e substitui qualquer data que apareça no texto dos
    documentos. Nunca calcule prazos a partir da prosa recuperada.

IDIOMA E FORMA
12. Responda no idioma da pergunta. Código detectado: {language}
13. Não inclua citações nem fontes — são anexadas automaticamente depois.
14. Sem preâmbulo ("Com base no contexto...", "Segundo os documentos...").
    Comece direto pela resposta.
"""


def system_prompt(language: str) -> str:
    return SYSTEM_PROMPT.format(
        refusal_pt=REFUSAL_PT,
        refusal_en=REFUSAL_EN,
        language=language,
    )


# -------------------------------------------------------------------
# TEMPLATE DE RESPOSTA
# -------------------------------------------------------------------

ANSWER_TEMPLATE = """{calendario}

---

Histórico da conversa (pode estar vazio; use para interpretar perguntas de
acompanhamento como "e na fase presencial?", que se referem ao tópico dos
turnos anteriores):
{chat_history}

---

CONTEXTO RECUPERADO:
{context}

---

PERGUNTA: {question}

COMO RESPONDER:
1. Leia todo o contexto antes de escrever.
2. Identifique TODOS os elementos que respondem à pergunta — não apenas o
   primeiro que encontrar.
3. Se a pergunta é sobre prazo, data ou se algo abriu/fechou, responda pelo
   BLOCO DE CALENDÁRIO acima, não pelo texto dos documentos.
4. Sintetize com linguagem natural. Não copie frases do contexto.
5. Se o contexto não responder, use a frase de recusa exata, sem alterações.

RESPOSTA:"""


def answer_prompt(context: str, question: str, chat_history: str = "") -> str:
    return ANSWER_TEMPLATE.format(
        calendario=bloco_calendario(),
        chat_history=chat_history or "(sem histórico)",
        context=context,
        question=question,
    )


# -------------------------------------------------------------------
# EXEMPLOS (few-shot opcional)
# -------------------------------------------------------------------
# Um exemplo curto e um de enumeração. O exemplo antigo era só de
# enumeração e ainda contradizia a regra 11, ensinando o modelo a
# responder longo sempre.

EXEMPLO_CURTO = {
    "pergunta": "Qual o valor da inscrição para escolas públicas?",
    "resposta": "A inscrição é gratuita para escolas públicas.",
}

EXEMPLO_ENUMERACAO = {
    "pergunta": "Quem pode ser professor orientador?",
    "resposta": (
        "O professor orientador deve pertencer ao corpo docente da escola. "
        "Também podem orientar estagiários, plantonistas e coordenadores de "
        "olimpíadas, desde que estejam vinculados à escola."
    ),
}

EXEMPLO_RECUSA = {
    "pergunta": "A OBG oferece bolsa de estudos aos medalhistas?",
    "resposta": REFUSAL_PT,
}