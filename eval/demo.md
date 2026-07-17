# Lumie RAG chatbot - re-evaluation report

Test units: **3** | Seeds: **[13]** | Judge/decomposer: **gpt-4o-mini** (temp 0.2)
Metrics: atomic Precision/Recall/F1 (vs. the gold answer) and FactScore (grounding vs. the official corpus). All values are percentages.

## 1. Overall results

| Slice     | Precision | Recall | F1   | FactScore | n |
|-----------|-----------|--------|------|-----------|---|
| Overall   | 47.8      | 50.0   | 48.5 | 41.1      | 3 |
| lang = en | 83.3      | 75.0   | 78.9 | 83.3      | 1 |
| lang = pt | 30.0      | 37.5   | 33.3 | 20.0      | 2 |

## 2. What the numbers say

Overall the chatbot reaches an F1 of 48.5 (precision 47.8, recall 50.0) and a FactScore of 41.1. Grounding is weak: about 59% of asserted facts are unsupported by the corpus - hallucination is a real problem and is the first thing to fix. Recall is low (50.0): the bot omits a large share of the facts the gold answer contains. In a RAG system this usually points to retrieval gaps - the right chunk was not retrieved or not used. See the missing facts in section 5.

## 3. Performance by intent (weakest first)

| Intent                               | Precision | Recall | F1   | FactScore | n |
|--------------------------------------|-----------|--------|------|-----------|---|
| Login Sistema Antigo / Erro de Senha | 0.0       | 0.0    | 0.0  | 0.0       | 1 |
| Cronograma e Datas                   | 60.0      | 75.0   | 66.7 | 40.0      | 1 |
| Token Invalido ou Expirado           | 83.3      | 75.0   | 78.9 | 83.3      | 1 |

Weakest intents: Login Sistema Antigo / Erro de Senha, Cronograma e Datas, Token Invalido ou Expirado.
Strongest intents: Token Invalido ou Expirado, Cronograma e Datas, Login Sistema Antigo / Erro de Senha.

## 4. Unsupported facts (hallucination set)

7 of 14 predicted atomic facts (50.0%) were not supported by the corpus. Examples (intent -> unsupported claim):

- *Login Sistema Antigo / Erro de Senha*: Para entrar, basta usar a mesma senha do ano passado.
- *Login Sistema Antigo / Erro de Senha*: Se a senha não funcionar, ligue para o número de suporte 0800-123-456.
- *Login Sistema Antigo / Erro de Senha*: O suporte reativa a conta antiga na hora.
- *Cronograma e Datas*: As inscrições acontecem de 06/04 a 19/06.
- *Cronograma e Datas*: A primeira fase, online, ocorre de 04/08 a 06/08.
- *Cronograma e Datas*: O gabarito é divulgado em 20/08/2025.
- *Token Invalido ou Expirado*: If the activation link expired, go to the login page.

## 5. Missing facts (recall gaps)

5 of 11 gold atomic facts (45.5%) were missing from the answers. Examples (intent -> missing fact the answer should have contained):

- *Login Sistema Antigo / Erro de Senha*: A 11a OBG 2026 usa um sistema novo.
- *Login Sistema Antigo / Erro de Senha*: Os dados antigos não foram migrados.
- *Login Sistema Antigo / Erro de Senha*: É preciso fazer um novo cadastro em https://sistema.obgeografia.com.br/signup.
- *Cronograma e Datas*: As inscrições vão de 06/04 a 19/06.
- *Token Invalido ou Expirado*: Users should click the new link quickly.

## 6. How these numbers were produced

Each gold and predicted answer was decomposed into atomic facts by gpt-4o-mini. Recall counts gold facts conveyed in the prediction; precision counts predicted facts inferable from the gold answer; FactScore counts predicted facts supported by the official OBG corpus. F1 is the macro-average of per-question harmonic means. PT<->EN paraphrases are treated as equivalent. Sections 2-5 are generated directly from the per-question results, not written in advance.
