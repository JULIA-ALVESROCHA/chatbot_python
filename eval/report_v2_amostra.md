# Lumie RAG chatbot - re-evaluation report

Test units: **50** | Seeds: **[42]** | Judge/decomposer: **gpt-4o-mini** (temp 0.2)
Metrics: atomic Precision/Recall/F1 (vs. the gold answer) and FactScore (grounding vs. the official corpus). All values are percentages.

## 1. Overall results

| Slice     | Precision | Recall | F1   | FactScore | n  |
|-----------|-----------|--------|------|-----------|----|
| Overall   | 50.8      | 39.2   | 39.6 | 61.8      | 50 |
| lang = en | 47.0      | 41.3   | 40.3 | 67.8      | 11 |
| lang = pt | 51.8      | 38.6   | 39.3 | 60.1      | 39 |

## 2. What the numbers say

Overall the chatbot reaches an F1 of 39.6 (precision 50.8, recall 39.2) and a FactScore of 61.8. Grounding is weak: about 38% of asserted facts are unsupported by the corpus - hallucination is a real problem and is the first thing to fix. FactScore (61.8) is clearly higher than precision-vs-gold (50.8). That gap means the bot adds true, corpus-grounded details that are not in the short canonical answer - i.e. it is verbose rather than wrong. This is a style issue, not a factual one. Recall is low (39.2): the bot omits a large share of the facts the gold answer contains. In a RAG system this usually points to retrieval gaps - the right chunk was not retrieved or not used. See the missing facts in section 5.

## 3. Performance by intent (weakest first)

| Intent                                    | Precision | Recall | F1    | FactScore | n |
|-------------------------------------------|-----------|--------|-------|-----------|---|
| Substituicao de membros                   | 0.0       | 0.0    | 0.0   | 0.0       | 1 |
| Navegadores suportados                    | 0.0       | 0.0    | 0.0   | 0.0       | 1 |
| Prazo de preenchimento de dados           | 50.0      | 0.0    | 0.0   | 0.0       | 1 |
| Quem pode ser professor orientador        | 33.3      | 0.0    | 0.0   | 66.7      | 1 |
| Calendario / datas                        | 0.0       | 0.0    | 0.0   | 0.0       | 1 |
| Temas e ODS das questoes                  | 2.7       | 30.0   | 5.0   | 4.1       | 2 |
| Responder todas as questoes               | 16.7      | 20.0   | 18.2  | 50.0      | 2 |
| Minimo de membros na presencial           | 16.7      | 25.0   | 20.0  | 50.0      | 2 |
| Certificados                              | 40.0      | 14.3   | 21.1  | 80.0      | 1 |
| Diretrizes de elaboracao de questoes      | 33.3      | 16.7   | 22.2  | 100.0     | 1 |
| Contato oficial                           | 50.0      | 16.7   | 25.0  | 100.0     | 1 |
| Criterios de desempate                    | 33.3      | 25.0   | 28.6  | 66.7      | 1 |
| Tipos de escola                           | 28.6      | 58.3   | 33.9  | 92.9      | 2 |
| Classificacao para a presencial           | 43.1      | 33.3   | 34.8  | 43.1      | 3 |
| Valor da inscricao                        | 38.9      | 50.0   | 35.7  | 38.9      | 3 |
| Cadastro nao e inscricao                  | 66.7      | 25.0   | 36.4  | 66.7      | 1 |
| Envio de respostas                        | 75.0      | 27.8   | 39.1  | 91.7      | 3 |
| Recuperar senha                           | 50.0      | 33.3   | 40.0  | 70.0      | 1 |
| Fases e estrutura das provas              | 66.7      | 28.6   | 40.0  | 100.0     | 2 |
| Corrigir dados da escola                  | 75.0      | 33.3   | 43.7  | 33.3      | 3 |
| Senha recuperada incompativel / navegador | 55.7      | 40.0   | 45.9  | 90.0      | 2 |
| Divulgacao do gabarito                    | 50.0      | 50.0   | 50.0  | 50.0      | 1 |
| Composicao da equipe                      | 50.0      | 50.0   | 50.0  | 75.0      | 2 |
| Quem pode participar / series             | 56.7      | 50.0   | 52.5  | 83.3      | 2 |
| Recursos de questao                       | 60.0      | 50.0   | 54.5  | 80.0      | 1 |
| Nome da equipe                            | 100.0     | 40.0   | 57.1  | 0.0       | 1 |
| Equipes de escolas diferentes             | 83.3      | 50.0   | 61.9  | 83.3      | 2 |
| Limite de equipes                         | 66.7      | 100.0  | 80.0  | 100.0     | 1 |
| Nome social                               | 90.0      | 75.0   | 81.6  | 90.0      | 2 |
| Aluno em mais de uma equipe               | 75.0      | 100.0  | 85.7  | 50.0      | 1 |
| Medalhas fisicas                          | 100.0     | 100.0  | 100.0 | 100.0     | 2 |

Weakest intents: Substituicao de membros, Navegadores suportados, Prazo de preenchimento de dados, Quem pode ser professor orientador, Calendario / datas.
Strongest intents: Medalhas fisicas, Aluno em mais de uma equipe, Nome social.

## 4. Unsupported facts (hallucination set)

105 of 231 predicted atomic facts (45.5%) were not supported by the corpus. Examples (intent -> unsupported claim):

- *Substituicao de membros*: Não encontrei essa informação nos documentos oficiais da OBG.
- *Substituicao de membros*: Para mais detalhes, contate obgeografia@unifal-mg.edu.br.
- *Nome da equipe*: Não é possível corrigir o nome da equipe uma vez que ele foi escolhido e registrado.
- *Nome da equipe*: Os nomes das equipes não serão alterados, salvo exceções mencionadas nos regulamentos.
- *Nome da equipe*: É importante que a equipe verifique se realmente deseja aquele nome antes de finalizar a inscrição.
- *Divulgacao do gabarito*: A divulgação dos gabaritos das Olimpíadas Brasileiras de Geografia ocorrerá no dia 20/08/2026.
- *Classificacao para a presencial*: Os critérios de classificação para a fase presencial das Olimpíadas Brasileiras de Geografia incluem a divulgação dos nomes das equipes classificadas.
- *Classificacao para a presencial*: A divulgação dos nomes das equipes classificadas ocorrerá em 01/09/2026.
- *Classificacao para a presencial*: As medalhas serão disponibilizadas com base na classificação das equipes nas três fases online.
- *Classificacao para a presencial*: As medalhas também serão disponibilizadas com base no resultado da fase final presencial.
- *Classificacao para a presencial*: Não há informações adicionais sobre outros critérios de classificação nos documentos.
- *Valor da inscricao*: As escolas particulares devem pagar R$ 65,00 por equipe para participar da Olimpíada Brasileira de Geografia.

## 5. Missing facts (recall gaps)

158 of 242 gold atomic facts (65.3%) were missing from the answers. Examples (intent -> missing fact the answer should have contained):

- *Substituicao de membros*: O professor coordenador pode substituir qualquer membro da equipe antes do início da primeira fase online.
- *Substituicao de membros*: Depois do início da primeira fase online, o sistema permite apenas a exclusão do estudante.
- *Substituicao de membros*: Equipes classificadas para a Presencial nacional não podem substituir membros.
- *Contato oficial*: As informações oficiais são divulgadas exclusivamente no site obgeografia.com.br.
- *Contato oficial*: As respostas para dúvidas e recursos são dadas em até 4 dias úteis.
- *Contato oficial*: Não há atendimento por telefone.
- *Contato oficial*: Não há atendimento por e-mail pessoal.
- *Contato oficial*: Não há atendimento por redes sociais pessoais.
- *Navegadores suportados*: A Comissao Organizadora garante o funcionamento para as versoes mais recentes do Firefox.
- *Navegadores suportados*: A Comissao Organizadora garante o funcionamento para as versoes mais recentes do Google Chrome.
- *Navegadores suportados*: Nao e recomendado usar versoes antigas do Firefox.
- *Navegadores suportados*: Nao e recomendado usar versoes antigas do Google Chrome.

## 6. How these numbers were produced

Each gold and predicted answer was decomposed into atomic facts by gpt-4o-mini. Recall counts gold facts conveyed in the prediction; precision counts predicted facts inferable from the gold answer; FactScore counts predicted facts supported by the official OBG corpus. F1 is the macro-average of per-question harmonic means. PT<->EN paraphrases are treated as equivalent. Sections 2-5 are generated directly from the per-question results, not written in advance.
