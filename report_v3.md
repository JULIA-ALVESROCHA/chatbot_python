# Lumie RAG chatbot - re-evaluation report

Test units: **200** | Seeds: **[13, 21, 42]** | Judge/decomposer: **gpt-4o-mini** (temp 0.2)
Metrics: atomic Precision/Recall/F1 (vs. the gold answer) and FactScore (grounding vs. the official corpus). All values are percentages, mean ± std across seeds.

## 1. Overall results

| Slice     | Precision   | Recall      | F1          | FactScore   | n   |
|-----------|-------------|-------------|-------------|-------------|-----|
| Overall   | 56.6 ± 0.2  | 22.2 ± 0.1  | 32.6 ± 0.2  | 74.9 ± 0.4  | 200 |
| lang = en | 46.9 ± 40.5 | 14.8 ± 27.4 | 23.1 ± 31.9 | 68.5 ± 33.7 | 126 |
| lang = pt | 58.9 ± 36.1 | 24.2 ± 27.9 | 34.9 ± 29.7 | 76.4 ± 29.9 | 474 |

## 2. What the numbers say

Overall the chatbot reaches an F1 of 32.6 (precision 56.6, recall 22.2) and a FactScore of 74.9. Grounding is weak: about 25% of asserted facts are unsupported by the corpus - hallucination is a real problem and is the first thing to fix. FactScore (74.9) is clearly higher than precision-vs-gold (56.6). That gap means the bot adds true, corpus-grounded details that are not in the short canonical answer - i.e. it is verbose rather than wrong. This is a style issue, not a factual one. Recall is low (22.2): the bot omits a large share of the facts the gold answer contains. In a RAG system this usually points to retrieval gaps - the right chunk was not retrieved or not used. See the missing facts in section 5. Results are stable across seeds (low standard deviation), so the scores are reliable.

## 3. Performance by intent (weakest first)

| Intent                                    | Precision   | Recall      | F1          | FactScore   | n  |
|-------------------------------------------|-------------|-------------|-------------|-------------|----|
| Minimo de membros na presencial           | 44.4 ± 7.9  | 0.0 ± 0.0   | 0.0 ± 0.0   | 85.2 ± 22.8 | 12 |
| Quem pode ser professor orientador        | 13.3 ± 17.2 | 0.0 ± 0.0   | 0.0 ± 0.0   | 38.3 ± 24.5 | 15 |
| Token / link expirado                     | 33.3 ± 0.0  | 0.0 ± 0.0   | 0.0 ± 0.0   | 44.4 ± 15.7 | 12 |
| Calendario / datas                        | 25.0 ± 25.0 | 0.0 ± 0.0   | 0.0 ± 0.0   | 25.0 ± 25.0 | 18 |
| Dados obrigatorios do estudante           | 11.1 ± 15.7 | 1.7 ± 3.3   | 4.4 ± 6.3   | 80.6 ± 14.2 | 15 |
| Inscricao exclusivamente online           | 11.1 ± 15.7 | 5.1 ± 11.2  | 9.3 ± 13.6  | 100.0 ± 0.0 | 15 |
| E-mail de confirmacao nao chega           | 30.0 ± 30.0 | 3.3 ± 6.7   | 13.0 ± 13.0 | 90.0 ± 10.0 | 15 |
| Cadastro da escola pelo INEP              | 22.2 ± 15.7 | 12.5 ± 21.7 | 13.3 ± 18.9 | 64.4 ± 3.1  | 12 |
| Composicao da equipe                      | 36.7 ± 35.6 | 18.5 ± 16.6 | 15.6 ± 19.4 | 81.1 ± 21.0 | 18 |
| Criterios de desempate                    | 20.8 ± 21.7 | 20.8 ± 13.8 | 16.9 ± 17.7 | 83.3 ± 16.7 | 12 |
| Classificacao para a presencial           | 25.0 ± 43.3 | 15.0 ± 27.8 | 18.2 ± 32.7 | 52.1 ± 29.1 | 12 |
| Fases e estrutura das provas              | 71.1 ± 21.2 | 14.3 ± 16.5 | 18.3 ± 19.8 | 86.9 ± 13.6 | 18 |
| Temas e ODS das questoes                  | 73.3 ± 29.7 | 11.1 ± 15.2 | 18.4 ± 15.6 | 93.8 ± 10.8 | 18 |
| Recuperar senha                           | 43.8 ± 44.6 | 10.0 ± 13.3 | 18.7 ± 19.7 | 60.4 ± 43.2 | 15 |
| Responder todas as questoes               | 52.8 ± 38.1 | 18.3 ± 19.1 | 19.8 ± 23.7 | 75.0 ± 35.4 | 12 |
| Corrigir dados da escola                  | 71.7 ± 32.8 | 9.6 ± 9.0   | 20.0 ± 13.5 | 61.7 ± 12.6 | 15 |
| Selecao para a iGeo                       | 57.3 ± 41.0 | 11.9 ± 9.8  | 21.3 ± 13.1 | 58.4 ± 31.3 | 18 |
| Cadastro nao e inscricao                  | 33.3 ± 23.6 | 12.5 ± 12.5 | 22.2 ± 15.7 | 66.7 ± 23.6 | 12 |
| Questao anulada                           | 16.7 ± 23.6 | 25.0 ± 43.3 | 22.2 ± 31.4 | 47.2 ± 27.5 | 12 |
| Quem pode participar / series             | 61.0 ± 26.6 | 15.2 ± 8.5  | 22.7 ± 13.1 | 80.5 ± 25.8 | 21 |
| Diretrizes de elaboracao de questoes      | 47.6 ± 8.7  | 11.3 ± 12.1 | 23.5 ± 11.5 | 88.5 ± 13.6 | 18 |
| Contato oficial                           | 46.7 ± 32.3 | 15.7 ± 15.2 | 24.5 ± 15.1 | 73.3 ± 38.9 | 18 |
| Certificados                              | 48.3 ± 30.0 | 17.1 ± 10.7 | 24.9 ± 15.2 | 55.0 ± 34.8 | 15 |
| Professor gera senha do estudante         | 45.0 ± 27.6 | 18.8 ± 14.0 | 26.0 ± 18.1 | 77.1 ± 30.8 | 12 |
| Recursos de questao                       | 53.3 ± 19.6 | 15.3 ± 18.6 | 26.5 ± 21.9 | 83.3 ± 14.7 | 12 |
| Substituicao de membros                   | 59.7 ± 26.8 | 20.0 ± 26.7 | 28.3 ± 32.8 | 83.3 ± 16.7 | 15 |
| Uso de IA generativa                      | 83.3 ± 16.7 | 10.0 ± 10.0 | 32.1 ± 1.3  | 83.3 ± 16.7 | 12 |
| Nome da equipe                            | 75.0 ± 43.3 | 20.0 ± 17.9 | 36.9 ± 23.4 | 8.3 ± 14.4  | 15 |
| Acesso simultaneo durante a prova         | 70.8 ± 29.8 | 29.2 ± 19.7 | 38.6 ± 26.9 | 62.5 ± 41.5 | 12 |
| Envio de respostas                        | 100.0 ± 0.0 | 25.0 ± 8.3  | 39.3 ± 10.7 | 100.0 ± 0.0 | 12 |
| Valor da inscricao                        | 50.0 ± 35.4 | 27.8 ± 29.9 | 44.4 ± 29.1 | 62.5 ± 41.5 | 18 |
| Prazo de preenchimento de dados           | 87.5 ± 21.7 | 31.2 ± 10.8 | 45.0 ± 12.8 | 100.0 ± 0.0 | 12 |
| Tipos de escola                           | 66.7 ± 27.2 | 29.2 ± 34.1 | 45.1 ± 37.1 | 94.4 ± 7.9  | 12 |
| Certificados de edicoes anteriores        | 52.8 ± 33.6 | 37.5 ± 21.7 | 46.7 ± 14.4 | 80.6 ± 22.9 | 12 |
| Navegadores suportados                    | 75.0 ± 25.0 | 19.8 ± 20.1 | 50.4 ± 7.4  | 87.5 ± 12.5 | 12 |
| Limite de equipes                         | 60.0 ± 37.4 | 50.0 ± 31.6 | 53.3 ± 32.3 | 76.7 ± 22.6 | 15 |
| Senha recuperada incompativel / navegador | 75.0 ± 0.0  | 15.0 ± 26.0 | 66.7 ± 0.0  | 100.0 ± 0.0 | 12 |
| Equipes de escolas diferentes             | 86.0 ± 28.2 | 62.2 ± 16.6 | 68.9 ± 18.9 | 90.7 ± 18.8 | 15 |
| Nome social                               | 91.7 ± 18.6 | 68.8 ± 10.8 | 76.7 ± 10.9 | 87.5 ± 21.7 | 12 |
| Medalhas fisicas                          | 86.7 ± 26.7 | 73.3 ± 24.9 | 78.7 ± 24.4 | 93.3 ± 13.3 | 15 |
| Divulgacao do gabarito                    | 100.0 ± 0.0 | 62.5 ± 41.5 | 88.9 ± 15.7 | 66.7 ± 23.6 | 12 |
| Aluno em mais de uma equipe               | 95.6 ± 11.3 | 90.0 ± 20.0 | 92.1 ± 16.0 | 100.0 ± 0.0 | 15 |

Weakest intents: Minimo de membros na presencial, Quem pode ser professor orientador, Token / link expirado, Calendario / datas, Dados obrigatorios do estudante.
Strongest intents: Aluno em mais de uma equipe, Divulgacao do gabarito, Medalhas fisicas.

## 4. Unsupported facts (hallucination set)

361 of 1469 predicted atomic facts (24.6%) were not supported by the corpus. Examples (intent -> unsupported claim):

- *Limite de equipes*: O orientador deve inscrever apenas a quantidade de equipes que consegue acompanhar adequadamente.
- *Quem pode ser professor orientador*: O professor orientador deve comprometer-se a seguir o regulamento da olimpíada.
- *Quem pode participar / series*: A participação é válida até 30 de junho de 2027.
- *Quem pode participar / series*: Os estudantes devem ter idades entre 16 e 19 anos em 30 de junho de 2027.
- *Cadastro nao e inscricao*: A equipe não está oficialmente inscrita na Olimpíada Brasileira de Geografia apenas após o cadastro.
- *Cadastro da escola pelo INEP*: Os dados que o professor deve atualizar incluem o nome completo do diretor.
- *Cadastro da escola pelo INEP*: Os dados que o professor deve atualizar incluem o nome completo da coordenação pedagógica.
- *Valor da inscricao*: O valor da inscrição para escolas particulares não foi mencionado nos documentos oficiais.
- *Acesso simultaneo durante a prova*: Não há informações nos documentos oficiais da OBG que permitam a realização simultânea da prova por três alunos.
- *Acesso simultaneo durante a prova*: A Olimpíada não possui meios de interferir na realização da prova.
- *Acesso simultaneo durante a prova*: A Olimpíada não possui meios de interferir nas tarefas das equipes.
- *Divulgacao do gabarito*: A data exata de divulgação do gabarito oficial não está especificada nos documentos.

## 5. Missing facts (recall gaps)

2524 of 3047 gold atomic facts (82.8%) were missing from the answers. Examples (intent -> missing fact the answer should have contained):

- *Limite de equipes*: Nao ha limite de equipes que uma escola pode inscrever.
- *Equipes de escolas diferentes*: As escolas diferentes podem ser da mesma rede de ensino ou mantenedora.
- *Nome social*: É garantido o uso do nome social durante toda a prova.
- *Nome social*: O uso do nome social é em cumprimento ao Decreto no 8.727/2016.
- *Composicao da equipe*: Cada equipe é formada por 1 professor(a) orientador(a).
- *Composicao da equipe*: O professor(a) orientador(a) é o responsável pela inscrição.
- *Minimo de membros na presencial*: Equipes classificadas para a Fase Presencial nao podem participar com menos de dois membros.
- *Minimo de membros na presencial*: Equipes que participarem com apenas dois membros nao concorrem a premiacoes por equipe.
- *Dados obrigatorios do estudante*: O estudante deve informar CPF no cadastro.
- *Dados obrigatorios do estudante*: O estudante deve informar e-mail válido no cadastro.
- *Dados obrigatorios do estudante*: O estudante deve informar nome completo no cadastro.
- *Dados obrigatorios do estudante*: O estudante deve informar celular WhatsApp no cadastro.

## 6. How these numbers were produced

Each gold and predicted answer was decomposed into atomic facts by gpt-4o-mini. Recall counts gold facts conveyed in the prediction; precision counts predicted facts inferable from the gold answer; FactScore counts predicted facts supported by the official OBG corpus. F1 is the macro-average of per-question harmonic means. PT<->EN paraphrases are treated as equivalent. Sections 2-5 are generated directly from the per-question results, not written in advance.
