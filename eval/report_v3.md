# Lumie RAG chatbot - re-evaluation report

Test units: **200** | Seeds: **[13, 21, 42]** | Judge/decomposer: **gpt-4o-mini** (temp 0.2)
Metrics: atomic Precision/Recall/F1 (vs. the gold answer) and FactScore (grounding vs. the official corpus). All values are percentages, mean ± std across seeds.

## 1. Overall results

| Slice     | Precision   | Recall      | F1          | FactScore   | n   |
|-----------|-------------|-------------|-------------|-------------|-----|
| Overall   | 41.3 ± 0.4  | 31.9 ± 0.0  | 31.3 ± 0.1  | 69.7 ± 0.1  | 200 |
| lang = en | 34.4 ± 32.7 | 30.3 ± 30.4 | 27.5 ± 27.4 | 66.1 ± 42.0 | 126 |
| lang = pt | 43.2 ± 36.1 | 32.3 ± 32.7 | 32.4 ± 30.8 | 70.7 ± 39.7 | 474 |

## 2. What the numbers say

Overall the chatbot reaches an F1 of 31.3 (precision 41.3, recall 31.9) and a FactScore of 69.7. Grounding is weak: about 30% of asserted facts are unsupported by the corpus - hallucination is a real problem and is the first thing to fix. FactScore (69.7) is clearly higher than precision-vs-gold (41.3). That gap means the bot adds true, corpus-grounded details that are not in the short canonical answer - i.e. it is verbose rather than wrong. This is a style issue, not a factual one. Recall is low (31.9): the bot omits a large share of the facts the gold answer contains. In a RAG system this usually points to retrieval gaps - the right chunk was not retrieved or not used. See the missing facts in section 5. Results are stable across seeds (low standard deviation), so the scores are reliable.

## 3. Performance by intent (weakest first)

| Intent                                    | Precision   | Recall      | F1          | FactScore   | n  |
|-------------------------------------------|-------------|-------------|-------------|-------------|----|
| Quem pode ser professor orientador        | 16.7 ± 21.1 | 0.0 ± 0.0   | 0.0 ± 0.0   | 70.7 ± 37.4 | 15 |
| Token / link expirado                     | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 12 |
| E-mail de confirmacao nao chega           | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 20.0 ± 40.0 | 15 |
| Calendario / datas                        | 2.2 ± 6.3   | 0.0 ± 0.0   | 0.0 ± 0.0   | 55.6 ± 41.8 | 18 |
| Dados obrigatorios do estudante           | 6.7 ± 14.3  | 1.7 ± 3.4   | 2.7 ± 5.4   | 60.0 ± 49.0 | 15 |
| Navegadores suportados                    | 25.0 ± 43.3 | 3.6 ± 6.2   | 6.2 ± 10.8  | 25.0 ± 43.3 | 12 |
| Temas e ODS das questoes                  | 9.9 ± 14.4  | 15.6 ± 21.7 | 7.4 ± 9.3   | 21.9 ± 26.9 | 18 |
| Cadastro da escola pelo INEP              | 15.8 ± 18.7 | 6.2 ± 10.8  | 7.9 ± 13.8  | 61.1 ± 36.0 | 12 |
| Prazo de preenchimento de dados           | 41.7 ± 25.0 | 6.2 ± 10.8  | 9.1 ± 15.7  | 19.4 ± 34.6 | 12 |
| Minimo de membros na presencial           | 8.3 ± 14.4  | 12.5 ± 21.7 | 10.0 ± 17.3 | 25.0 ± 43.3 | 12 |
| Criterios de desempate                    | 19.6 ± 12.3 | 14.6 ± 16.0 | 13.2 ± 13.4 | 49.2 ± 24.6 | 12 |
| Uso de IA generativa                      | 37.5 ± 41.5 | 10.8 ± 12.1 | 16.5 ± 18.1 | 41.7 ± 43.7 | 12 |
| Inscricao exclusivamente online           | 21.1 ± 39.7 | 23.8 ± 34.4 | 18.5 ± 37.1 | 70.6 ± 37.5 | 15 |
| Tipos de escola                           | 16.4 ± 24.5 | 29.2 ± 34.1 | 19.3 ± 29.0 | 50.0 ± 50.0 | 12 |
| Recursos de questao                       | 22.9 ± 27.6 | 19.4 ± 20.2 | 20.2 ± 21.9 | 52.3 ± 30.9 | 12 |
| Contato oficial                           | 69.0 ± 21.3 | 14.8 ± 14.6 | 21.5 ± 20.8 | 65.3 ± 46.5 | 18 |
| Certificados                              | 33.5 ± 24.5 | 17.1 ± 10.7 | 21.6 ± 13.4 | 94.2 ± 10.1 | 15 |
| Classificacao para a presencial           | 30.6 ± 40.5 | 22.2 ± 26.9 | 22.2 ± 31.1 | 88.2 ± 12.5 | 12 |
| Responder todas as questoes               | 35.4 ± 25.3 | 20.0 ± 20.0 | 23.6 ± 23.7 | 68.8 ± 40.6 | 12 |
| Cadastro nao e inscricao                  | 39.6 ± 28.5 | 20.8 ± 12.5 | 26.1 ± 16.2 | 87.2 ± 12.8 | 12 |
| Quem pode participar / series             | 43.3 ± 36.2 | 24.8 ± 19.4 | 29.1 ± 21.8 | 69.0 ± 44.0 | 21 |
| Recuperar senha                           | 31.0 ± 21.8 | 30.0 ± 21.3 | 29.2 ± 20.1 | 67.0 ± 33.7 | 15 |
| Diretrizes de elaboracao de questoes      | 31.4 ± 23.2 | 31.3 ± 24.4 | 29.4 ± 21.6 | 74.3 ± 35.4 | 18 |
| Professor gera senha do estudante         | 39.7 ± 19.8 | 25.0 ± 15.3 | 29.8 ± 15.9 | 74.3 ± 16.6 | 12 |
| Valor da inscricao                        | 30.6 ± 24.4 | 50.0 ± 0.0  | 31.2 ± 22.8 | 100.0 ± 0.0 | 18 |
| Envio de respostas                        | 61.3 ± 29.7 | 25.0 ± 8.3  | 33.9 ± 12.1 | 100.0 ± 0.0 | 12 |
| Selecao para a iGeo                       | 49.7 ± 30.1 | 31.0 ± 15.2 | 35.9 ± 17.0 | 75.8 ± 35.4 | 18 |
| Fases e estrutura das provas              | 75.2 ± 22.2 | 31.7 ± 21.6 | 39.2 ± 22.8 | 100.0 ± 0.0 | 18 |
| Substituicao de membros                   | 56.3 ± 29.0 | 33.3 ± 21.1 | 41.0 ± 22.8 | 69.3 ± 36.9 | 15 |
| Certificados de edicoes anteriores        | 50.0 ± 36.1 | 37.5 ± 21.7 | 41.4 ± 25.2 | 70.0 ± 41.2 | 12 |
| Corrigir dados da escola                  | 73.7 ± 9.9  | 37.8 ± 24.9 | 44.0 ± 20.8 | 87.0 ± 13.0 | 15 |
| Acesso simultaneo durante a prova         | 52.8 ± 34.4 | 40.4 ± 26.7 | 44.8 ± 28.5 | 68.1 ± 40.9 | 12 |
| Nome da equipe                            | 72.7 ± 37.6 | 34.7 ± 20.0 | 46.4 ± 25.5 | 76.7 ± 39.2 | 15 |
| Divulgacao do gabarito                    | 44.0 ± 6.8  | 50.0 ± 0.0  | 46.5 ± 4.1  | 89.3 ± 18.6 | 12 |
| Senha recuperada incompativel / navegador | 57.3 ± 19.3 | 46.7 ± 23.6 | 47.2 ± 24.6 | 86.5 ± 6.6  | 12 |
| Questao anulada                           | 43.1 ± 33.0 | 100.0 ± 0.0 | 54.0 ± 26.7 | 100.0 ± 0.0 | 12 |
| Composicao da equipe                      | 65.3 ± 29.0 | 59.3 ± 23.7 | 54.9 ± 16.4 | 97.2 ± 6.2  | 18 |
| Nome social                               | 94.2 ± 10.2 | 62.5 ± 12.5 | 73.7 ± 7.9  | 100.0 ± 0.0 | 12 |
| Limite de equipes                         | 65.0 ± 8.2  | 90.0 ± 20.0 | 75.1 ± 12.8 | 93.3 ± 13.3 | 15 |
| Equipes de escolas diferentes             | 78.7 ± 12.2 | 80.0 ± 24.5 | 76.3 ± 12.6 | 100.0 ± 0.0 | 15 |
| Medalhas fisicas                          | 75.0 ± 38.7 | 80.0 ± 40.0 | 77.1 ± 39.0 | 73.3 ± 38.9 | 15 |
| Aluno em mais de uma equipe               | 75.0 ± 27.4 | 90.0 ± 20.0 | 81.0 ± 24.7 | 96.7 ± 8.5  | 15 |

Weakest intents: Quem pode ser professor orientador, Token / link expirado, E-mail de confirmacao nao chega, Calendario / datas, Dados obrigatorios do estudante.
Strongest intents: Aluno em mais de uma equipe, Medalhas fisicas, Equipes de escolas diferentes.

## 4. Unsupported facts (hallucination set)

619 of 2525 predicted atomic facts (24.5%) were not supported by the corpus. Examples (intent -> unsupported claim):

- *Quem pode participar / series*: Não encontrei essa informação nos documentos oficiais da OBG.
- *Quem pode participar / series*: Para mais detalhes, contate obgeografia@unifal-mg.edu.br.
- *Cadastro nao e inscricao*: A inscrição é considerada válida somente após a formação das equipes.
- *Professor gera senha do estudante*: O professor deve inserir a nova senha do estudante.
- *Professor gera senha do estudante*: A nova senha deve ser compartilhada com o aluno.
- *Professor gera senha do estudante*: É necessário confirmar a alteração da senha.
- *Recuperar senha*: Para recuperar a senha de acesso ao sistema, o professor ou coordenador da equipe deve seguir o procedimento de geração de senhas.
- *Recuperar senha*: O professor deve gerar uma nova senha para o estudante.
- *Prazo de preenchimento de dados*: O prazo final para a atualização dos dados dos alunos foi até 01/08/2026.
- *Prazo de preenchimento de dados*: Após 01/08/2026, não é possível realizar alterações nas informações dos estudantes.
- *Responder todas as questoes*: Se uma equipe não enviar a documentação necessária em uma fase do regulamento das Olimpíadas Brasileiras de Geografia, ela será automaticamente desclassificada.
- *Criterios de desempate*: Em caso de empate, será realizada uma análise de desempenho em todas as etapas.

## 5. Missing facts (recall gaps)

2277 of 3043 gold atomic facts (74.8%) were missing from the answers. Examples (intent -> missing fact the answer should have contained):

- *Minimo de membros na presencial*: Equipes classificadas para a Fase Presencial nao podem participar com menos de dois membros.
- *Quem pode participar / series*: Podem participar estudantes regularmente matriculados em escolas públicas ou particulares do Brasil.
- *Quem pode participar / series*: Os estudantes devem estar no 9o ano do Ensino Fundamental até o 3o (ou 4o, se houver) ano do Ensino Médio.
- *Quem pode participar / series*: São aceitos ensino regular, profissionalizante, supletivo e EJA.
- *Quem pode participar / series*: Quem já concluiu o Ensino Médio não pode participar.
- *Quem pode participar / series*: Quem está no Ensino Superior não pode participar.
- *Quem pode ser professor orientador*: O professor orientador deve pertencer ao corpo docente da escola.
- *Quem pode ser professor orientador*: O professor orientador pode orientar estagiários.
- *Quem pode ser professor orientador*: O professor orientador pode orientar plantonistas.
- *Quem pode ser professor orientador*: O professor orientador pode orientar coordenadores de olimpíadas.
- *Quem pode ser professor orientador*: Os estagiários, plantonistas e coordenadores de olimpíadas devem estar vinculados à escola.
- *Quem pode ser professor orientador*: Somente o professor orientador pode alterar a composição da equipe.

## 6. How these numbers were produced

Each gold and predicted answer was decomposed into atomic facts by gpt-4o-mini. Recall counts gold facts conveyed in the prediction; precision counts predicted facts inferable from the gold answer; FactScore counts predicted facts supported by the official OBG corpus. F1 is the macro-average of per-question harmonic means. PT<->EN paraphrases are treated as equivalent. Sections 2-5 are generated directly from the per-question results, not written in advance.
