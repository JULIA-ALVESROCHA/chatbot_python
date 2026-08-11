# Lumie RAG chatbot - re-evaluation report

Test units: **200** | Seeds: **[13, 21, 42]** | Judge/decomposer: **gpt-4o-mini** (temp 0.2)
Metrics: atomic Precision/Recall/F1 (vs. the gold answer) and FactScore (grounding vs. the official corpus). All values are percentages, mean ± std across seeds.

## 1. Overall results

| Slice     | Precision   | Recall      | F1          | FactScore   | n   |
|-----------|-------------|-------------|-------------|-------------|-----|
| Overall   | 35.2 ± 0.3  | 16.9 ± 0.3  | 19.7 ± 0.2  | 44.7 ± 0.5  | 200 |
| lang = en | 11.1 ± 24.8 | 3.8 ± 11.6  | 4.2 ± 12.0  | 24.0 ± 34.8 | 126 |
| lang = pt | 41.6 ± 41.0 | 20.4 ± 27.0 | 23.8 ± 29.4 | 50.2 ± 43.3 | 474 |

## 2. What the numbers say

Overall the chatbot reaches an F1 of 19.7 (precision 35.2, recall 16.9) and a FactScore of 44.7. Grounding is weak: about 55% of asserted facts are unsupported by the corpus - hallucination is a real problem and is the first thing to fix. Recall is low (16.9): the bot omits a large share of the facts the gold answer contains. In a RAG system this usually points to retrieval gaps - the right chunk was not retrieved or not used. See the missing facts in section 5. Results are stable across seeds (low standard deviation), so the scores are reliable.

## 3. Performance by intent (weakest first)

| Intent                                    | Precision   | Recall      | F1          | FactScore   | n  |
|-------------------------------------------|-------------|-------------|-------------|-------------|----|
| Cadastro da escola pelo INEP              | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 15.0 ± 26.0 | 12 |
| Inscricao exclusivamente online           | 0.0 ± 0.0   | 7.6 ± 15.2  | 0.0 ± 0.0   | 53.3 ± 45.2 | 15 |
| Tipos de escola                           | 30.0 ± 41.2 | 0.0 ± 0.0   | 0.0 ± 0.0   | 45.0 ± 45.6 | 12 |
| Token / link expirado                     | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 12 |
| E-mail de confirmacao nao chega           | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 20.0 ± 40.0 | 15 |
| Senha recuperada incompativel / navegador | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 12 |
| Navegadores suportados                    | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 16.7 ± 28.9 | 12 |
| Calendario / datas                        | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 16.7 ± 25.5 | 18 |
| Dados obrigatorios do estudante           | 6.7 ± 13.3  | 1.7 ± 3.3   | 2.7 ± 5.3   | 53.3 ± 45.2 | 15 |
| Quem pode ser professor orientador        | 30.0 ± 29.2 | 3.3 ± 6.7   | 5.5 ± 10.9  | 65.0 ± 37.4 | 15 |
| Temas e ODS das questoes                  | 18.8 ± 30.9 | 5.6 ± 9.0   | 7.5 ± 12.3  | 25.7 ± 38.7 | 18 |
| Minimo de membros na presencial           | 25.0 ± 25.0 | 8.3 ± 18.6  | 8.3 ± 18.6  | 45.8 ± 47.7 | 12 |
| Cadastro nao e inscricao                  | 33.3 ± 20.4 | 6.2 ± 10.8  | 8.3 ± 14.4  | 25.0 ± 25.0 | 12 |
| Responder todas as questoes               | 10.0 ± 17.3 | 10.0 ± 17.3 | 10.0 ± 17.3 | 15.0 ± 26.0 | 12 |
| Professor gera senha do estudante         | 47.5 ± 35.6 | 8.3 ± 9.3   | 11.7 ± 12.3 | 58.8 ± 36.8 | 12 |
| Selecao para a iGeo                       | 33.3 ± 47.1 | 7.9 ± 12.8  | 12.4 ± 19.2 | 38.9 ± 35.6 | 18 |
| Contato oficial                           | 47.2 ± 31.1 | 8.3 ± 8.3   | 12.6 ± 12.8 | 55.6 ± 41.6 | 18 |
| Fases e estrutura das provas              | 38.7 ± 19.6 | 9.5 ± 10.6  | 12.7 ± 13.0 | 56.9 ± 27.1 | 18 |
| Recuperar senha                           | 20.0 ± 40.0 | 10.0 ± 20.0 | 13.3 ± 26.7 | 13.3 ± 26.7 | 15 |
| Valor da inscricao                        | 13.9 ± 22.4 | 16.7 ± 23.6 | 13.9 ± 22.4 | 16.7 ± 23.6 | 18 |
| Recursos de questao                       | 16.7 ± 28.9 | 12.5 ± 21.7 | 14.3 ± 24.7 | 42.1 ± 32.0 | 12 |
| Corrigir dados da escola                  | 53.8 ± 39.3 | 8.9 ± 8.3   | 14.4 ± 12.7 | 43.0 ± 26.4 | 15 |
| Diretrizes de elaboracao de questoes      | 35.7 ± 38.5 | 11.1 ± 15.0 | 15.4 ± 17.3 | 50.0 ± 50.0 | 18 |
| Uso de IA generativa                      | 41.7 ± 43.3 | 10.0 ± 10.0 | 16.0 ± 16.1 | 41.7 ± 43.3 | 12 |
| Composicao da equipe                      | 19.4 ± 27.9 | 22.2 ± 24.8 | 17.8 ± 26.3 | 50.0 ± 50.0 | 18 |
| Substituicao de membros                   | 50.0 ± 44.7 | 13.3 ± 16.3 | 20.0 ± 24.5 | 60.0 ± 49.0 | 15 |
| Certificados                              | 43.3 ± 40.3 | 14.3 ± 15.6 | 21.1 ± 22.0 | 60.0 ± 49.0 | 15 |
| Criterios de desempate                    | 52.1 ± 34.0 | 16.7 ± 18.6 | 22.3 ± 23.2 | 77.8 ± 28.3 | 12 |
| Classificacao para a presencial           | 31.2 ± 41.0 | 22.2 ± 26.9 | 24.9 ± 31.0 | 64.6 ± 25.3 | 12 |
| Envio de respostas                        | 65.3 ± 41.1 | 16.7 ± 11.8 | 26.2 ± 17.7 | 75.0 ± 43.3 | 12 |
| Quem pode participar / series             | 61.0 ± 21.1 | 21.9 ± 12.2 | 28.6 ± 13.7 | 76.4 ± 22.2 | 21 |
| Acesso simultaneo durante a prova         | 75.0 ± 43.3 | 23.3 ± 23.6 | 31.7 ± 31.9 | 33.3 ± 40.8 | 12 |
| Prazo de preenchimento de dados           | 75.0 ± 43.3 | 25.0 ± 17.7 | 36.7 ± 23.8 | 75.0 ± 43.3 | 12 |
| Nome da equipe                            | 66.7 ± 42.2 | 28.0 ± 24.0 | 37.9 ± 31.6 | 0.0 ± 0.0   | 15 |
| Questao anulada                           | 33.3 ± 33.3 | 50.0 ± 50.0 | 40.0 ± 40.0 | 41.7 ± 43.3 | 12 |
| Certificados de edicoes anteriores        | 50.0 ± 37.3 | 37.5 ± 21.7 | 41.0 ± 25.5 | 59.7 ± 39.9 | 12 |
| Divulgacao do gabarito                    | 50.0 ± 50.0 | 37.5 ± 41.5 | 41.7 ± 43.3 | 37.5 ± 41.5 | 12 |
| Equipes de escolas diferentes             | 60.0 ± 49.0 | 40.0 ± 35.9 | 47.1 ± 39.7 | 60.0 ± 49.0 | 15 |
| Aluno em mais de uma equipe               | 46.7 ± 45.2 | 50.0 ± 44.7 | 48.0 ± 44.9 | 53.3 ± 45.2 | 15 |
| Nome social                               | 66.7 ± 42.5 | 43.8 ± 27.2 | 52.0 ± 32.1 | 66.7 ± 42.5 | 12 |
| Limite de equipes                         | 70.0 ± 24.5 | 50.0 ± 0.0  | 56.7 ± 8.2  | 80.0 ± 24.5 | 15 |
| Medalhas fisicas                          | 65.0 ± 37.4 | 57.8 ± 37.4 | 60.8 ± 37.2 | 78.3 ± 23.9 | 15 |

Weakest intents: Cadastro da escola pelo INEP, Inscricao exclusivamente online, Tipos de escola, Token / link expirado, E-mail de confirmacao nao chega.
Strongest intents: Medalhas fisicas, Limite de equipes, Nome social.

## 4. Unsupported facts (hallucination set)

996 of 1870 predicted atomic facts (53.3%) were not supported by the corpus. Examples (intent -> unsupported claim):

- *Nome social*: O participante deve optar por registrá-lo durante o cadastro ou atualização de dados.
- *Limite de equipes*: Recomenda-se que o orientador inscreva apenas a quantidade de equipes que consiga acompanhar adequadamente.
- *Quem pode participar / series*: Estudantes regularmente matriculados no 8º ano do ensino fundamental II podem participar.
- *Quem pode participar / series*: Estudantes regularmente matriculados em qualquer série do ensino médio podem participar.
- *Cadastro da escola pelo INEP*: Não encontrei essa informação nos documentos oficiais da OBG.
- *Cadastro da escola pelo INEP*: Você pode reformular a pergunta.
- *Cadastro da escola pelo INEP*: Você pode contatar obgeografia@unifal-mg.edu.br.
- *Valor da inscricao*: Não há informações sobre o valor da inscrição para escolas particulares nos documentos disponíveis.
- *Tipos de escola*: Estudantes do 4º ano do ensino médio podem participar.
- *Recuperar senha*: A nova senha será enviada automaticamente para o e-mail cadastrado.
- *Professor gera senha do estudante*: O procedimento garante acesso seguro à plataforma.
- *Navegadores suportados*: O contexto não menciona um navegador específico recomendado para realizar a prova de Geografia das Olimpíadas Brasileiras.

## 5. Missing facts (recall gaps)

2668 of 3047 gold atomic facts (87.6%) were missing from the answers. Examples (intent -> missing fact the answer should have contained):

- *Composicao da equipe*: Cada equipe é formada por 1 professor(a) orientador(a).
- *Composicao da equipe*: O professor(a) orientador(a) é o responsável pela inscrição.
- *Quem pode ser professor orientador*: O professor orientador deve pertencer ao corpo docente da escola.
- *Quem pode ser professor orientador*: O professor orientador pode orientar estagiários.
- *Quem pode ser professor orientador*: O professor orientador pode orientar plantonistas.
- *Quem pode ser professor orientador*: O professor orientador pode orientar coordenadores de olimpíadas.
- *Quem pode ser professor orientador*: Os estagiários, plantonistas e coordenadores de olimpíadas devem estar vinculados à escola.
- *Nome social*: É garantido o uso do nome social durante toda a prova.
- *Nome social*: O uso do nome social é em cumprimento ao Decreto no 8.727/2016.
- *Limite de equipes*: Nao ha limite de equipes que uma escola pode inscrever.
- *Equipes de escolas diferentes*: As escolas diferentes podem ser da mesma rede de ensino ou mantenedora.
- *Minimo de membros na presencial*: Equipes classificadas para a Fase Presencial nao podem participar com menos de dois membros.

## 6. How these numbers were produced

Each gold and predicted answer was decomposed into atomic facts by gpt-4o-mini. Recall counts gold facts conveyed in the prediction; precision counts predicted facts inferable from the gold answer; FactScore counts predicted facts supported by the official OBG corpus. F1 is the macro-average of per-question harmonic means. PT<->EN paraphrases are treated as equivalent. Sections 2-5 are generated directly from the per-question results, not written in advance.
