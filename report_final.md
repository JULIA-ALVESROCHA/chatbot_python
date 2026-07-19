# Lumie RAG chatbot - re-evaluation report

Test units: **200** | Seeds: **[13, 21, 42]** | Judge/decomposer: **gpt-4o-mini** (temp 0.2)
Metrics: atomic Precision/Recall/F1 (vs. the gold answer) and FactScore (grounding vs. the official corpus). All values are percentages, mean ± std across seeds.

## 1. Overall results

| Slice     | Precision   | Recall      | F1          | FactScore   | n   |
|-----------|-------------|-------------|-------------|-------------|-----|
| Overall   | 58.0 ± 0.5  | 22.4 ± 0.2  | 33.8 ± 0.1  | 75.7 ± 0.4  | 200 |
| lang = en | 43.3 ± 41.9 | 16.8 ± 27.6 | 26.1 ± 31.7 | 68.9 ± 34.3 | 126 |
| lang = pt | 61.5 ± 36.0 | 23.9 ± 26.7 | 35.6 ± 29.5 | 77.3 ± 28.7 | 474 |

## 2. What the numbers say

Overall the chatbot reaches an F1 of 33.8 (precision 58.0, recall 22.4) and a FactScore of 75.7. Grounding is weak: about 24% of asserted facts are unsupported by the corpus - hallucination is a real problem and is the first thing to fix. FactScore (75.7) is clearly higher than precision-vs-gold (58.0). That gap means the bot adds true, corpus-grounded details that are not in the short canonical answer - i.e. it is verbose rather than wrong. This is a style issue, not a factual one. Recall is low (22.4): the bot omits a large share of the facts the gold answer contains. In a RAG system this usually points to retrieval gaps - the right chunk was not retrieved or not used. See the missing facts in section 5. Results are stable across seeds (low standard deviation), so the scores are reliable.

## 3. Performance by intent (weakest first)

| Intent                                    | Precision   | Recall      | F1          | FactScore   | n  |
|-------------------------------------------|-------------|-------------|-------------|-------------|----|
| Quem pode ser professor orientador        | 8.3 ± 14.4  | 0.0 ± 0.0   | 0.0 ± 0.0   | 33.3 ± 16.7 | 15 |
| Inscricao exclusivamente online           | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 100.0 ± 0.0 | 15 |
| Token / link expirado                     | 33.3 ± 0.0  | 0.0 ± 0.0   | 0.0 ± 0.0   | 44.4 ± 15.7 | 12 |
| Calendario / datas                        | 22.2 ± 24.8 | 0.0 ± 0.0   | 0.0 ± 0.0   | 16.7 ± 16.7 | 18 |
| Dados obrigatorios do estudante           | 11.1 ± 15.7 | 1.7 ± 3.3   | 4.4 ± 6.3   | 88.9 ± 15.7 | 15 |
| E-mail de confirmacao nao chega           | 54.2 ± 9.3  | 3.3 ± 6.7   | 12.9 ± 12.9 | 87.5 ± 12.5 | 15 |
| Responder todas as questoes               | 33.3 ± 23.6 | 13.3 ± 14.9 | 13.1 ± 19.0 | 77.8 ± 31.4 | 12 |
| Minimo de membros na presencial           | 61.1 ± 28.3 | 12.5 ± 21.7 | 13.3 ± 18.9 | 77.8 ± 31.4 | 12 |
| Cadastro da escola pelo INEP              | 33.3 ± 27.2 | 12.5 ± 21.7 | 13.3 ± 18.9 | 83.3 ± 13.6 | 12 |
| Cadastro nao e inscricao                  | 22.2 ± 24.8 | 12.5 ± 12.5 | 14.8 ± 16.6 | 70.4 ± 21.9 | 12 |
| Composicao da equipe                      | 40.0 ± 37.4 | 18.5 ± 16.6 | 16.0 ± 19.6 | 87.8 ± 20.6 | 18 |
| Corrigir dados da escola                  | 58.3 ± 42.5 | 6.7 ± 8.9   | 18.1 ± 14.1 | 55.6 ± 7.9  | 15 |
| Classificacao para a presencial           | 25.0 ± 43.3 | 15.0 ± 27.8 | 18.2 ± 32.7 | 56.2 ± 29.7 | 12 |
| Selecao para a iGeo                       | 55.0 ± 43.7 | 12.7 ± 11.6 | 20.0 ± 16.3 | 61.0 ± 29.1 | 18 |
| Uso de IA generativa                      | 55.6 ± 41.6 | 10.0 ± 10.0 | 21.4 ± 15.1 | 50.0 ± 38.5 | 12 |
| Fases e estrutura das provas              | 68.3 ± 24.6 | 16.7 ± 15.2 | 21.6 ± 18.1 | 85.6 ± 15.1 | 18 |
| Quem pode participar / series             | 57.6 ± 30.1 | 15.2 ± 8.5  | 22.0 ± 12.9 | 79.9 ± 25.5 | 21 |
| Professor gera senha do estudante         | 45.0 ± 26.8 | 15.6 ± 13.6 | 22.2 ± 17.1 | 71.2 ± 27.9 | 12 |
| Temas e ODS das questoes                  | 49.9 ± 35.5 | 15.6 ± 14.2 | 22.6 ± 13.7 | 75.5 ± 25.0 | 18 |
| Criterios de desempate                    | 33.3 ± 20.4 | 18.8 ± 10.8 | 23.8 ± 13.9 | 66.7 ± 20.4 | 12 |
| Certificados                              | 50.0 ± 35.0 | 17.1 ± 16.7 | 23.9 ± 22.8 | 56.7 ± 35.5 | 15 |
| Substituicao de membros                   | 79.2 ± 21.7 | 13.3 ± 16.3 | 25.0 ± 25.0 | 91.7 ± 14.4 | 15 |
| Recursos de questao                       | 51.1 ± 22.0 | 16.7 ± 20.4 | 27.9 ± 23.3 | 87.8 ± 8.7  | 12 |
| Diretrizes de elaboracao de questoes      | 68.1 ± 27.2 | 18.9 ± 15.7 | 32.1 ± 16.4 | 90.8 ± 13.0 | 18 |
| Contato oficial                           | 56.7 ± 38.9 | 19.4 ± 15.0 | 32.4 ± 19.1 | 72.8 ± 38.2 | 18 |
| Senha recuperada incompativel / navegador | 37.5 ± 37.5 | 16.7 ± 25.6 | 33.3 ± 33.3 | 75.0 ± 25.0 | 12 |
| Questao anulada                           | 33.3 ± 47.1 | 25.0 ± 43.3 | 33.3 ± 47.1 | 38.9 ± 31.4 | 12 |
| Nome da equipe                            | 87.5 ± 21.7 | 20.0 ± 17.9 | 36.9 ± 23.4 | 20.8 ± 24.7 | 15 |
| Tipos de escola                           | 55.6 ± 42.1 | 20.8 ± 21.7 | 36.9 ± 27.7 | 82.9 ± 12.7 | 12 |
| Valor da inscricao                        | 50.0 ± 35.4 | 27.8 ± 29.9 | 44.4 ± 29.1 | 62.5 ± 41.5 | 18 |
| Prazo de preenchimento de dados           | 87.5 ± 21.7 | 31.2 ± 10.8 | 45.0 ± 12.8 | 100.0 ± 0.0 | 12 |
| Recuperar senha                           | 75.0 ± 20.4 | 20.0 ± 19.4 | 45.9 ± 17.0 | 88.9 ± 15.7 | 15 |
| Acesso simultaneo durante a prova         | 83.3 ± 23.6 | 27.1 ± 28.2 | 46.4 ± 33.6 | 100.0 ± 0.0 | 12 |
| Envio de respostas                        | 100.0 ± 0.0 | 33.3 ± 11.8 | 48.8 ± 13.5 | 97.2 ± 9.2  | 12 |
| Certificados de edicoes anteriores        | 55.6 ± 20.8 | 37.5 ± 21.7 | 50.9 ± 9.0  | 79.6 ± 23.3 | 12 |
| Limite de equipes                         | 60.0 ± 37.4 | 50.0 ± 31.6 | 53.3 ± 32.3 | 80.0 ± 24.5 | 15 |
| Navegadores suportados                    | 79.2 ± 22.4 | 26.2 ± 27.9 | 58.5 ± 4.5  | 87.5 ± 12.5 | 12 |
| Divulgacao do gabarito                    | 83.3 ± 23.6 | 37.5 ± 21.7 | 61.1 ± 7.9  | 61.1 ± 20.8 | 12 |
| Equipes de escolas diferentes             | 85.0 ± 30.0 | 62.2 ± 16.6 | 68.2 ± 20.1 | 90.0 ± 20.0 | 15 |
| Medalhas fisicas                          | 86.7 ± 26.7 | 73.3 ± 24.9 | 78.7 ± 24.4 | 93.3 ± 13.3 | 15 |
| Nome social                               | 95.8 ± 13.8 | 68.8 ± 10.8 | 78.8 ± 9.9  | 97.2 ± 9.2  | 12 |
| Aluno em mais de uma equipe               | 95.6 ± 11.3 | 90.0 ± 20.0 | 92.1 ± 16.0 | 100.0 ± 0.0 | 15 |

Weakest intents: Quem pode ser professor orientador, Inscricao exclusivamente online, Token / link expirado, Calendario / datas, Dados obrigatorios do estudante.
Strongest intents: Aluno em mais de uma equipe, Nome social, Medalhas fisicas.

## 4. Unsupported facts (hallucination set)

347 of 1478 predicted atomic facts (23.5%) were not supported by the corpus. Examples (intent -> unsupported claim):

- *Valor da inscricao*: Não há informações sobre o valor da inscrição para escolas particulares nos documentos disponíveis.
- *Limite de equipes*: O orientador deve acompanhar adequadamente as equipes que inscrever.
- *Cadastro nao e inscricao*: A equipe não está oficialmente inscrita na Olimpíada Brasileira de Geografia.
- *Quem pode participar / series*: A data limite para participação é 30 de junho de 2027.
- *Quem pode participar / series*: Os estudantes devem ter entre 16 e 19 anos em 30 de junho de 2027.
- *Professor gera senha do estudante*: É importante que a senha seja compartilhada com o aluno após a alteração.
- *Divulgacao do gabarito*: A data exata de divulgação do gabarito oficial não está especificada nos documentos.
- *Cadastro da escola pelo INEP*: Os dados a serem verificados incluem o nome da escola, cidade e estado.
- *Uso de IA generativa*: As discussões sobre as provas devem ser restritas apenas entre os membros da própria equipe.
- *Certificados*: Os certificados devem ser impressos até o dia 31/12/2026.
- *Certificados*: Não há garantia de que os certificados permanecerão disponíveis online após o dia 31/12/2026.
- *Questao anulada*: Os regulamentos não mencionam o impacto específico na pontuação de uma questão anulada nas olimpíadas brasileiras de Geografia.

## 5. Missing facts (recall gaps)

2507 of 3047 gold atomic facts (82.3%) were missing from the answers. Examples (intent -> missing fact the answer should have contained):

- *Equipes de escolas diferentes*: As escolas diferentes podem ser da mesma rede de ensino ou mantenedora.
- *Quem pode ser professor orientador*: O professor orientador deve pertencer ao corpo docente da escola.
- *Quem pode ser professor orientador*: O professor orientador pode orientar estagiários.
- *Quem pode ser professor orientador*: O professor orientador pode orientar plantonistas.
- *Quem pode ser professor orientador*: O professor orientador pode orientar coordenadores de olimpíadas.
- *Quem pode ser professor orientador*: Os estagiários, plantonistas e coordenadores de olimpíadas devem estar vinculados à escola.
- *Quem pode ser professor orientador*: Somente o professor orientador pode alterar a composição da equipe.
- *Substituicao de membros*: O professor coordenador pode substituir qualquer membro da equipe antes do início da primeira fase online.
- *Substituicao de membros*: Depois do início da primeira fase online, o sistema permite apenas a exclusão do estudante.
- *Substituicao de membros*: Equipes classificadas para a Presencial nacional não podem substituir membros.
- *Nome social*: É garantido o uso do nome social durante toda a prova.
- *Nome social*: O uso do nome social é em cumprimento ao Decreto no 8.727/2016.

## 6. How these numbers were produced

Each gold and predicted answer was decomposed into atomic facts by gpt-4o-mini. Recall counts gold facts conveyed in the prediction; precision counts predicted facts inferable from the gold answer; FactScore counts predicted facts supported by the official OBG corpus. F1 is the macro-average of per-question harmonic means. PT<->EN paraphrases are treated as equivalent. Sections 2-5 are generated directly from the per-question results, not written in advance.
