# Lumie RAG chatbot - re-evaluation report

Test units: **200** | Seeds: **[13, 21, 42]** | Judge/decomposer: **gpt-4o-mini** (temp 0.2)
Metrics: atomic Precision/Recall/F1 (vs. the gold answer) and FactScore (grounding vs. the official corpus). All values are percentages, mean ± std across seeds.

## 1. Overall results

| Slice     | Precision   | Recall      | F1          | FactScore   | n   |
|-----------|-------------|-------------|-------------|-------------|-----|
| Overall   | 41.3 ± 0.3  | 32.2 ± 0.2  | 31.5 ± 0.3  | 53.7 ± 0.1  | 200 |
| lang = en | 34.2 ± 32.7 | 30.6 ± 30.5 | 27.4 ± 27.3 | 53.9 ± 37.7 | 126 |
| lang = pt | 43.2 ± 36.4 | 32.6 ± 32.6 | 32.6 ± 30.9 | 53.7 ± 39.6 | 474 |

## 2. What the numbers say

Overall the chatbot reaches an F1 of 31.5 (precision 41.3, recall 32.2) and a FactScore of 53.7. Grounding is weak: about 46% of asserted facts are unsupported by the corpus - hallucination is a real problem and is the first thing to fix. FactScore (53.7) is clearly higher than precision-vs-gold (41.3). That gap means the bot adds true, corpus-grounded details that are not in the short canonical answer - i.e. it is verbose rather than wrong. This is a style issue, not a factual one. Recall is low (32.2): the bot omits a large share of the facts the gold answer contains. In a RAG system this usually points to retrieval gaps - the right chunk was not retrieved or not used. See the missing facts in section 5. Results are stable across seeds (low standard deviation), so the scores are reliable.

## 3. Performance by intent (weakest first)

| Intent                                    | Precision   | Recall      | F1          | FactScore   | n  |
|-------------------------------------------|-------------|-------------|-------------|-------------|----|
| Quem pode ser professor orientador        | 15.0 ± 19.3 | 0.0 ± 0.0   | 0.0 ± 0.0   | 45.8 ± 24.7 | 15 |
| Token / link expirado                     | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 12 |
| E-mail de confirmacao nao chega           | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 20.0 ± 40.0 | 15 |
| Calendario / datas                        | 3.3 ± 7.5   | 1.2 ± 3.5   | 0.0 ± 0.0   | 6.7 ± 9.4   | 18 |
| Dados obrigatorios do estudante           | 5.0 ± 10.0  | 1.7 ± 3.4   | 2.6 ± 5.1   | 48.8 ± 40.0 | 15 |
| Navegadores suportados                    | 25.0 ± 43.3 | 3.6 ± 6.2   | 6.2 ± 10.8  | 37.5 ± 41.5 | 12 |
| Temas e ODS das questoes                  | 9.9 ± 14.4  | 15.6 ± 21.7 | 7.4 ± 9.3   | 17.2 ± 24.4 | 18 |
| Cadastro da escola pelo INEP              | 17.5 ± 20.5 | 6.2 ± 10.8  | 8.3 ± 14.4  | 48.3 ± 28.4 | 12 |
| Prazo de preenchimento de dados           | 37.5 ± 27.3 | 6.2 ± 10.8  | 9.1 ± 15.7  | 25.0 ± 43.3 | 12 |
| Minimo de membros na presencial           | 8.3 ± 14.4  | 12.5 ± 21.7 | 10.0 ± 17.3 | 25.0 ± 43.3 | 12 |
| Criterios de desempate                    | 19.6 ± 12.3 | 16.7 ± 18.6 | 13.8 ± 13.9 | 55.4 ± 20.5 | 12 |
| Uso de IA generativa                      | 37.5 ± 41.5 | 12.5 ± 15.3 | 18.6 ± 22.3 | 43.8 ± 44.6 | 12 |
| Recursos de questao                       | 22.9 ± 27.6 | 18.1 ± 18.6 | 19.2 ± 20.5 | 39.2 ± 27.1 | 12 |
| Inscricao exclusivamente online           | 20.0 ± 40.0 | 24.7 ± 36.8 | 19.3 ± 38.6 | 46.1 ± 28.4 | 15 |
| Tipos de escola                           | 16.4 ± 24.5 | 33.3 ± 36.0 | 19.3 ± 29.0 | 46.4 ± 46.7 | 12 |
| Certificados                              | 33.5 ± 24.5 | 17.1 ± 10.7 | 21.6 ± 13.4 | 44.2 ± 30.2 | 15 |
| Classificacao para a presencial           | 32.2 ± 39.6 | 22.2 ± 24.8 | 21.9 ± 29.5 | 41.2 ± 36.4 | 12 |
| Responder todas as questoes               | 32.6 ± 23.4 | 20.0 ± 20.0 | 22.5 ± 22.7 | 55.6 ± 38.7 | 12 |
| Contato oficial                           | 70.4 ± 22.4 | 15.7 ± 14.1 | 23.5 ± 20.7 | 63.9 ± 45.8 | 18 |
| Cadastro nao e inscricao                  | 39.6 ± 28.5 | 20.8 ± 12.5 | 26.1 ± 16.2 | 61.7 ± 13.8 | 12 |
| Recuperar senha                           | 32.7 ± 25.5 | 27.8 ± 18.9 | 28.7 ± 20.3 | 66.2 ± 35.6 | 15 |
| Quem pode participar / series             | 43.1 ± 36.0 | 24.8 ± 18.4 | 28.9 ± 20.6 | 64.3 ± 42.2 | 21 |
| Professor gera senha do estudante         | 38.2 ± 20.7 | 25.0 ± 15.3 | 29.4 ± 16.2 | 68.0 ± 10.4 | 12 |
| Diretrizes de elaboracao de questoes      | 30.2 ± 22.1 | 33.3 ± 23.6 | 29.8 ± 20.3 | 68.2 ± 33.7 | 18 |
| Valor da inscricao                        | 30.6 ± 24.4 | 50.0 ± 0.0  | 31.2 ± 22.8 | 50.0 ± 23.6 | 18 |
| Envio de respostas                        | 61.3 ± 29.7 | 25.0 ± 8.3  | 33.9 ± 12.1 | 88.8 ± 11.4 | 12 |
| Selecao para a iGeo                       | 49.7 ± 30.1 | 31.0 ± 15.2 | 35.9 ± 17.0 | 30.8 ± 29.0 | 18 |
| Fases e estrutura das provas              | 73.6 ± 23.6 | 32.5 ± 22.7 | 39.5 ± 23.3 | 96.7 ± 7.5  | 18 |
| Substituicao de membros                   | 56.3 ± 29.0 | 33.3 ± 21.1 | 41.0 ± 22.8 | 51.0 ± 30.5 | 15 |
| Certificados de edicoes anteriores        | 50.0 ± 36.1 | 37.5 ± 21.7 | 41.4 ± 25.2 | 55.0 ± 35.7 | 12 |
| Acesso simultaneo durante a prova         | 54.9 ± 36.4 | 37.5 ± 26.0 | 43.8 ± 29.1 | 63.2 ± 38.6 | 12 |
| Corrigir dados da escola                  | 75.7 ± 7.8  | 37.8 ± 24.9 | 45.2 ± 22.1 | 36.4 ± 19.6 | 15 |
| Divulgacao do gabarito                    | 44.0 ± 6.8  | 50.0 ± 0.0  | 46.5 ± 4.1  | 53.0 ± 19.6 | 12 |
| Senha recuperada incompativel / navegador | 56.5 ± 19.1 | 46.7 ± 23.6 | 46.8 ± 24.3 | 91.9 ± 8.5  | 12 |
| Nome da equipe                            | 72.7 ± 37.6 | 36.0 ± 19.6 | 47.9 ± 25.2 | 0.0 ± 0.0   | 15 |
| Questao anulada                           | 43.1 ± 33.0 | 100.0 ± 0.0 | 54.0 ± 26.7 | 100.0 ± 0.0 | 12 |
| Composicao da equipe                      | 65.3 ± 29.0 | 61.1 ± 22.9 | 56.1 ± 16.4 | 91.7 ± 18.6 | 18 |
| Nome social                               | 96.2 ± 8.4  | 62.5 ± 12.5 | 74.6 ± 8.6  | 96.2 ± 8.4  | 12 |
| Limite de equipes                         | 65.0 ± 8.2  | 90.0 ± 20.0 | 75.1 ± 12.8 | 65.0 ± 22.6 | 15 |
| Equipes de escolas diferentes             | 78.7 ± 12.2 | 80.0 ± 24.5 | 76.3 ± 12.6 | 78.7 ± 12.2 | 15 |
| Medalhas fisicas                          | 75.0 ± 38.7 | 80.0 ± 40.0 | 77.1 ± 39.0 | 80.0 ± 40.0 | 15 |
| Aluno em mais de uma equipe               | 75.0 ± 27.4 | 90.0 ± 20.0 | 81.0 ± 24.7 | 90.0 ± 20.0 | 15 |

Weakest intents: Quem pode ser professor orientador, Token / link expirado, E-mail de confirmacao nao chega, Calendario / datas, Dados obrigatorios do estudante.
Strongest intents: Aluno em mais de uma equipe, Medalhas fisicas, Equipes de escolas diferentes.

## 4. Unsupported facts (hallucination set)

1078 of 2526 predicted atomic facts (42.7%) were not supported by the corpus. Examples (intent -> unsupported claim):

- *Limite de equipes*: Recomenda-se que o orientador inscreva apenas a quantidade de equipes que consiga acompanhar e gerenciar adequadamente ao longo das fases.
- *Substituicao de membros*: Não encontrei essa informação nos documentos oficiais da OBG.
- *Substituicao de membros*: Para mais detalhes, contate obgeografia@unifal-mg.edu.br.
- *Equipes de escolas diferentes*: Os estudantes podem ser de salas ou séries diferentes dentro da mesma escola.
- *Valor da inscricao*: O valor da inscrição para a Olimpíada Brasileira de Geografia é R$ 65,00 por equipe para escolas particulares.
- *Cadastro nao e inscricao*: Após o cadastro, o professor responsável deve formar as equipes.
- *Cadastro nao e inscricao*: A inscrição é considerada válida somente após a formação das equipes.
- *Inscricao exclusivamente online*: As inscrições para a Olimpíada Brasileira de Geografia foram encerradas em 16/06/2026.
- *Inscricao exclusivamente online*: Escolas particulares pagam uma taxa de R$ 65,00 por equipe para participar da Olimpíada Brasileira de Geografia.
- *Inscricao exclusivamente online*: A Olimpíada Brasileira de Geografia pode encerrar as inscrições antecipadamente se atingir o limite de participantes.
- *Inscricao exclusivamente online*: A Olimpíada Brasileira de Geografia respeita a ordem de inscrição ao encerrar as inscrições.
- *Professor gera senha do estudante*: O professor deve inserir a nova senha do estudante.

## 5. Missing facts (recall gaps)

2263 of 3036 gold atomic facts (74.5%) were missing from the answers. Examples (intent -> missing fact the answer should have contained):

- *Limite de equipes*: Nao ha limite de equipes que uma escola pode inscrever.
- *Minimo de membros na presencial*: Equipes classificadas para a Fase Presencial nao podem participar com menos de dois membros.
- *Substituicao de membros*: O professor coordenador pode substituir qualquer membro da equipe antes do início da primeira fase online.
- *Substituicao de membros*: Depois do início da primeira fase online, o sistema permite apenas a exclusão do estudante.
- *Substituicao de membros*: Equipes classificadas para a Presencial nacional não podem substituir membros.
- *Quem pode participar / series*: Podem participar estudantes regularmente matriculados em escolas públicas ou particulares do Brasil.
- *Quem pode participar / series*: Os estudantes devem estar no 9o ano do Ensino Fundamental até o 3o (ou 4o, se houver) ano do Ensino Médio.
- *Quem pode participar / series*: São aceitos ensino regular, profissionalizante, supletivo e EJA.
- *Quem pode participar / series*: Quem já concluiu o Ensino Médio não pode participar.
- *Quem pode participar / series*: Quem está no Ensino Superior não pode participar.
- *Quem pode ser professor orientador*: O professor orientador deve pertencer ao corpo docente da escola.
- *Quem pode ser professor orientador*: O professor orientador pode orientar estagiários.

## 6. How these numbers were produced

Each gold and predicted answer was decomposed into atomic facts by gpt-4o-mini. Recall counts gold facts conveyed in the prediction; precision counts predicted facts inferable from the gold answer; FactScore counts predicted facts supported by the official OBG corpus. F1 is the macro-average of per-question harmonic means. PT<->EN paraphrases are treated as equivalent. Sections 2-5 are generated directly from the per-question results, not written in advance.
