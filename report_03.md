# Lumie RAG chatbot - re-evaluation report

Test units: **200** | Seeds: **[13, 21, 42]** | Judge/decomposer: **gpt-4o-mini** (temp 0.2)
Metrics: atomic Precision/Recall/F1 (vs. the gold answer) and FactScore (grounding vs. the official corpus). All values are percentages, mean ± std across seeds.

## 1. Overall results

| Slice     | Precision  | Recall    | F1        | FactScore  | n   |
|-----------|------------|-----------|-----------|------------|-----|
| Overall   | 1.6 ± 0.0  | 0.6 ± 0.0 | 0.8 ± 0.0 | 1.9 ± 0.0  | 200 |
| lang = en | 0.0 ± 0.0  | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0  | 126 |
| lang = pt | 2.1 ± 11.6 | 0.8 ± 5.1 | 1.0 ± 6.7 | 2.4 ± 13.5 | 474 |

## 2. What the numbers say

Overall the chatbot reaches an F1 of 0.8 (precision 1.6, recall 0.6) and a FactScore of 1.9. Grounding is weak: about 98% of asserted facts are unsupported by the corpus - hallucination is a real problem and is the first thing to fix. Recall is low (0.6): the bot omits a large share of the facts the gold answer contains. In a RAG system this usually points to retrieval gaps - the right chunk was not retrieved or not used. See the missing facts in section 5. Results are stable across seeds (low standard deviation), so the scores are reliable.

## 3. Performance by intent (weakest first)

| Intent                                    | Precision   | Recall      | F1          | FactScore   | n  |
|-------------------------------------------|-------------|-------------|-------------|-------------|----|
| Tipos de escola                           | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 12 |
| Equipes de escolas diferentes             | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 15 |
| Aluno em mais de uma equipe               | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 15 |
| Substituicao de membros                   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 15 |
| Inscricao exclusivamente online           | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 15 |
| Nome social                               | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 12 |
| Quem pode ser professor orientador        | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 15 |
| Composicao da equipe                      | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 18 |
| Cadastro da escola pelo INEP              | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 12 |
| Limite de equipes                         | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 15 |
| Minimo de membros na presencial           | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 12 |
| Dados obrigatorios do estudante           | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 15 |
| Valor da inscricao                        | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 18 |
| Cadastro nao e inscricao                  | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 12 |
| Recuperar senha                           | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 15 |
| Professor gera senha do estudante         | 8.3 ± 14.4  | 0.0 ± 0.0   | 0.0 ± 0.0   | 16.7 ± 28.9 | 12 |
| Senha recuperada incompativel / navegador | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 12 |
| Token / link expirado                     | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 12 |
| E-mail de confirmacao nao chega           | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 15 |
| Navegadores suportados                    | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 12 |
| Acesso simultaneo durante a prova         | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 12 |
| Prazo de preenchimento de dados           | 16.7 ± 28.9 | 0.0 ± 0.0   | 0.0 ± 0.0   | 16.7 ± 28.9 | 12 |
| Envio de respostas                        | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 12 |
| Responder todas as questoes               | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 12 |
| Divulgacao do gabarito                    | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 12 |
| Uso de IA generativa                      | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 12 |
| Classificacao para a presencial           | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 12 |
| Criterios de desempate                    | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 12 |
| Recursos de questao                       | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 12 |
| Questao anulada                           | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 12 |
| Certificados                              | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 15 |
| Medalhas fisicas                          | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 15 |
| Certificados de edicoes anteriores        | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 12 |
| Contato oficial                           | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 18 |
| Selecao para a iGeo                       | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 18 |
| Temas e ODS das questoes                  | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 18 |
| Calendario / datas                        | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 18 |
| Corrigir dados da escola                  | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 15 |
| Nome da equipe                            | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 15 |
| Fases e estrutura das provas              | 3.3 ± 7.5   | 2.4 ± 5.3   | 2.8 ± 6.2   | 0.0 ± 0.0   | 18 |
| Diretrizes de elaboracao de questoes      | 6.7 ± 14.9  | 4.8 ± 11.1  | 5.5 ± 12.5  | 13.3 ± 29.8 | 18 |
| Quem pode participar / series             | 23.8 ± 38.7 | 11.4 ± 18.1 | 15.3 ± 24.3 | 23.8 ± 38.7 | 21 |

Weakest intents: Tipos de escola, Equipes de escolas diferentes, Aluno em mais de uma equipe, Substituicao de membros, Inscricao exclusivamente online.
Strongest intents: Quem pode participar / series, Diretrizes de elaboracao de questoes, Fases e estrutura das provas.

## 4. Unsupported facts (hallucination set)

1194 of 1230 predicted atomic facts (97.1%) were not supported by the corpus. Examples (intent -> unsupported claim):

- *Tipos de escola*: Não encontrei nada no regulamento relacionado à pergunta.
- *Tipos de escola*: A pessoa está pedindo para reformular a pergunta.
- *Prazo de preenchimento de dados*: Mais informações podem ser encontradas na seção de Dúvidas e Suporte sobre Problemas de Acesso à 10ª OBG na página 2.
- *Fases e estrutura das provas*: A 11ª Olimpíada Brasileira de Geografia 2026 possui cinco fases no total.
- *Fases e estrutura das provas*: A 11ª Olimpíada Brasileira de Geografia 2026 tem três fases online em nível estadual.
- *Fases e estrutura das provas*: A 11ª Olimpíada Brasileira de Geografia 2026 tem uma Fase Final Presencial.
- *Fases e estrutura das provas*: A 11ª Olimpíada Brasileira de Geografia 2026 tem uma fase internacional.
- *Fases e estrutura das provas*: O regulamento da 11ª Olimpíada Brasileira de Geografia 2026 pode ser encontrado em Regulamento_11OBG_2026 (1) — pag 1.
- *Selecao para a iGeo*: O documento fornecido não especifica quem pode participar da seleção da Equipe Brasil para a iGeo 2025.
- *Selecao para a iGeo*: Sugiro que entre em contato com a organização do evento pelo email obgeografia@unifal-mg.edu.br para obter essa informação.
- *Selecao para a iGeo*: Você pode encontrar mais informações no Edital Seleção_EquipeBrasil_iGeo 2025 — item 4.3 — pag 3.
- *Quem pode participar / series*: O regulamento da Olimpíada Brasileira de Geografia pode ser encontrado no documento Regulamento_11OBG_2026, item 1.1, página 1.

## 5. Missing facts (recall gaps)

3026 of 3047 gold atomic facts (99.3%) were missing from the answers. Examples (intent -> missing fact the answer should have contained):

- *Tipos de escola*: As escolas são categorizadas em Escolas Particulares.
- *Tipos de escola*: As escolas são categorizadas em Escolas Públicas de Ensino Regular.
- *Tipos de escola*: As escolas são categorizadas em Escolas Públicas com Acesso Seletivo.
- *Tipos de escola*: As Escolas Públicas com Acesso Seletivo incluem escolas técnicas/profissionalizantes.
- *Tipos de escola*: As Escolas Públicas com Acesso Seletivo incluem escolas militares.
- *Tipos de escola*: As Escolas Públicas com Acesso Seletivo incluem colégios de aplicação de universidades públicas.
- *Equipes de escolas diferentes*: Nao sao permitidas equipes com alunos de escolas diferentes.
- *Equipes de escolas diferentes*: As escolas diferentes podem ser da mesma rede de ensino ou mantenedora.
- *Equipes de escolas diferentes*: Os alunos de uma equipe devem pertencer a mesma escola.
- *Aluno em mais de uma equipe*: Um aluno não pode participar de mais de uma equipe.
- *Aluno em mais de uma equipe*: Se um aluno participar de mais de uma equipe, as equipes envolvidas serão sumariamente desclassificadas.
- *Substituicao de membros*: O professor coordenador pode substituir qualquer membro da equipe antes do início da primeira fase online.

## 6. How these numbers were produced

Each gold and predicted answer was decomposed into atomic facts by gpt-4o-mini. Recall counts gold facts conveyed in the prediction; precision counts predicted facts inferable from the gold answer; FactScore counts predicted facts supported by the official OBG corpus. F1 is the macro-average of per-question harmonic means. PT<->EN paraphrases are treated as equivalent. Sections 2-5 are generated directly from the per-question results, not written in advance.
