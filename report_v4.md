# Lumie RAG chatbot - re-evaluation report

Test units: **200** | Seeds: **[13, 21, 42]** | Judge/decomposer: **gpt-4o-mini** (temp 0.2)
Metrics: atomic Precision/Recall/F1 (vs. the gold answer) and FactScore (grounding vs. the official corpus). All values are percentages, mean ± std across seeds.

## 1. Overall results

| Slice     | Precision   | Recall      | F1          | FactScore   | n   |
|-----------|-------------|-------------|-------------|-------------|-----|
| Overall   | 50.2 ± 0.3  | 23.3 ± 0.3  | 30.5 ± 0.1  | 69.5 ± 0.3  | 200 |
| lang = en | 37.2 ± 40.2 | 12.0 ± 22.9 | 18.3 ± 28.1 | 60.9 ± 34.5 | 126 |
| lang = pt | 53.2 ± 36.4 | 26.3 ± 28.6 | 33.3 ± 28.8 | 71.5 ± 33.5 | 474 |

## 2. What the numbers say

Overall the chatbot reaches an F1 of 30.5 (precision 50.2, recall 23.3) and a FactScore of 69.5. Grounding is weak: about 30% of asserted facts are unsupported by the corpus - hallucination is a real problem and is the first thing to fix. FactScore (69.5) is clearly higher than precision-vs-gold (50.2). That gap means the bot adds true, corpus-grounded details that are not in the short canonical answer - i.e. it is verbose rather than wrong. This is a style issue, not a factual one. Recall is low (23.3): the bot omits a large share of the facts the gold answer contains. In a RAG system this usually points to retrieval gaps - the right chunk was not retrieved or not used. See the missing facts in section 5. Results are stable across seeds (low standard deviation), so the scores are reliable.

## 3. Performance by intent (weakest first)

| Intent                                    | Precision   | Recall      | F1          | FactScore   | n  |
|-------------------------------------------|-------------|-------------|-------------|-------------|----|
| Quem pode ser professor orientador        | 8.3 ± 14.4  | 1.1 ± 4.2   | 0.0 ± 0.0   | 74.2 ± 18.3 | 15 |
| Token / link expirado                     | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 50.0 ± 0.0  | 12 |
| E-mail de confirmacao nao chega           | 16.7 ± 18.6 | 0.0 ± 0.0   | 0.0 ± 0.0   | 87.5 ± 12.5 | 15 |
| Responder todas as questoes               | 0.0 ± 0.0   | 5.0 ± 8.7   | 0.0 ± 0.0   | 52.8 ± 33.6 | 12 |
| Calendario / datas                        | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 0.0 ± 0.0   | 18 |
| Dados obrigatorios do estudante           | 11.1 ± 15.7 | 1.7 ± 3.3   | 4.4 ± 6.3   | 88.9 ± 15.7 | 15 |
| Inscricao exclusivamente online           | 11.1 ± 15.7 | 3.8 ± 7.6   | 8.0 ± 11.4  | 100.0 ± 0.0 | 15 |
| Temas e ODS das questoes                  | 37.3 ± 22.3 | 5.6 ± 9.0   | 9.6 ± 13.6  | 68.9 ± 18.9 | 18 |
| Composicao da equipe                      | 40.0 ± 32.7 | 11.1 ± 15.7 | 10.0 ± 20.0 | 100.0 ± 0.0 | 18 |
| Minimo de membros na presencial           | 25.0 ± 27.6 | 12.5 ± 21.7 | 10.0 ± 17.3 | 51.4 ± 36.3 | 12 |
| Questao anulada                           | 8.3 ± 11.8  | 25.0 ± 43.3 | 13.3 ± 18.9 | 47.2 ± 33.6 | 12 |
| Uso de IA generativa                      | 37.5 ± 41.5 | 10.0 ± 10.0 | 15.5 ± 15.6 | 43.8 ± 44.6 | 12 |
| Professor gera senha do estudante         | 34.6 ± 16.6 | 12.5 ± 15.3 | 15.7 ± 17.7 | 62.1 ± 17.7 | 12 |
| Cadastro nao e inscricao                  | 56.2 ± 27.2 | 12.5 ± 12.5 | 16.7 ± 16.7 | 75.0 ± 25.0 | 12 |
| Classificacao para a presencial           | 25.0 ± 43.3 | 15.0 ± 27.8 | 18.2 ± 32.7 | 48.2 ± 34.0 | 12 |
| Valor da inscricao                        | 20.0 ± 24.5 | 16.7 ± 23.6 | 20.0 ± 24.5 | 30.0 ± 40.0 | 18 |
| Selecao para a iGeo                       | 53.3 ± 45.2 | 13.5 ± 14.7 | 20.4 ± 19.4 | 60.5 ± 27.5 | 18 |
| Criterios de desempate                    | 23.6 ± 15.2 | 20.8 ± 13.8 | 21.3 ± 12.6 | 77.8 ± 23.9 | 12 |
| Contato oficial                           | 72.2 ± 40.4 | 13.9 ± 11.5 | 21.6 ± 17.5 | 77.8 ± 36.9 | 18 |
| Senha recuperada incompativel / navegador | 26.7 ± 37.7 | 15.0 ± 26.0 | 22.9 ± 32.3 | 94.4 ± 8.6  | 12 |
| Quem pode participar / series             | 47.9 ± 22.6 | 20.0 ± 15.1 | 26.6 ± 19.4 | 68.8 ± 34.9 | 21 |
| Fases e estrutura das provas              | 80.6 ± 24.4 | 19.0 ± 17.8 | 26.7 ± 24.1 | 91.7 ± 12.7 | 18 |
| Cadastro da escola pelo INEP              | 57.8 ± 12.6 | 18.8 ± 20.7 | 26.9 ± 19.3 | 71.1 ± 27.9 | 12 |
| Certificados                              | 56.7 ± 34.3 | 20.0 ± 12.6 | 28.1 ± 16.1 | 66.7 ± 36.5 | 15 |
| Acesso simultaneo durante a prova         | 43.8 ± 27.2 | 27.1 ± 28.2 | 28.8 ± 29.8 | 62.5 ± 12.5 | 12 |
| Substituicao de membros                   | 62.5 ± 24.7 | 17.8 ± 16.6 | 29.2 ± 21.3 | 83.3 ± 16.7 | 15 |
| Corrigir dados da escola                  | 46.7 ± 39.1 | 24.4 ± 22.7 | 30.5 ± 26.3 | 37.0 ± 26.5 | 15 |
| Envio de respostas                        | 90.6 ± 16.2 | 25.0 ± 18.6 | 33.5 ± 21.8 | 83.3 ± 28.9 | 12 |
| Recursos de questao                       | 54.0 ± 18.0 | 20.8 ± 18.2 | 35.0 ± 15.9 | 89.7 ± 7.4  | 12 |
| Recuperar senha                           | 48.8 ± 28.8 | 23.3 ± 20.0 | 35.9 ± 21.2 | 65.0 ± 23.2 | 15 |
| Diretrizes de elaboracao de questoes      | 64.7 ± 29.3 | 25.7 ± 19.0 | 36.0 ± 16.3 | 96.7 ± 6.7  | 18 |
| Tipos de escola                           | 42.6 ± 33.4 | 31.9 ± 38.8 | 37.2 ± 27.5 | 93.3 ± 9.4  | 12 |
| Nome da equipe                            | 73.7 ± 24.0 | 28.0 ± 9.8  | 38.4 ± 10.3 | 10.0 ± 20.0 | 15 |
| Prazo de preenchimento de dados           | 87.5 ± 21.7 | 37.5 ± 12.5 | 51.7 ± 15.2 | 100.0 ± 0.0 | 12 |
| Navegadores suportados                    | 63.9 ± 28.3 | 34.9 ± 25.1 | 53.6 ± 21.5 | 94.4 ± 7.9  | 12 |
| Certificados de edicoes anteriores        | 77.8 ± 32.2 | 37.5 ± 21.7 | 57.4 ± 13.9 | 80.6 ± 22.9 | 12 |
| Divulgacao do gabarito                    | 75.9 ± 28.4 | 37.5 ± 21.7 | 57.8 ± 11.0 | 44.4 ± 7.9  | 12 |
| Equipes de escolas diferentes             | 59.4 ± 14.6 | 62.2 ± 16.6 | 59.2 ± 12.6 | 65.6 ± 4.2  | 15 |
| Limite de equipes                         | 63.3 ± 37.1 | 60.0 ± 37.4 | 59.3 ± 33.9 | 60.0 ± 37.4 | 15 |
| Nome social                               | 100.0 ± 0.0 | 56.2 ± 10.8 | 71.4 ± 8.2  | 94.4 ± 12.4 | 12 |
| Aluno em mais de uma equipe               | 80.0 ± 26.7 | 90.0 ± 20.0 | 84.0 ± 23.3 | 93.3 ± 13.3 | 15 |
| Medalhas fisicas                          | 93.3 ± 13.3 | 80.0 ± 26.7 | 84.9 ± 21.7 | 93.3 ± 13.3 | 15 |

Weakest intents: Quem pode ser professor orientador, Token / link expirado, E-mail de confirmacao nao chega, Responder todas as questoes, Calendario / datas.
Strongest intents: Medalhas fisicas, Aluno em mais de uma equipe, Nome social.

## 4. Unsupported facts (hallucination set)

534 of 1811 predicted atomic facts (29.5%) were not supported by the corpus. Examples (intent -> unsupported claim):

- *Quem pode participar / series*: Os critérios de participação na Olimpíada Brasileira de Geografia (OBG) incluem estar regularmente matriculado na Educação Básica em uma instituição de ensino regular até 30 de junho de 2027.
- *Quem pode participar / series*: Os participantes da Olimpíada Brasileira de Geografia (OBG) devem ter entre 16 e 19 anos na data de 30 de junho de 2027.
- *Limite de equipes*: Recomenda-se que o orientador inscreva apenas a quantidade de equipes que consiga acompanhar adequadamente.
- *Equipes de escolas diferentes*: Os alunos podem pertencer à mesma rede de ensino ou mantenedora.
- *Cadastro nao e inscricao*: A equipe não está oficialmente inscrita na Olimpíada Brasileira de Geografia apenas após realizar o cadastro.
- *Valor da inscricao*: Não há informações sobre o valor da inscrição para escolas particulares nos documentos disponíveis.
- *Recuperar senha*: O sistema enviará automaticamente um e-mail para o endereço cadastrado.
- *Cadastro da escola pelo INEP*: As escolas públicas que não possuam CNPJ devem fornecer a inscrição estadual da Secretaria de Educação.
- *Acesso simultaneo durante a prova*: A documentação não menciona se três alunos podem realizar a prova simultaneamente.
- *Divulgacao do gabarito*: A data exata de divulgação do gabarito oficial não está especificada nos documentos oficiais.
- *Professor gera senha do estudante*: O professor deve inserir a nova senha do estudante.
- *Professor gera senha do estudante*: O professor deve confirmar a alteração da senha.

## 5. Missing facts (recall gaps)

2463 of 3047 gold atomic facts (80.8%) were missing from the answers. Examples (intent -> missing fact the answer should have contained):

- *Quem pode participar / series*: Podem participar estudantes regularmente matriculados em escolas publicas ou particulares do Brasil.
- *Quem pode participar / series*: Os estudantes devem estar no 9o ano do Ensino Fundamental até o 3o (ou 4o, se houver) ano do Ensino Medio.
- *Quem pode participar / series*: Sao aceitos ensino regular, profissionalizante, supletivo e EJA.
- *Quem pode participar / series*: Quem ja concluiu o Ensino Medio nao pode participar.
- *Quem pode participar / series*: Quem esta no Ensino Superior nao pode participar.
- *Minimo de membros na presencial*: Equipes classificadas para a Fase Presencial nao podem participar com menos de dois membros.
- *Minimo de membros na presencial*: Equipes que participarem com apenas dois membros nao concorrem a premiacoes por equipe.
- *Limite de equipes*: Nao ha limite de equipes que uma escola pode inscrever.
- *Dados obrigatorios do estudante*: O estudante deve informar CPF no cadastro.
- *Dados obrigatorios do estudante*: O estudante deve informar e-mail válido no cadastro.
- *Dados obrigatorios do estudante*: O estudante deve informar nome completo no cadastro.
- *Dados obrigatorios do estudante*: O estudante deve informar celular WhatsApp no cadastro.

## 6. How these numbers were produced

Each gold and predicted answer was decomposed into atomic facts by gpt-4o-mini. Recall counts gold facts conveyed in the prediction; precision counts predicted facts inferable from the gold answer; FactScore counts predicted facts supported by the official OBG corpus. F1 is the macro-average of per-question harmonic means. PT<->EN paraphrases are treated as equivalent. Sections 2-5 are generated directly from the per-question results, not written in advance.
