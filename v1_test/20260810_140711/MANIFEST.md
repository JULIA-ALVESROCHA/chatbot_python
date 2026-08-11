# Bateria — 20260810_140711

| etapa | status | tempo |
|---|---|---|
| qa_regression_runner | `1` FALHOU | 0s |
| qa_evaluator | `0` OK | 0s |
| rtv_runner | `0` OK | 561s |
| rtv_evaluator | `0` OK | 3s |
| reports | `0` OK | 0s |
| diagnose_v2 | — | pulado (arquivo ausente) |
| diagnose_retrieval | `0` OK | 9s |
| check_chunks | `0` OK | 3s |
| check_metadata | `0` OK | 4s |
| pytest_src | `5` FALHOU | 1s |
| pytest_tests | `5` FALHOU | 0s |
| lumie_eval | — | pulado (--fast) |

## lumie_eval

| recorte | n | P | R | F1 | FactScore |
|---|---|---|---|---|---|
| tudo | 600 | 35.2 | 16.9 | 19.7 | 44.7 |
| recusas | 222 | 1.4 | 0.0 | 0.0 | 0.9 |
| respondidas | 378 | 55.1 | 26.8 | 31.2 | 70.4 |

**Taxa de recusa: 37.0%**

- commit: `ca0dbd3`
- etapas com falha: **3**
