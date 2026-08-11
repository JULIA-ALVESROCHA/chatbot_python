# Lumie re-evaluation — results

- Test units per seed: **3**
- Seeds: **[13]**
- Judge / decomposer: **gpt-4o-mini** (temperature 0.2)
- All numbers are percentages, reported as mean ± std across seeds.

## Main table

| Slice | Precision | Recall | F1 (macro) | FactScore |
|---|---|---|---|---|
| **Overall** | 47.8 ± 0.0 | 50.0 ± 0.0 | 48.5 ± 0.0 | 61.1 ± 0.0 |
| lang=en | 83.3 ± 0.0 | 75.0 ± 0.0 | 78.9 ± 0.0 | 83.3 ± 0.0 |
| lang=pt | 30.0 ± 30.0 | 37.5 ± 37.5 | 33.3 ± 33.3 | 50.0 ± 50.0 |

## By intent

| Intent | Precision | Recall | F1 | FactScore |
|---|---|---|---|---|
| Cronograma e Datas | 60.0 ± 0.0 | 75.0 ± 0.0 | 66.7 ± 0.0 | 100.0 ± 0.0 |
| Login Sistema Antigo / Erro de Senha | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Token Invalido ou Expirado | 83.3 ± 0.0 | 75.0 ± 0.0 | 78.9 ± 0.0 | 83.3 ± 0.0 |
