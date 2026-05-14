# Rinha de Backend 2026 — Detecção de fraude por busca vetorial

Repositório de **especificação** (sem código próprio ainda). Toda a doc fonte está em `docs/br` (PT-BR) e `docs/en` (EN). Os arquivos de dados (`references.json.gz`, `mcc_risk.json`, `normalization.json`) vivem em `/resources` quando o repo oficial for clonado.

## TL;DR do desafio

Construir um módulo de **detecção de fraude** que expõe HTTP na porta `9999`:

- `GET /ready` → 2xx quando pronto.
- `POST /fraud-score` → recebe a transação, devolve `{ "approved": bool, "fraud_score": float }`.

Decisão obrigatória (na referência): vetoriza o payload em 14 dimensões, busca os **5 vizinhos mais próximos** no dataset rotulado, `fraud_score = #fraudes / 5`, `approved = fraud_score < 0.6`. O dataset oficial de teste foi rotulado com **k-NN k=5, distância euclidiana, brute force** — mas você é livre para usar qualquer técnica desde que o resultado bata.

## Vetor de 14 dimensões (ordem fixa)

`limitar(x)` = clamp em `[0, 1]`. Sentinela `-1` em índices 5 e 6 quando `last_transaction == null`.

| i  | dim                     | fórmula                                                                 |
|----|-------------------------|-------------------------------------------------------------------------|
| 0  | amount                  | `limitar(transaction.amount / max_amount)`                              |
| 1  | installments            | `limitar(transaction.installments / max_installments)`                  |
| 2  | amount_vs_avg           | `limitar((transaction.amount / customer.avg_amount) / amount_vs_avg_ratio)` |
| 3  | hour_of_day             | `hora_utc / 23`                                                         |
| 4  | day_of_week             | `dow / 6` (seg=0, dom=6)                                                |
| 5  | minutes_since_last_tx   | `limitar(min / max_minutes)` ou `-1`                                    |
| 6  | km_from_last_tx         | `limitar(km / max_km)` ou `-1`                                          |
| 7  | km_from_home            | `limitar(terminal.km_from_home / max_km)`                               |
| 8  | tx_count_24h            | `limitar(customer.tx_count_24h / max_tx_count_24h)`                     |
| 9  | is_online               | `0`/`1`                                                                 |
| 10 | card_present            | `0`/`1`                                                                 |
| 11 | unknown_merchant        | `1` se `merchant.id` ∉ `customer.known_merchants`                       |
| 12 | mcc_risk                | `mcc_risk.json[mcc]`, default `0.5`                                     |
| 13 | merchant_avg_amount     | `limitar(merchant.avg_amount / max_merchant_avg_amount)`                |

Constantes (`normalization.json`):
`max_amount=10000`, `max_installments=12`, `amount_vs_avg_ratio=10`, `max_minutes=1440`, `max_km=1000`, `max_tx_count_24h=20`, `max_merchant_avg_amount=10000`.

## Dataset

- `references.json.gz` — 3.000.000 vetores rotulados (`fraud` | `legit`). ~16 MB gzipado, ~284 MB descomprimido. **Não muda durante o teste** → pré-processar no build/startup é livre e recomendado.
- `mcc_risk.json` — tabela MCC → score `0..1`.
- `normalization.json` — constantes acima.

## Restrições de infra (críticas)

- **1 load balancer + ≥2 instâncias da API**, round-robin. O LB **não pode aplicar lógica** (não inspeciona payload, não decide, não transforma).
- Soma de todos os serviços: **≤ 1 CPU, ≤ 350 MB** de memória.
- `docker-compose.yml` na raiz da branch `submission`. Imagens públicas, `linux/amd64`.
- Network mode `bridge`. **Sem** `host`, **sem** `privileged`.
- API responde em `9999` (LB é quem expõe).
- Ambiente alvo: Mac Mini Late 2014, 2.6 GHz, 8 GB RAM, Ubuntu 24.04.

## Pontuação (resumo)

`final = score_p99 + score_det`, cada um em `[-3000, +3000]` → total `[-6000, +6000]`.

- **Latência (`score_p99`)**: `1000 · log₁₀(1000 / max(p99, 1ms))`. Satura `+3000` em `p99 ≤ 1ms`. Corte rígido `-3000` se `p99 > 2000ms`.
- **Detecção (`score_det`)**:
  - `E = 1·FP + 3·FN + 5·Err`, `ε = E/N`, `falhas = FP+FN+Err`.
  - Se `falhas/N > 15%` → `score_det = -3000` (corte rígido).
  - Senão: `1000·log₁₀(1/max(ε, 0.001)) − 300·log₁₀(1+E)`.

Pesos: erro HTTP (5) > falso negativo (3) > falso positivo (1). HTTP 500 conta duplo (no `E` e na `failure_rate`).

## Estrutura de submissão

- Branch `main`: código-fonte.
- Branch `submission`: apenas o necessário pra rodar (`docker-compose.yml`, `info.json`, configs). **Sem** código-fonte.
- `participants/<github-user>.json` no repo da Rinha (PR).
- `info.json` com `participants`, `social`, `source-code-repo`, `stack`, `open_to_work`.
- Trigger de teste: issue com `rinha/test` na descrição.
- Licença obrigatória: MIT.

## Proibições explícitas

- LB com lógica de fraude.
- Usar payloads de teste como lookup.
- Repositório privado.
