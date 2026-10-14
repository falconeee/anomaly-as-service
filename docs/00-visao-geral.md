# Visão geral

## O problema

Recebemos uma série temporal **multivariada** (ex.: 38 métricas de um servidor) e queremos saber
**quando** o sistema se comportou de forma anormal e **por quê**: quais variáveis causaram a
anomalia, quanto elas desviaram do normal, e se o problema foi de magnitude (um valor estourou)
ou de relação (duas métricas que andavam juntas se descolaram).

## O modelo (MSCVAE), em uma página

1. **Matrizes de atributos multiescala.** Para cada instante, o `AttributeMatrixGenerator`
   calcula `M = X·Xᵀ / w` sobre janelas de 10, 30 e 60 pontos. Cada matriz N×N descreve como as
   N variáveis se correlacionam naquela escala de tempo.
2. **Encoder CNN + núcleo VAE.** O encoder comprime as matrizes num espaço latente
   probabilístico (μ, σ) que aprende o que é o "normal".
3. **Transformer temporal.** Olha a sequência das últimas `seq_len` janelas e captura a inércia
   do sistema.
4. **Decoder duplo (híbrido).** Uma rota reconstrói as matrizes (sensível a quebras de
   correlação) e outra reconstrói os valores brutos (sensível a picos).
5. **Score φ.** É o erro de reconstrução das duas rotas, ponderado pelas precisões que o próprio
   modelo aprende (Kendall et al.).
6. **Limiar por EVT (SPOT).** Uma distribuição de Pareto generalizada é ajustada na cauda dos
   scores de treino. O limiar fica no quantil de probabilidade `q = 1e-4`, sem supor que o erro é
   gaussiano.
7. **RCA.** O erro de cada variável é padronizado contra o próprio baseline (mediana + MAD) e
   separado em três canais: diagonal (a variável sozinha), fora da diagonal (as relações dela) e
   valor bruto.

## A arquitetura do serviço

```
                 ┌──────────── docker compose ─────────────────────────────┐
 cliente ──HTTP──▶ api (FastAPI) ──enqueue──▶ redis ◀──consume── worker (Celery)
                 │   │ sync /v1/detect            (broker+results)   │      │
                 │   └──── InferenceService ◀── models:/mscvae-smd@champion ┘
                 │                                   ▲                      │
                 │ trainer (one-off) ──log/register──▶ mlflow ──▶ postgres (metadata)
                 │                                     └──────▶ minio (artefatos S3)
                 │ [profile monitoring] prometheus ──▶ grafana              │
                 └─────────────────────────────────────────────────────────┘
```

## Roteiro de fases

| Fase | Tema | O que você aprende |
|---|---|---|
| 0 | [Fundação](fase-0-fundacao.md) | uv, lockfile, grupos de dependências, src-layout, ruff, pre-commit |
| 1 | Modelo pronto para produção | Serialização segura, reprodutibilidade, testes de código de ML |
| 2 | Dados, treino e avaliação | Experiment tracking, pyfunc, métricas de anomalia em séries temporais |
| 3 | Versionamento | Model Registry, aliases, quality gate, linhagem, rollback |
| 4 | API síncrona | FastAPI lifespan, DI, Pydantic v2, CPU-bound num servidor async |
| 5 | Filas | Celery, at-least-once, idempotência, backpressure |
| 6 | Containers | Docker multi-stage com uv, compose, healthchecks, MLflow com S3 |
| 7 | CI/CD | GitHub Actions, cache, teste e2e, GHCR |
| 8 | Observabilidade | Prometheus/Grafana, logs estruturados, drift |
| 9 | Portfólio | README, model card, ADRs |

Cada fase tem um guia em `docs/fase-N-*.md` com **O que construímos**, **Conceitos**,
**Decisões** e **Como verificar**. As decisões de arquitetura ficam registradas em
[`docs/adr/`](adr/).
