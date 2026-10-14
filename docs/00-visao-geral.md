# Visão geral

Este diretório documenta, etapa por etapa, como o serviço foi montado e como reproduzir cada
etapa. Os guias podem ser lidos em ordem (do ambiente de desenvolvimento até o monitoramento) ou
de forma independente, consultando apenas o tema de interesse.

## O problema

Um sistema monitorado (um servidor, uma máquina industrial, uma aplicação) gera várias métricas
ao mesmo tempo: CPU, memória, temperatura, latência etc. Isso é uma **série temporal
multivariada**: uma tabela com uma linha por instante e uma coluna por variável.

O objetivo é responder três perguntas:

1. **Quando** o sistema se comportou de forma anormal?
2. **Quão grave** foi?
3. **Por quê**: quais variáveis causaram a anomalia, e se o problema foi um valor fora do comum
   (*magnitude*) ou uma mudança na forma como as variáveis se relacionam (*correlação*)?

## O modelo (MSCVAE) em linguagem simples

1. **Matrizes de correlação em várias escalas.** Para cada instante, o modelo olha as últimas 10,
   30 e 60 medições e calcula o quanto cada par de variáveis se move junto. O resultado é uma
   "fotografia" das relações entre as variáveis naquele momento.
2. **Aprender o normal.** Uma rede neural (um *autoencoder variacional*) é treinada só com dados
   normais para comprimir e reconstruir essas fotografias e os valores brutos.
3. **Medir a surpresa.** Em dados novos, o erro de reconstrução vira o **score** de anomalia:
   quanto pior o modelo consegue reconstruir, mais incomum é o momento.
4. **Decidir o que é anômalo.** O **limiar** (*threshold*) é calculado com Teoria de Valores
   Extremos (algoritmo SPOT): ele modela os erros mais altos observados em dados normais e escolhe
   um valor que dados normais só ultrapassariam com probabilidade de 1 em 10.000.
5. **Explicar.** O erro de cada variável é comparado com o erro típico dela mesma. As variáveis
   que mais se desviaram do próprio normal são apontadas como causas.

## Arquitetura do serviço

```
                 ┌──────────────────────── docker compose ───────────────────────────┐
 cliente ──HTTP──▶ api (FastAPI) ──enfileira──▶ redis ◀──consome── worker (Celery)   │
                 │   │ POST /v1/detect          (fila + resultados)      │            │
                 │   └──▶ InferenceService ◀── models:/<nome>@champion ──┘            │
                 │                                   ▲  (recarga automática)          │
                 │ trainer (job avulso) ──registra e promove──▶ mlflow ──▶ postgres   │
                 │                                                  └──▶ volume       │
                 │ [perfil monitoring] prometheus ──▶ grafana       (artefatos)      │
                 └────────────────────────────────────────────────────────────────────┘
```

- **api**: recebe as séries e responde com as anomalias. Para séries grandes, envia o trabalho
  para a fila.
- **worker**: processa os trabalhos da fila com o mesmo código da API.
- **mlflow**: guarda o histórico de treinos e as versões dos modelos.
- **trainer**: treina um modelo, registra no MLflow e decide se ele substitui o atual.

## Roteiro das etapas

| Etapa | Guia | Tema |
|---|---|---|
| 0 | [Fundação](fase-0-fundacao.md) | Ambiente Python com uv, organização do código, ferramentas de qualidade |
| 1 | [Modelo pronto para produção](fase-1-modelo.md) | Salvar e carregar o modelo com segurança, reprodutibilidade, testes |
| 2 | [Treino e avaliação](fase-2-treino-mlflow.md) | Pipeline de treino, métricas, registro de experimentos no MLflow, dados próprios |
| 3 | [Versionamento](fase-3-registry.md) | Versões de modelo, promoção automática, rollback |
| 4 | [API](fase-4-api.md) | Endpoints, validação, eventos e explicações |
| 5 | [Filas](fase-5-filas.md) | Processamento assíncrono com Celery e Redis |
| 6 | [Containers](fase-6-containers.md) | Imagens Docker e a stack completa com Compose |
| 7 | [CI/CD](fase-7-cicd.md) | Testes automáticos e publicação de imagens no GitHub |
| 8 | [Observabilidade](fase-8-observabilidade.md) | Métricas, dashboard e logs |

## Glossário

| Termo | Significado |
|---|---|
| Janela | Trecho com as últimas N medições, usado para calcular uma "fotografia" do sistema |
| Score (φ) | Número que mede o quão incomum é um instante; maior significa mais anômalo |
| Limiar (*threshold*) | Valor de score acima do qual um instante é considerado anômalo |
| SPOT / EVT | Método estatístico que define o limiar a partir dos valores extremos dos dados normais |
| Evento | Período contínuo de instantes anômalos, reportado como uma única anomalia |
| RCA (*root cause analysis*) | Análise de causa raiz: quais variáveis explicam a anomalia |
| Aquecimento (*warm-up*) | Primeiros pontos de uma série, sem histórico suficiente para uma avaliação confiável |
| MLflow | Ferramenta que registra treinos (parâmetros, métricas, arquivos) e versiona modelos |
| Model Registry | Catálogo de modelos do MLflow, com versões numeradas (1, 2, 3...) |
| Alias | Apelido que aponta para uma versão: `@champion` (em produção) e `@challenger` (candidato) |
| Quality gate | Regra que decide se o candidato substitui o modelo em produção |
| Fila / broker | Lista de trabalhos pendentes (aqui, no Redis), consumida pelos workers |
| Worker | Processo que executa os trabalhos da fila |
| Hot reload | Troca do modelo em uso sem reiniciar o serviço |
