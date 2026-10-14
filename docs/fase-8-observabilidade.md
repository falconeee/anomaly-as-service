# Fase 8: Observabilidade e monitoramento do modelo

Esta etapa torna visível o que acontece com o serviço em funcionamento: volume e tempo de resposta,
erros, tamanho da fila e, principalmente, o comportamento do modelo diante dos dados que chegam.

## Arquivos principais

| Arquivo | Função |
|---|---|
| `src/anomaly_service/observability.py` | Definição das métricas do modelo e da fila |
| `src/anomaly_service/logger.py` | Logs em texto ou JSON, com identificadores de requisição e de trabalho |
| `monitoring/prometheus.yml` | Onde o Prometheus coleta as métricas |
| `monitoring/grafana/` | Fonte de dados e dashboard carregados automaticamente pelo Grafana |

## Onde as métricas ficam disponíveis

- API: `http://localhost:8000/metrics`
- Cada worker: porta 9100 dentro da rede Docker

O Prometheus coleta esses endereços a cada 15 s, e o Grafana exibe o dashboard
**Anomaly-as-a-Service**.

## Métricas do serviço

Seguem os quatro "sinais de ouro" usados em operações:

| Sinal | Métrica | Painel |
|---|---|---|
| Latência | `http_request_duration_seconds`, `anomaly_inference_seconds{mode}` | Tempo de resposta (p95) por rota; tempo de inferência imediata vs. em fila |
| Tráfego | `http_requests_total` | Requisições por segundo por rota |
| Erros | `http_requests_total{status="5xx"}`, `anomaly_detections_total{outcome="error"}` | Taxa de erros |
| Saturação | `anomaly_queue_depth` | Trabalhos aguardando na fila |

Uma fila que só cresce indica que são necessários mais workers.

## Métricas do modelo

Um serviço pode responder rápido e sem erros e, ainda assim, dar respostas ruins: modelos de
anomalia falham em silêncio quando os dados mudam. Estas métricas acompanham o comportamento do
modelo:

| Métrica | O que revela |
|---|---|
| `anomaly_feature_drift_z{feature}` | **Mudança nos dados**: distância entre a média da última requisição e a média do treino, em desvios-padrão do treino, por variável |
| `anomaly_max_score_ratio` | **Mudança nos scores**: distribuição de "maior score ÷ limiar" por requisição. Se sobe em tráfego normal, o modelo deixou de representar bem o sistema |
| `anomaly_events_total{severity}` | Quantidade de anomalias por severidade. Um salto pode ser um incidente real ou um modelo desatualizado |
| `anomaly_model_info{name,version,alias}` | Versão do modelo em uso em cada processo. Muda quando o `@champion` é trocado, o que permite relacionar mudanças de comportamento a trocas de modelo |

Foi esse tipo de mudança nos dados que gerou excesso de alarmes no SMD (Fase 2). Em produção, essas
duas primeiras métricas indicariam o problema antes das reclamações sobre alertas.

## Vários workers

Cada container de worker expõe suas próprias métricas. O Prometheus descobre todos os containers do
serviço `worker` pelo DNS interno do Docker (`dns_sd_configs`), então aumentar o número de workers
não exige alterar configuração. As métricas do modelo aparecem tanto na API (`mode="sync"`) quanto
nos workers (`mode="async"`), e o dashboard soma as duas origens.

## Logs estruturados

Com `LOG_JSON=true` (padrão no Docker), cada linha de log é um objeto JSON:

```json
{"ts": "2026-10-14T09:58:46.845+00:00", "level": "INFO", "logger": "anomaly_service", "msg": "Job done: 1500 points, 3 anomalies", "job_id": "ef4fbc65..."}
```

- Na API, cada requisição recebe um `request_id` (aproveitando o cabeçalho `X-Request-ID`
  enviado pelo cliente, ou gerando um novo), que aparece em todos os logs da requisição e é
  devolvido no cabeçalho da resposta.
- No worker, cada linha traz o `job_id` do trabalho em execução.

Com isso, é possível filtrar todos os logs de uma requisição ou trabalho específico.

## Escolhas de projeto

- A mudança nos dados é medida por requisição, comparada com o treino. Análises agregadas (janelas
  de horas, testes estatísticos como PSI ou KS) seriam uma evolução, por exemplo com relatórios
  periódicos sobre amostras das requisições.
- O dashboard é um arquivo JSON versionado no repositório e carregado automaticamente na
  inicialização do Grafana.

## Passo a passo para reproduzir

1. Subir a stack, treinar um modelo e gerar algum tráfego:
   ```bash
   make up && make e2e
   ```
2. Subir Prometheus e Grafana:
   ```bash
   make monitoring
   ```
3. Abrir o Grafana em http://localhost:3000 (acesso anônimo de leitura; administrador:
   `admin`/`admin`) e o dashboard **Anomaly-as-a-Service**.
4. Conferir em http://localhost:9090/targets que `api` e `worker` estão `UP`.
5. Ver as métricas do modelo diretamente:
   ```bash
   curl -s localhost:8000/metrics | grep ^anomaly_
   ```
6. Ver os logs com identificadores:
   ```bash
   docker compose logs api | grep request_id
   ```
