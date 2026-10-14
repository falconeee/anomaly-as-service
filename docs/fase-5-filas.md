# Fase 5: Processamento em segundo plano (filas)

Séries muito grandes demoram para ser analisadas. Em vez de manter a conexão HTTP aberta durante
todo o cálculo, a API registra o pedido em uma **fila** e responde na hora com um identificador.
Processos separados (**workers**) consomem a fila, e o cliente consulta o resultado depois.

## Arquivos principais

| Arquivo | Função |
|---|---|
| `src/anomaly_service/worker/celery_app.py` | Configuração do Celery (fila e entrega das mensagens) |
| `src/anomaly_service/worker/tasks.py` | Tarefas: detecção e envio de webhook |
| `src/anomaly_service/jobs.py` | Registro dos trabalhos e controle de duplicidade no Redis |
| `src/anomaly_service/api/routes.py` | Endpoints `/v1/jobs` |

## Fluxo

```
cliente ──POST /v1/jobs──▶ API ──valida──▶ Redis (fila "detections") ──▶ worker
   ▲                         │                                             │
   └──GET /v1/jobs/{id}──────┴──────── Redis (resultados, guardados 24 h) ◀┘
                                                                           └─▶ webhook (opcional)
```

| Endpoint | Resposta |
|---|---|
| `POST /v1/jobs` | **202** com `job_id`, `status` e `status_url` (também no cabeçalho `Location`) |
| `GET /v1/jobs/{id}` | `status`: `queued` (na fila), `running`, `succeeded` (com `result`) ou `failed` (com `error`) |

O `result` tem exatamente o mesmo formato da resposta de `/v1/detect`.

## Vantagens da fila

- A API só valida e enfileira, então responde em milissegundos.
- Os workers processam no ritmo que conseguem; picos de demanda ficam aguardando na fila em vez de
  derrubar o serviço.
- API e workers podem ser escalados separadamente.
- Se um worker cair no meio de um trabalho, o trabalho volta para a fila.

## Garantia de entrega

A configuração garante que cada trabalho é executado **pelo menos uma vez**:

| Configuração | Efeito |
|---|---|
| `task_acks_late=True` | A mensagem só é removida da fila **depois** que o trabalho termina |
| `task_reject_on_worker_lost=True` | Se o processo do worker morrer, a mensagem volta para a fila |
| `worker_prefetch_multiplier=1` | Cada worker pega um trabalho por vez, sem reservar outros enquanto está ocupado |
| `visibility_timeout` = 2 × limite de tempo | Tempo máximo antes de o Redis considerar uma mensagem perdida e reenviá-la; precisa ser maior que o trabalho mais longo |

"Pelo menos uma vez" significa que, em caso de falha, um trabalho pode ser executado duas vezes.
Isso não causa problema aqui: a detecção sempre produz o mesmo resultado para a mesma entrada.

## Evitar trabalhos duplicados: `Idempotency-Key`

Se o cliente reenviar o mesmo pedido (por exemplo, após um timeout de rede), não deve ser criado
um segundo trabalho. Com o cabeçalho `Idempotency-Key: <chave>`, a primeira requisição associa a
chave ao `job_id`; as seguintes com a mesma chave recebem **200** com o mesmo `job_id`.

## Novas tentativas

| Situação | Comportamento |
|---|---|
| Modelo indisponível (MLflow fora do ar, nenhum modelo promovido) | Nova tentativa com espera crescente, até 5 vezes |
| Variáveis erradas na série | Falha imediata, sem nova tentativa (a API já recusa na submissão quando conhece o modelo) |
| Falha ao entregar o webhook | Nova tentativa, até 6 vezes, sem repetir a detecção |

## Notificação por webhook

Se a requisição incluir `"callback_url": "https://..."`, ao terminar o worker envia um POST para
esse endereço com `{"job_id", "status", "result"}`.

## Um trabalho por worker (`--pool solo`)

O Celery normalmente cria subprocessos copiando o processo principal. Fazer essa cópia depois de o
PyTorch ter iniciado suas threads pode travar o processo, e cada subprocesso teria sua própria
cópia do modelo. Com `--pool solo`, cada container de worker executa um trabalho por vez; para
processar mais trabalhos em paralelo, sobem-se mais containers
(`docker compose up -d --scale worker=3`).

## Diferenciar "na fila" de "não existe"

O Celery informa `PENDING` para **qualquer** identificador desconhecido. Por isso, a API registra
cada trabalho no Redis no momento da submissão e responde **404** para identificadores que nunca
existiram ou que já expiraram.

## Escolhas de projeto

- Celery com Redis: o Redis atende ao mesmo tempo como fila, armazenamento de resultados e
  registro dos trabalhos. RabbitMQ exigiria um serviço extra para os resultados; Kafka é voltado a
  fluxos de eventos, não a filas de trabalho com resultado individual.
- Mesma imagem para API e worker (Fase 6): a detecção imediata e a em fila executam exatamente o
  mesmo código.
- Nos testes, os trabalhos rodam dentro do próprio processo de teste
  (`CELERY_TASK_ALWAYS_EAGER=true`), com fila e resultados em memória e Redis simulado
  (`fakeredis`); o código executado é o mesmo de produção.
- Para séries extremamente grandes, uma evolução possível é guardar a série em um armazenamento de
  arquivos e enviar à fila apenas a referência.

## Passo a passo para reproduzir

Requer a stack Docker (Fase 6), que inclui Redis e worker.

1. Subir a stack e treinar um modelo:
   ```bash
   make up && make train-docker
   ```
2. Enviar um trabalho:
   ```bash
   curl -s -X POST localhost:8000/v1/jobs -H 'X-API-Key: change-me' \
     -H 'Content-Type: application/json' -H 'Idempotency-Key: teste-1' \
     -d @examples/request_synthetic.json
   ```
   Resultado esperado: `{"job_id": "...", "status": "queued", "status_url": "/v1/jobs/..."}`.
3. Consultar o resultado (substituindo o identificador):
   ```bash
   curl -s localhost:8000/v1/jobs/<job_id> -H 'X-API-Key: change-me' | python -m json.tool | head
   ```
   Resultado esperado: `"status": "succeeded"` e o campo `result`.
4. Repetir o passo 2 com a mesma `Idempotency-Key`: o mesmo `job_id` é devolvido.
5. Acompanhar os logs do worker (cada linha traz o `job_id`):
   ```bash
   docker compose logs -f worker
   ```
