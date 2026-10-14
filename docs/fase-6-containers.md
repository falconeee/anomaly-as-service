# Fase 6: Containers e Docker Compose

Esta etapa empacota o serviço em imagens Docker e descreve a infraestrutura completa (API, worker,
MLflow, banco de dados, fila e monitoramento) em um único arquivo, `docker-compose.yml`. Com isso,
o ambiente inteiro sobe com um comando, igual em qualquer máquina.

## Serviços

| Serviço | Imagem | Função | Porta no host |
|---|---|---|---|
| `api` | `docker/Dockerfile` | API FastAPI | 8000 |
| `worker` | mesma imagem | Processa a fila | — |
| `trainer` | mesma imagem + grupo `train` | Treina e registra modelos (executado sob demanda) | — |
| `mlflow` | `docker/mlflow.Dockerfile` | Histórico de treinos e catálogo de modelos | 5050 |
| `postgres` | `postgres:16-alpine` | Banco de dados do MLflow | — |
| `redis` | `redis:7.4-alpine` | Fila, resultados e registro dos trabalhos | — |
| `prometheus` | `prom/prometheus` | Coleta de métricas (perfil `monitoring`) | 9090 |
| `grafana` | `grafana/grafana` | Dashboard (perfil `monitoring`) | 3000 |

Serviços com perfil só sobem quando pedidos: `trainer` com `--profile train` (ou
`docker compose run trainer`) e Prometheus/Grafana com `--profile monitoring`.

## Como a imagem é construída

O `docker/Dockerfile` tem duas etapas:

```
build:   Python + uv ──▶ instala dependências ──▶ instala o pacote   (resultado: /app/.venv)
runtime: Python ◀── copia apenas /app/.venv  (sem uv, sem compiladores, usuário sem privilégios)
```

Detalhes relevantes:

- **Ordem das camadas.** Primeiro são instaladas só as dependências, a partir de `pyproject.toml` e
  `uv.lock`. Essa camada (grande, por causa do PyTorch) só é refeita quando as dependências mudam;
  alterar o código refaz apenas a camada pequena do pacote.
- **Cache do uv.** Os downloads ficam em cache entre builds (`--mount=type=cache`).
- **`--frozen`.** Instala exatamente o `uv.lock` e falha se ele estiver desatualizado.
- **PyTorch sem GPU.** A imagem tem cerca de 1,9 GB, dos quais 650 MB são o PyTorch; com a versão
  para GPU passaria de 5 GB.
- **Usuário sem privilégios** (uid 10001): limita o estrago caso o processo seja comprometido.

## Uma imagem, vários papéis

O script `docker/entrypoint.sh` decide o que o container executa:

| Comando | Executa |
|---|---|
| `api` | Servidor da API (uvicorn) |
| `worker` | Worker do Celery |
| `train ...` | Pipeline de treino (mesmos argumentos de `python -m anomaly_service.training.train`) |
| `promote ...` | Comandos de versão: `challenger`, `rollback`, `status` |

Como API e worker usam a mesma imagem, executam exatamente o mesmo código.

## MLflow: banco de dados e arquivos separados

- **Postgres** guarda informações pequenas e consultáveis: treinos, parâmetros, métricas, versões,
  aliases e tags.
- Um **volume** (`/mlartifacts`) guarda os arquivos grandes: modelos, gráficos e model cards.
- Com `--serve-artifacts`, o próprio servidor MLflow recebe e entrega esses arquivos. API, worker e
  trainer só precisam do endereço do MLflow (`MLFLOW_TRACKING_URI`); nenhum deles precisa de
  credenciais de armazenamento.

Em nuvem, basta trocar o destino dos arquivos por um bucket (por exemplo
`--artifacts-destination s3://bucket/caminho`); a biblioteca necessária (boto3) já está na imagem.

## Ordem de inicialização

`depends_on` com `condition: service_healthy` faz cada serviço esperar o anterior estar pronto:
Postgres → MLflow → API e worker. A verificação de saúde da API usa `/health/live` (processo no ar)
e não `/health/ready` (modelo carregado), porque numa instalação nova ainda não existe modelo e a
API precisa subir mesmo assim.

## Escolhas de projeto

- **Volume em vez de MinIO.** A ideia inicial era usar o MinIO como armazenamento compatível com
  S3, mas o projeto deixou de publicar imagens públicas (o repositório `minio/minio` não responde
  mais no Docker Hub). O volume com o proxy do MLflow mantém a separação entre banco e arquivos.
- **Porta 5050 para o MLflow.** No macOS, a porta 5000 é ocupada pelo Receptor AirPlay. A porta é
  configurável (`MLFLOW_PORT`) e também é incluída em `--allowed-hosts`, a lista de endereços que o
  MLflow aceita (proteção contra ataques de DNS rebinding).
- **Driver `postgresql+psycopg://`.** A versão atual do SQLAlchemy usa o driver psycopg 3 por
  padrão; o driver fica explícito para não depender disso.
- **`OMP_NUM_THREADS=2`.** Sem isso, cada processo PyTorch criaria uma thread por núcleo da máquina.
- **`GIT_PYTHON_REFRESH=quiet`.** A imagem não tem git; o commit é informado na construção da
  imagem (`GIT_SHA`).

## Passo a passo para reproduzir

Pré-requisitos: Docker Desktop (ou Docker Engine) com Compose e `make`. Se o comando `docker` não
for encontrado no macOS, adicionar `/Applications/Docker.app/Contents/Resources/bin` ao `PATH`.

1. Construir e subir a stack:
   ```bash
   make up
   ```
   Resultado esperado: `Container ... Healthy` para `postgres`, `redis`, `mlflow`, `api` e `worker`.
2. Conferir o estado:
   ```bash
   docker compose ps
   ```
3. Treinar e promover um modelo dentro do Docker:
   ```bash
   make train-docker
   ```
   Resultado esperado no final: `promoted=True`.
4. Executar o teste de ponta a ponta:
   ```bash
   make e2e
   ```
   Resultado esperado no final: `E2E OK`.
5. Abrir as interfaces: http://localhost:5050 (MLflow) e http://localhost:8000/docs (API).
6. Parar a stack:
   ```bash
   make down           # mantém os dados (modelos, histórico)
   make clean-docker   # apaga também os volumes
   ```
