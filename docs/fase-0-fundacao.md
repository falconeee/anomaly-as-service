# Fase 0: Fundação do projeto

## O que construímos

- Um projeto Python empacotado (`anomaly-service`) em **src-layout**, com o modelo movido para
  `src/anomaly_service/model/mscvae.py`.
- Ambiente gerenciado pelo **uv**: `.venv`, `pyproject.toml` e `uv.lock`.
- Dependências separadas em runtime, `dev` (ferramentas) e `train` (só para treinar).
- PyTorch **CPU-only no Linux** (imagens Docker menores) e o wheel padrão com MPS no macOS.
- Qualidade automatizada: ruff (lint e formatação), mypy, pytest e pre-commit.
- O import quebrado `from src.utils import logger` foi substituído por
  `anomaly_service/logger.py`.

## Conceitos

### `uv venv`, `uv add`, `uv sync` e `uv run`

| Comando | O que faz |
|---|---|
| `uv venv --python 3.12` | Cria o `.venv` (baixa o Python se necessário) |
| `uv add pacote` | Adiciona ao `pyproject.toml`, **resolve** e atualiza o `uv.lock`, depois instala |
| `uv add --dev` / `--group train` | O mesmo, mas num grupo de dependências (PEP 735) |
| `uv sync` | Deixa o `.venv` **idêntico** ao lock (instala o que falta, remove o que sobra) |
| `uv sync --frozen` | Usa o lock como está, sem re-resolver. É o modo de CI e Docker |
| `uv run cmd` | Roda o comando dentro do ambiente, sincronizando antes se preciso |

### Por que existe um lockfile

O `pyproject.toml` declara **intervalos** (`torch>=2.14.1`), e o `uv.lock` grava a **versão exata
e o hash** de cada pacote, transitivos incluídos (141 aqui). Com ele, seu Mac, o CI e a imagem
Docker instalam exatamente os mesmos bytes. O lock do uv é **universal**: um único arquivo serve
macOS e Linux. O `environments` em `[tool.uv]` limita a resolução a esses dois sistemas.

### Índice explícito do PyTorch

No Linux, o `torch` do PyPI vem com CUDA, o que soma alguns GB de bibliotecas NVIDIA. O índice
`download.pytorch.org/whl/cpu` publica builds só de CPU. Com `explicit = true`, esse índice
**só** é consultado para pacotes que optam por ele (`[tool.uv.sources]`), o que evita que um
índice de terceiros "sequestre" outras dependências (*dependency confusion*). Confira no lock:
`torch 2.14.1+cpu` para Linux e `torch 2.14.1` para macOS.

### Grupos de dependências

- **runtime** (`[project.dependencies]`): o que a API e o worker precisam em produção. Usamos
  `mlflow-skinny`, que basta para carregar modelos e falar com o servidor.
- **dev**: pytest, ruff, mypy e pre-commit. Nunca vão para a imagem.
- **train**: matplotlib e o `mlflow` completo (UI e servidor local).

`default-groups = ["dev", "train"]` faz o `uv sync` local instalar tudo. O Docker vai usar
`--no-default-groups`.

### src-layout

Com o código em `src/anomaly_service/`, o pacote **não** é importável a partir da raiz do repo
por acidente. Os testes passam a exercitar o pacote *instalado* (o `uv sync` o instala em modo
editável), igual ao que acontece em produção. No layout plano, `import anomaly_service`
funcionaria só porque você está na pasta certa.

### Pre-commit com hooks locais

Os hooks do ruff rodam via `uv run`, então usam **a versão do lock**. Não há uma segunda versão
de ruff no `.pre-commit-config.yaml` para ficar dessincronizada. O hook `uv lock --check` impede
commitar um `pyproject.toml` alterado sem atualizar o lock.

## Decisões

- [ADR 0001: uv como gerenciador de pacotes e ambientes](adr/0001-uv.md)
- O módulo de log se chama `logger.py`, não `logging.py`, para não ter o mesmo nome do
  `logging` da biblioteca padrão.
- O `mscvae.py` foi **só formatado** (ruff format) e teve o import corrigido. Lint e mypy estão
  desligados para ele via `per-file-ignores`. A limpeza e as correções de verdade são a Fase 1.

## Como verificar

```bash
make setup        # venv + uv sync --frozen + git hooks
make check        # ruff + mypy + pytest
uv tree --depth 1 # árvore de dependências diretas
uv run python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```
