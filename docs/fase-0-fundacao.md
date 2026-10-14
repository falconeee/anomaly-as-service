# Fase 0: Fundação do projeto

Esta etapa prepara o ambiente de desenvolvimento: um pacote Python instalável, dependências com
versões travadas e ferramentas que verificam o código automaticamente. Tudo o que vem depois
(treino, API, Docker, CI) depende dessa base para ser reproduzível em qualquer máquina.

## Arquivos principais

| Arquivo | Função |
|---|---|
| `pyproject.toml` | Nome do pacote, dependências e configuração das ferramentas |
| `uv.lock` | Versão exata e hash de cada pacote instalado (gerado automaticamente) |
| `.python-version` | Versão do Python usada (3.12) |
| `src/anomaly_service/` | Código do pacote (o modelo está em `model/mscvae.py`) |
| `.pre-commit-config.yaml` | Verificações executadas antes de cada commit |
| `Makefile` | Atalhos para os comandos mais comuns (`make help` lista todos) |

## Gerenciamento do ambiente com uv

O [uv](https://docs.astral.sh/uv/) cria o ambiente virtual, instala as dependências e executa
comandos dentro dele. Os comandos usados no projeto:

| Comando | O que faz |
|---|---|
| `uv venv --python 3.12` | Cria o ambiente virtual `.venv` (baixa o Python se necessário) |
| `uv add pacote` | Adiciona uma dependência ao `pyproject.toml`, atualiza o `uv.lock` e instala |
| `uv add --dev pacote` | O mesmo, no grupo de ferramentas de desenvolvimento |
| `uv sync` | Deixa o `.venv` idêntico ao `uv.lock` (instala o que falta, remove o que sobra) |
| `uv sync --frozen` | Instala exatamente o `uv.lock`, sem recalcular nada (modo usado no CI e no Docker) |
| `uv run comando` | Executa um comando dentro do ambiente |

### Por que existe o `uv.lock`

O `pyproject.toml` declara **faixas** de versão (por exemplo, `torch>=2.14.1`). O `uv.lock`
registra a **versão exata e o hash** de cada pacote, inclusive das dependências indiretas. Com
ele, a máquina de desenvolvimento, o CI e a imagem Docker instalam exatamente os mesmos arquivos.
Um único `uv.lock` atende macOS e Linux.

### PyTorch sem CUDA no Linux

No Linux, o PyTorch publicado no PyPI inclui bibliotecas da NVIDIA para GPU, que somam vários GB.
O projeto não usa GPU em produção, então o `pyproject.toml` aponta o PyTorch do Linux para o
índice de pacotes *CPU-only* do próprio PyTorch. No macOS, continua o pacote padrão, que usa a GPU
do Mac (MPS). O índice é declarado como `explicit = true`, ou seja, é consultado **apenas** para o
PyTorch; isso impede que um índice de terceiros forneça outras dependências por engano.

No `uv.lock`, isso aparece como `torch 2.14.1+cpu` para Linux e `torch 2.14.1` para macOS.

### Grupos de dependências

| Grupo | Conteúdo | Onde é instalado |
|---|---|---|
| principal | Tudo que a API e o worker precisam (torch, pandas, FastAPI, Celery, mlflow-skinny...) | Sempre |
| `dev` | pytest, ruff, mypy, pre-commit, fakeredis, mlflow completo (interface web local) | Só na máquina de desenvolvimento |
| `train` | matplotlib (gráficos do treino) | Desenvolvimento e imagem do trainer |

`uv sync` instala os três grupos. As imagens Docker instalam só o necessário.

## Organização do código (src-layout)

O código fica em `src/anomaly_service/`, e não na raiz do repositório. Assim, o pacote só pode ser
importado depois de instalado (o `uv sync` faz isso). Os testes passam a usar o pacote instalado,
como acontece em produção, em vez de funcionar por acaso por estarem na pasta certa.

## Verificações automáticas

| Ferramenta | Verifica |
|---|---|
| ruff | Erros comuns, imports, estilo e formatação |
| mypy | Tipos (por exemplo, passar texto onde se espera número) |
| pytest | Testes automatizados |
| pre-commit | Executa ruff, formatação e `uv lock --check` antes de cada commit |

Os hooks do pre-commit chamam as ferramentas via `uv run`, usando as mesmas versões do `uv.lock`.
O hook `uv lock --check` impede commitar um `pyproject.toml` alterado sem atualizar o lock.

## Escolhas de projeto

- uv como gerenciador de pacotes: um único `uv.lock` serve macOS e Linux, permite escolher o
  índice do PyTorch por sistema operacional e instala em segundos. Alternativas como
  `pip` + `requirements.txt` ou Poetry exigiriam arquivos separados por plataforma ou mais
  configuração.
- O módulo de log se chama `logger.py` (e não `logging.py`) para não ter o mesmo nome do módulo
  `logging` da biblioteca padrão.
- O arquivo original do modelo (`src/MSCVAE.py`) foi movido para
  `src/anomaly_service/model/mscvae.py`, e o import quebrado `from src.utils import logger` foi
  substituído pelo novo módulo de log.

## Passo a passo para reproduzir

Pré-requisitos: [uv](https://docs.astral.sh/uv/getting-started/installation/) (no macOS:
`brew install uv`) e `make`.

1. Criar o ambiente e instalar as dependências:
   ```bash
   make setup
   ```
   Resultado esperado: pasta `.venv` criada e hooks do pre-commit instalados.
2. Executar todas as verificações:
   ```bash
   make check
   ```
   Resultado esperado: `All checks passed!` (ruff), `Success: no issues found` (mypy) e todos os
   testes passando.
3. (Opcional) Conferir a instalação do PyTorch:
   ```bash
   uv run python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
   ```
   No macOS com Apple Silicon, o segundo valor é `True`.
