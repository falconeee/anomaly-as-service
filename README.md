# anomaly-as-service

API que recebe uma série temporal multivariada e devolve as anomalias **com explicação** (variáveis
causadoras, desvio em relação ao normal, valor observado vs. esperado).

- ⚙️ **FastAPI**: detecção síncrona e assíncrona
- 📬 **Filas**: Celery + Redis para séries grandes
- 🏷️ **Versionamento de modelos**: MLflow Model Registry com aliases `@champion` / `@challenger`
- 🔁 **MLOps**: tracking de experimentos, quality gate de promoção, CI/CD e monitoramento

O modelo é o **MSCVAE**: um VAE convolucional multiescala sobre matrizes de correlação, com um
Transformer temporal, limiar estatístico por Extreme Value Theory (SPOT) e análise de causa raiz
por variável.

> 🚧 Em construção, fase a fase. Veja o roteiro e os estudos em [`docs/`](docs/00-visao-geral.md).

## Rodando localmente (fase atual)

```bash
uv venv --python 3.12   # cria o .venv
uv sync                 # instala as dependências exatas do uv.lock (inclui o grupo dev)
make test               # roda a suíte de testes
```
