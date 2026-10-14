# Fase 7: Integração e entrega contínuas (CI/CD)

Os workflows do GitHub Actions executam todas as verificações a cada push e pull request, incluindo
um teste de ponta a ponta com a stack Docker, e publicam as imagens quando uma versão é marcada com
uma tag.

## Arquivos principais

| Arquivo | Quando executa | O que faz |
|---|---|---|
| `.github/workflows/ci.yml` (job `test`) | Push na `main` e pull requests | Instala dependências do `uv.lock`, ruff, mypy, pytest com cobertura |
| `.github/workflows/ci.yml` (job `e2e`) | Depois do `test` | Constrói as imagens, sobe a stack, executa `scripts/e2e.sh`, mostra os logs se falhar |
| `.github/workflows/release.yml` | Push de tag `v*` (ex.: `v0.1.0`) | Publica as imagens no GitHub Container Registry (GHCR) |
| `.github/dependabot.yml` | Semanalmente | Abre pull requests com atualizações de dependências |
| `scripts/e2e.sh` | CI e `make e2e` | Teste de ponta a ponta |

## Camadas de teste

```
        ponta a ponta (CI, Docker)    stack real: MLflow + Postgres + Redis + worker
      ─────────────────────────────
     API e integração                 API em memória + Redis simulado; MLflow em SQLite
   ───────────────────────────────
  unitários                           modelo, métricas, eventos, explicação, leitura de dados
```

- Os testes unitários e de integração são rápidos (~20 s) e rodam a cada commit.
- O teste de ponta a ponta é mais lento, mas é o único que detecta erros de configuração: porta
  ocupada, driver de banco ausente, permissões, endereço aceito pelo MLflow. Vários erros desse tipo
  só apareceram na primeira execução real do Docker Compose deste projeto.

## O que o teste de ponta a ponta verifica

1. Treina um modelo com dados sintéticos dentro do container `trainer`, registra e promove.
2. Pede à API para recarregar o modelo e espera `/health/ready` responder 200.
3. Envia `examples/request_synthetic.json` para `/v1/detect` e confere que há anomalias.
4. Envia a mesma série para `/v1/jobs` e acompanha até `succeeded`.
5. Confere que `/metrics` expõe as métricas do serviço.

O teste usa um modelo treinado de verdade, e não uma simulação, para percorrer o mesmo caminho da
produção.

## Cache

- `astral-sh/setup-uv` guarda os pacotes baixados entre execuções; sem mudança no `uv.lock`, nada é
  baixado de novo.
- No release, as camadas Docker são reaproveitadas (`cache-from/to: type=gha`).

## Publicação das imagens

Ao criar uma tag `v1.2.3`, o workflow publica duas imagens, cada uma para processadores Intel/AMD
(`amd64`) e ARM (`arm64`, como Macs com Apple Silicon):

- `ghcr.io/<usuário>/anomaly-service` com as tags `1.2.3`, `1.2` e `sha-<commit>`;
- `ghcr.io/<usuário>/anomaly-service-mlflow` com as mesmas tags.

A autenticação usa o token automático do GitHub Actions (`GITHUB_TOKEN`); nenhum segredo adicional
é necessário.

## Permissões

Os workflows pedem apenas o necessário: leitura do repositório (`contents: read`); o release também
pede `packages: write` para publicar imagens. Execuções antigas do mesmo branch são canceladas
quando uma nova começa (`concurrency`).

## Passo a passo para reproduzir

1. Executar localmente as mesmas verificações do job `test`:
   ```bash
   make check
   ```
2. Executar localmente o equivalente ao job `e2e`:
   ```bash
   make up && make e2e
   ```
3. Disparar o CI: enviar um commit para o GitHub e acompanhar na aba *Actions* (jobs `test` e
   `e2e`).
4. Publicar imagens (quando desejado):
   ```bash
   git tag v0.1.0 && git push origin v0.1.0
   ```
   As imagens aparecem em *Packages* no perfil do GitHub.
