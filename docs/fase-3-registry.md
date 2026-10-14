# Fase 3: Versionamento e promoção de modelos

Cada treino registrado vira uma nova **versão** do modelo no catálogo do MLflow (Model Registry).
Uma regra automática decide se a nova versão substitui a que está em produção. Promover ou voltar
atrás é apenas mudar um apelido, sem reconstruir ou reiniciar o serviço.

## Arquivos principais

| Arquivo | Função |
|---|---|
| `src/anomaly_service/training/promote.py` | Regra de promoção, rollback e listagem de versões |
| `src/anomaly_service/training/train.py` | Registra a versão nova (`--register`) e chama a promoção (`--promote`) |

## Versões e apelidos (aliases)

- **Versão**: cada modelo registrado recebe um número (1, 2, 3...) e não muda mais.
- **Alias**: apelido que aponta para uma versão e pode ser movido:
  - `@challenger`: a versão mais recente, candidata;
  - `@champion`: a versão em produção, usada pela API e pelos workers.

A API sempre carrega `models:/<nome>@champion`. Quando o alias muda, a API percebe e troca de
modelo sozinha (Fase 4).

O MLflow tinha um mecanismo anterior, os *stages* (`Staging`, `Production`), que foi descontinuado
em favor dos aliases.

## Regra de promoção (quality gate)

```
treino ──registra──▶ versão N com @challenger ──regra──▶ aprovada: @champion passa para N
                                                    └──▶ reprovada: @champion continua igual
```

A regra carrega as duas versões (candidata e atual) **do próprio catálogo**, do mesmo jeito que a
API faria, e avalia ambas nos **mesmos dados de teste**. A candidata é aprovada se:

1. a latência estiver dentro do limite (`max_latency_ms`); e
2. não houver modelo em produção, **ou** a métrica escolhida (`f1` por padrão) for maior ou igual à
   do modelo atual menos uma tolerância (`tolerance`, 0,01 por padrão).

Motivos para avaliar as duas versões novamente, em vez de comparar números antigos:

- a comparação é justa (mesmos dados, mesmo código de avaliação);
- se o modelo registrado não carregar (dependência ausente, arquivo corrompido), a falha aparece
  antes de ele entrar em produção.

A tolerância evita trocas desnecessárias: dois treinos equivalentes variam um pouco por acaso, e
sem margem um modelo igual ao atual poderia ser reprovado ou aprovado aleatoriamente.

## Registro da decisão

A decisão fica gravada como *tags* na própria versão, visíveis na interface do MLflow:

| Tag | Exemplo |
|---|---|
| `promotion.decision` | `promoted`, `rejected` ou `rollback` |
| `promotion.reason` | `f1 0.3694 < champion 0.3816 - tolerance 0.01` |
| `promotion.compared_with` | versão comparada (`none` se não havia) |
| `promotion.dataset_digest` | hash dos dados usados na comparação |
| `replaced_by` | na versão antiga: qual versão a substituiu |

Também ficam na versão o commit do código (`git_sha`), o hash dos dados (`data.digest`) e o
*model card* como descrição. Com isso, a partir do modelo em produção é possível chegar ao código,
aos dados e à configuração que o geraram.

### Exemplo real

No SMD, a versão 2 foi reprovada com o motivo `f1 0.3694 < champion 0.3816 - tolerance 0.01`, e
a versão 1 continuou em produção.

## Escolhas de projeto

- Aliases em vez de fixar o número da versão na configuração da API: promover ou voltar atrás não
  exige novo deploy.
- A métrica padrão é o F1 no limiar real, por ser o comportamento que o serviço entrega. Ela pode
  ser trocada em `promotion.metric` na configuração.

## Passo a passo para reproduzir

1. Treinar duas vezes com a mesma configuração:
   ```bash
   make train      # versão 1 vira @champion ("no champion yet")
   make train      # versão 2 é comparada com a 1; mesmo resultado, então é promovida
   ```
2. Listar as versões:
   ```bash
   make status
   ```
   Resultado esperado: uma tabela com `version`, `aliases`, `f1`, `pa_f1`, `auc_pr`, `decision`
   e `git_sha`; a versão 2 com `challenger,champion`.
3. Voltar para a versão 1 (rollback) e conferir:
   ```bash
   uv run python -m anomaly_service.training.promote rollback --model mscvae-synthetic --version 1
   make status
   ```
   Resultado esperado: `champion` na versão 1.
4. Executar os testes do ciclo completo (registro, promoção, reprovação e rollback):
   ```bash
   uv run pytest tests/integration/test_training_pipeline.py -v
   ```
