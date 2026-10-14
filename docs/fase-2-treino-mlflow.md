# Fase 2: Treino, avaliação e registro no MLflow

Esta etapa transforma o treino em um processo repetível: um arquivo de configuração descreve
dados, modelo e treino; um único comando treina, avalia, registra tudo no MLflow e, se pedido,
publica o modelo no catálogo de versões.

## Arquivos principais

| Arquivo | Função |
|---|---|
| `configs/train_synthetic.yaml` | Treino rápido com dados sintéticos (~15 s em CPU) |
| `configs/train_smd.yaml` | Treino no dataset público SMD |
| `configs/train_csv_example.yaml` | Modelo de configuração para dados próprios (CSV) |
| `src/anomaly_service/training/config.py` | Validação da configuração e carregamento dos dados |
| `src/anomaly_service/training/train.py` | Pipeline: dados → treino → avaliação → MLflow → registro |
| `src/anomaly_service/training/evaluate.py` | Métricas de detecção e de causa raiz |
| `src/anomaly_service/data/smd.py` | Download e leitura do SMD |
| `src/anomaly_service/data/csv_source.py` | Leitura de dados próprios em CSV |
| `src/anomaly_service/model/pyfunc.py` | Adaptador do modelo para o formato padrão do MLflow |

## O arquivo de configuração

```yaml
experiment: mscvae-synthetic        # nome do experimento no MLflow
registered_model: mscvae-synthetic  # nome do modelo no catálogo (usado pela API)
data:
  source: synthetic                 # synthetic | smd | csv
model:
  window_sizes: [5, 10, 20]         # tamanhos das janelas, em linhas
  seq_len: 3                        # quantas janelas anteriores o modelo considera
training:
  epochs: 10
  calibration_fraction: 0.2         # fração final do treino usada para calibrar o limiar
promotion:
  metric: f1                        # métrica usada para decidir a promoção (Fase 3)
```

Campos desconhecidos geram erro, o que evita que um nome digitado errado seja ignorado sem aviso.

## Fontes de dados

| `source` | Dados | Rótulos de anomalia |
|---|---|---|
| `synthetic` | Gerados com anomalias em posições conhecidas | Sim (detecção e causa) |
| `smd` | Server Machine Dataset, 38 métricas de servidor por minuto | Sim (detecção e causa) |
| `csv` | Arquivo próprio | Opcionais (coluna `label`) |

### Dados próprios (CSV)

Formato esperado: uma linha por instante, uma coluna por variável.

```
timestamp,cpu,memoria,latencia_ms,label
2024-01-01T00:00:00,41.2,63.0,120.5,0
2024-01-01T00:01:00,40.8,63.4,118.9,0
```

- `timestamp` (opcional): usado para ordenar as linhas e para mostrar datas nas respostas.
- `label` (opcional): `1` em anomalias conhecidas e `0` no restante. Serve **apenas para avaliar**
  o modelo; o treino não usa rótulos.
- As demais colunas numéricas são as variáveis (ou apenas as listadas em `features`).
- O arquivo é dividido no tempo: as primeiras linhas (70% por padrão) treinam o modelo e as
  últimas (30%) avaliam. O trecho de treino deve representar o funcionamento **normal**.
- Sem a coluna `label`, as métricas de detecção ficam zeradas e a promoção (Fase 3) só verifica a
  latência.

### Variáveis constantes

Uma variável que não muda durante o treino tem desvio-padrão próximo de zero. Na normalização
(subtrair a média e dividir pelo desvio), qualquer variação posterior viraria um número enorme.
Por isso, essas variáveis são removidas (`drop_constant: true`). No SMD, 8 das 38 métricas são
constantes no treino.

## O que fica registrado no MLflow

Cada treino é um *run* no MLflow com:

| Item | Para que serve |
|---|---|
| Parâmetros (toda a configuração) e `config.yaml` | Saber exatamente como o modelo foi treinado |
| Loss por época (`train.loss`) | Acompanhar a convergência do treino |
| Dataset com hash do conteúdo (`data.digest`) | Saber quais dados geraram o modelo |
| Commit do código (`git_sha`, com `-dirty` se havia alterações não commitadas) | Reproduzir o treino |
| Métricas de avaliação | Comparar treinos e versões |
| `plots/test_scores.png` | Gráfico do score ao longo do teste, com o limiar e as anomalias reais |
| `model_card.md` | Resumo do modelo: dados, configuração, métricas, uso previsto, limitações |
| `model/` | O modelo em formato padrão do MLflow (*pyfunc*) |

O modelo não depende do MLflow: o treino recebe uma função (`callback`) chamada a cada época, e é
o pipeline que envia esses valores ao MLflow.

### Formato do modelo no MLflow

O MLflow possui um formato genérico chamado *pyfunc*: qualquer modelo passa a ter um método
`predict(tabela)`. O modelo é registrado no modo *models from code*, em que o MLflow guarda um
pequeno script (`pyfunc_model.py`) em vez de um objeto serializado com `pickle`; os pesos vêm da
pasta segura da Fase 1. A API não usa o `predict` genérico: ela extrai o MSCVAE completo, que
também calcula as causas.

## Métricas de avaliação

| Métrica | O que mede |
|---|---|
| `precision` | Dos pontos marcados como anômalos, quantos realmente eram |
| `recall` | Dos pontos anômalos reais, quantos foram marcados |
| `f1` | Equilíbrio entre precision e recall, **no limiar usado em produção** |
| `pa_f1` | F1 com *point-adjust*: basta acertar um ponto de um trecho anômalo para o trecho inteiro contar como acerto |
| `auc_pr` | Qualidade do score independentemente do limiar |
| `best_f1` | F1 no melhor limiar possível, escolhido olhando o próprio teste (referência teórica) |
| `false_alarm_rate` | Fração dos pontos normais marcados como anômalos |
| `hitrate_100`, `hitrate_150` | Fração das variáveis causadoras reais que aparecem entre as primeiras do ranking de causas |
| `latency_ms` | Tempo para analisar 1.000 pontos |

O *point-adjust* é o protocolo usado na maioria dos artigos sobre SMD, mas infla os números: um
único acerto em um trecho de 50 pontos leva o F1 de cerca de 0,04 a 1,0. Por isso as duas versões
são reportadas, e a promoção usa o F1 sem ajuste.

`best_f1` e `auc_pr` ajudam a separar dois problemas: score ruim (o modelo não distingue
anomalias) ou limiar ruim (o modelo distingue, mas o corte está no lugar errado).

## Calibração do limiar com dados não vistos

O limiar é calculado a partir dos scores de dados normais. Se esses dados forem os mesmos usados
no treino, os erros são menores do que seriam em dados novos (a rede já os conhece), e o limiar
fica baixo demais. Com `calibration_fraction: 0.2`, os últimos 20% do período de treino não são
usados para ajustar a rede; servem para calcular o limiar e o erro típico de cada variável.

Efeito medido nos dados sintéticos:

| | Limiar | Falsos alarmes em dados normais novos | F1 no teste | Falsos alarmes no teste |
|---|---|---|---|---|
| Calibrado nos dados de treino | 74,5 | 2,40% | 0,390 | 5,59% |
| Calibrado em dados não vistos | 116,3 | 0,00% | 0,455 | 3,04% |

## Resultados no SMD (`machine-1-1`)

| Versão | F1 | PA-F1 | AUC-PR | F1 ótimo | Falsos alarmes | HitRate@100% | HitRate@150% |
|---|---|---|---|---|---|---|---|
| v1 (limiar calibrado no treino) | 0,382 | 0,408 | 0,623 | 0,615 | 30,3% | 0,511 | 0,688 |
| v2 (limiar calibrado em dados não vistos) | 0,369 | 0,377 | 0,635 | 0,617 | 34,5% | 0,605 | 0,784 |

Interpretação:

- O score separa bem as anomalias (AUC-PR de cerca de 0,63; com o melhor limiar possível, o F1
  chegaria a 0,62), e a análise de causa acerta de 60% a 78% das variáveis responsáveis.
- O limiar fixo gera muitos falsos alarmes: a segunda metade do SMD (usada como teste) tem um
  comportamento diferente da primeira (usada no treino). Um limiar fixo não acompanha essa
  mudança. O algoritmo SPOT original tem um modo que ajusta o limiar continuamente, mas exige
  guardar estado entre requisições, e a API foi projetada sem estado (ver Fase 4).
- A v2 melhorou a análise de causa, mas teve F1 menor e foi **rejeitada** pela regra de promoção
  (Fase 3).

## Correção no gerador sintético

Os parâmetros que definem o sistema simulado (médias e escalas) eram sorteados depois do ruído;
pedir uma série maior gerava um sistema diferente. Agora são sorteados antes, e uma série maior
estende a menor com os mesmos valores iniciais.

## Passo a passo para reproduzir

1. Treinar com dados sintéticos, registrar e promover:
   ```bash
   make train
   ```
   Resultado esperado no final: `version=1` e `promoted=True reason=no champion yet`.
2. Abrir a interface do MLflow:
   ```bash
   make mlflow        # http://localhost:5050
   ```
   No experimento `mscvae-synthetic`, abrir o run e conferir *Parameters*, *Metrics*
   (`train.loss` por época), *Artifacts* (`plots/test_scores.png`, `model_card.md`, `model/`) e a
   aba *Datasets*.
3. (Opcional) Treinar no SMD (download de ~30 MB; alguns minutos):
   ```bash
   make train-smd
   ```
4. (Opcional) Treinar com um CSV de exemplo no formato de dados próprios:
   ```bash
   make sample-csv                                    # cria datasets/exemplo.csv
   make train CONFIG=configs/train_csv_example.yaml   # modelo "mscvae-exemplo"
   ```
5. Executar os testes desta etapa:
   ```bash
   uv run pytest tests/unit/test_evaluate.py tests/unit/test_smd.py tests/unit/test_csv_source.py tests/integration -v
   ```
