# Fase 1: Modelo pronto para produção

O código original do MSCVAE foi escrito para pesquisa: treinava e avaliava, mas não salvava o
modelo, não validava as colunas de entrada e tinha um laço que podia não terminar. Esta etapa
torna o modelo seguro para ser salvo, carregado por outro processo e usado por uma API.

## Arquivos principais

| Arquivo | Função |
|---|---|
| `src/anomaly_service/model/mscvae.py` | O modelo (com as correções desta etapa) |
| `src/anomaly_service/model/persistence.py` | Salvar e carregar o modelo (`save_model`, `load_model`) |
| `src/anomaly_service/data/synthetic.py` | Gerador de dados sintéticos com anomalias conhecidas |
| `tests/unit/test_mscvae.py`, `test_persistence.py` | Testes do modelo |

## Salvar e carregar o modelo com segurança

O modelo é salvo como uma **pasta** com três arquivos:

| Arquivo | Conteúdo |
|---|---|
| `weights.pt` | Pesos da rede neural (apenas números, sempre salvos para CPU) |
| `metadata.json` | Configuração, nomes das variáveis, médias e desvios usados na normalização, limiar, versões das bibliotecas |
| `baselines.npz` | Erro típico de cada variável, usado para explicar as anomalias |

```python
model.save("artifacts/meu_modelo")             # salva
model = MSCVAE.load("artifacts/meu_modelo")    # carrega (em CPU por padrão)
```

### Por que não usar `pickle`

O formato `pickle`, padrão do Python para salvar objetos, guarda instruções que são **executadas**
ao abrir o arquivo. Um arquivo adulterado pode executar qualquer código. Como o modelo é produzido
por um processo (o treino) e aberto por outro (a API), o projeto evita esse risco:

- os pesos são lidos com `torch.load(..., weights_only=True)`, que recusa qualquer coisa além de
  números;
- o restante é JSON e NPZ (lido com `allow_pickle=False`).

O campo `format_version` do `metadata.json` permite evoluir o formato: uma pasta em formato
desconhecido é recusada com uma mensagem clara.

## Validação das colunas de entrada

O modelo guarda os nomes das variáveis usadas no treino (`model.feature_names`). Em qualquer
predição:

- as colunas são **reordenadas pelo nome** para a ordem do treino;
- colunas a mais são ignoradas;
- se faltar alguma variável, o erro `FeatureSchemaError` informa quais faltam (`missing`) e quais
  sobram (`extra`).

Essa verificação corrigiu um problema silencioso. O pandas alinha operações pelo **nome** da
coluna, mas a rede neural lê os valores pela **posição**. Com as colunas em outra ordem, uma
variável era lida como se fosse outra, sem nenhum erro:

```python
media = pd.Series([...], index=["b", "a"])   # ordem do treino: b, a
df = pd.DataFrame(..., columns=["a", "b"])   # dados chegam em outra ordem
(df - media).values                          # pandas reordena para (a, b); a rede espera (b, a)
```

Em uma API que recebe JSON, onde a ordem das chaves não é garantida, isso aconteceria com
frequência.

## Reprodutibilidade

- `set_deterministic` fixa as sementes aleatórias do Python, NumPy e PyTorch. O treino (`fit`)
  refaz isso ao começar, então dois treinos com a mesma configuração e os mesmos dados produzem o
  mesmo modelo.
- Em CPU, os scores de dois treinos idênticos coincidem (o teste usa tolerância de `1e-6`).
- Um modelo treinado na GPU do Mac (MPS) e carregado em CPU produz scores com diferença relativa
  de cerca de `3e-7`, efeito normal de arredondamento entre dispositivos.

## Cálculo do limiar sem laço infinito

O limiar usa o algoritmo SPOT (Teoria de Valores Extremos):

1. separa os scores normais mais altos (acima do percentil 98), chamados de "cauda";
2. ajusta uma distribuição estatística (Pareto generalizada) a essa cauda;
3. escolhe o valor que dados normais só ultrapassariam com probabilidade `q = 0,0001`.

Quando não há cauda (scores todos iguais ou poucos dados), o ajuste falha. O código original
repetia a tentativa indefinidamente. Agora há no máximo 500 tentativas; depois disso, o limiar
passa a ser o percentil empírico `1 − q` dos scores, e um aviso é registrado no log.

## Score e explicação em uma única passada

`model.score_windows(df)` executa a rede uma vez e devolve, para cada janela:

| Campo | Significado |
|---|---|
| `phi` | Score de anomalia (comparável com `model.threshold`) |
| `s_diag` | Desvio de cada variável considerada **isoladamente** |
| `s_off` | Desvio nas **relações** de cada variável com as demais |
| `s_val` | Desvio na reconstrução do **valor bruto** de cada variável |
| `recon_z` | Valor que o modelo considera normal (normalizado) |
| `row_index` | Linha da entrada em que cada janela termina |

Os desvios são medidos em relação ao erro típico **da própria variável** (mediana e MAD calculados
em dados normais) e valem zero quando o erro está dentro do normal. A API (Fase 4) usa esses
canais para dizer se a causa é de **magnitude** (`s_diag`, `s_val`) ou de **correlação**
(`s_off`).

## Dados sintéticos para testes

O gerador cria variáveis correlacionadas (combinações de ondas senoidais com ruído) e injeta
anomalias em posições conhecidas, o que permite verificar se o modelo detecta e explica
corretamente:

| Tipo | O que acontece | Como aparece no modelo |
|---|---|---|
| `spike` | A variável salta alguns desvios-padrão por poucos pontos | Erro no valor bruto |
| `drift` | A variável se afasta gradualmente | Erro no valor bruto, crescendo devagar |
| `correlation_break` | A variável passa a se mover ao contrário das outras, dentro da faixa normal | Erro nas relações; invisível para alertas por variável |

## Como o código de ML é testado

Não há um "valor certo" fixo para comparar o score. Os testes verificam propriedades:

| Propriedade | Exemplo de teste |
|---|---|
| Reprodutibilidade | Dois treinos com a mesma semente dão o mesmo score |
| Invariância | Mudar a ordem das colunas ou acrescentar colunas não altera o resultado |
| Ida e volta | Salvar e carregar mantém scores, limiar e explicações |
| Casos de borda | Série curta, valores faltantes, scores constantes |
| Comportamento | Um spike injetado é detectado e atribuído à variável certa |

O treino é a parte lenta, então `tests/conftest.py` treina um modelo pequeno **uma única vez** e
o compartilha entre os testes.

## Correções feitas no código original

- `np.round(X, decimals=5)` não guardava o resultado (não tinha efeito); agora o arredondamento é
  aplicado.
- A loss exibida durante o treino era dividida pelo número de amostras, embora já fosse uma média
  por lote, o que a mostrava cerca de 128 vezes menor; agora é a média por lote.
- `predict` escondia erros de entrada e devolvia um resultado vazio; agora o erro é informado.
- Sem timestamps, `predict` devolvia o número da janela; agora devolve a linha da entrada.
- Uma chamada a `gc.collect()` em cada análise de causa foi removida (custava tempo sem benefício).
- O arquivo passa pelas mesmas verificações (ruff e mypy) que o resto do projeto.

Sobre o formato de armazenamento: o registro padrão de modelos PyTorch no MLflow também usa
`pickle` e não guardaria o restante do estado (normalização, limiar, erros típicos); o formato
ONNX serviria apenas para a rede neural. Por isso foi criado o formato próprio em pasta.

## Passo a passo para reproduzir

1. Executar os testes do modelo:
   ```bash
   uv run pytest tests/unit/test_mscvae.py tests/unit/test_persistence.py -v
   ```
   Resultado esperado: todos os testes passando.
2. Treinar, salvar e carregar um modelo pequeno:
   ```bash
   uv run python -c "
   from anomaly_service.data.synthetic import make_dataset
   from anomaly_service.model import MSCVAE
   train, test = make_dataset()
   m = MSCVAE(window_sizes=(5, 10, 20), device='cpu'); m.fit(train, epochs=5)
   loaded = MSCVAE.load(m.save('artifacts/demo'))
   print(loaded.feature_names, loaded.threshold)
   "
   ```
   Resultado esperado: a lista `['f00', ..., 'f07']` e um valor numérico de limiar.
3. Inspecionar o modelo salvo:
   ```bash
   cat artifacts/demo/metadata.json
   ```
