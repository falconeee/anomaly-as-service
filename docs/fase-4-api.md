# Fase 4: API

A API recebe uma série temporal em JSON e devolve as anomalias encontradas, cada uma com
severidade, variáveis causadoras e uma explicação em texto. O modelo servido é sempre o
`@champion` do catálogo do MLflow, recarregado automaticamente quando muda.

## Arquivos principais

| Arquivo | Função |
|---|---|
| `src/anomaly_service/api/main.py` | Criação da aplicação, inicialização e recarga do modelo |
| `src/anomaly_service/api/routes.py` | Endpoints |
| `src/anomaly_service/api/deps.py` | Autenticação e acesso ao modelo/configurações |
| `src/anomaly_service/schemas.py` | Formato das requisições e respostas |
| `src/anomaly_service/inference/service.py` | Detecção (`InferenceService`) e gestão do modelo (`ModelManager`) |
| `src/anomaly_service/inference/events.py` | Agrupamento de janelas em eventos e severidade |
| `src/anomaly_service/inference/explain.py` | Causas e texto da explicação |

## Endpoints

| Método | Rota | Função |
|---|---|---|
| POST | `/v1/detect` | Detecção imediata, para séries de até `SYNC_MAX_POINTS` pontos (padrão 20.000) |
| POST | `/v1/jobs` | Detecção em segundo plano, para séries grandes (Fase 5) |
| GET | `/v1/jobs/{id}` | Situação e resultado de um trabalho em segundo plano |
| GET | `/v1/models/current` | Modelo em uso: versão, limiar, variáveis esperadas, janelas |
| POST | `/admin/reload` | Recarrega o modelo imediatamente |
| GET | `/health/live` | Indica se o processo está no ar |
| GET | `/health/ready` | Indica se há modelo carregado (e se a fila está acessível) |
| GET | `/docs` | Documentação interativa (Swagger) |

## Da série à resposta

1. **Validação.** A requisição é verificada: todas as séries com o mesmo tamanho, timestamps na
   mesma quantidade, opções dentro dos limites. Valores `null` são aceitos como dados faltantes.
2. **Colunas.** As variáveis são conferidas com as do modelo: colunas a mais são ignoradas
   (listadas em `ignored_features`) e a ordem não importa. Se faltar alguma, a resposta é **422**
   com `missing` e `extra`.
3. **Score.** O modelo calcula um score por janela.
4. **Eventos.** Janelas acima do limiar viram eventos. Trechos anômalos separados por até
   `max_gap` janelas normais são unidos em um único evento, para não gerar vários alertas para o
   mesmo incidente.
5. **Severidade.** Depende de quantas vezes o maior score do evento supera o limiar: menos de 1,5
   vez é `low`, menos de 3 vezes é `medium`, 3 vezes ou mais é `high`.
6. **Causas.** O desvio de cada variável é somado ao longo do evento. Uma variável é causa se
   desviou do próprio normal acima de `z_thresh` **e** se destacou em relação às demais. São
   devolvidas no máximo `top_k` causas.
7. **Tipo de causa.**
   - `magnitude`: o valor da variável saiu do comum;
   - `correlation`: o valor parece normal, mas a variável deixou de acompanhar as outras como
     costumava.
8. **Observado e esperado.** Para cada causa, o valor recebido e o valor que o modelo considera
   normal, no instante em que essa variável mais desviou.
9. **Texto.** Um modelo de frase (português ou inglês) monta a explicação a partir desses números.
   Não há modelo de linguagem envolvido: a mesma entrada sempre gera o mesmo texto, sem custo
   adicional e sem risco de o texto citar causas que os números não sustentam.

### Aquecimento

Os primeiros `max(window_sizes) − 1` pontos não têm histórico suficiente (o modelo completa as
janelas repetindo o primeiro valor). Por padrão, eles não geram eventos; a resposta informa
quantos são em `warmup_points`. Para analisá-los mesmo assim: `"include_warmup": true`.

### Fim dos eventos

Cada janela olha para trás. Por isso, logo depois do fim de uma anomalia, as próximas janelas
ainda contêm pontos anômalos, e o fim do evento reportado tende a ser um pouco posterior ao fim
real.

## Recarga automática do modelo

A cada `MODEL_REFRESH_INTERVAL_S` segundos, a API consulta para qual versão o `@champion` aponta.
Se mudou, carrega a nova versão em segundo plano e passa a usá-la; requisições em andamento
terminam com a versão anterior. Também é possível forçar a recarga com `POST /admin/reload`.

## Inicialização sem modelo

Em uma instalação nova ainda não existe modelo promovido. Nesse caso a API sobe mesmo assim:
`/health/live` responde 200, `/health/ready` responde 503 e `/v1/detect` responde 503 com
instruções. Assim que um modelo é promovido, a recarga automática o carrega.

## Processamento pesado sem travar a API

O cálculo do modelo ocupa a CPU. Se fosse executado diretamente no laço principal da API
(assíncrona), nenhuma outra requisição seria atendida enquanto ele roda, nem mesmo as de saúde.
Por isso a detecção roda em uma thread separada (`run_in_threadpool`). Para séries grandes, a API
responde **413** e indica o uso de `/v1/jobs`.

## Autenticação

Enviada no cabeçalho `X-API-Key`. A comparação é feita em tempo constante, para que a chave não
possa ser descoberta medindo o tempo de resposta. Sem a variável `API_KEY` configurada, a
autenticação fica desligada (uso apenas em desenvolvimento local).

## Escolhas de projeto

- Os formatos de requisição e resposta ficam em `schemas.py`, fora da pasta `api/`, porque também
  são usados pelo worker e pela camada de inferência.
- Um processo de API por container: cada processo teria sua própria cópia do modelo em memória;
  para atender mais carga, sobem-se mais containers.
- Processamento sem estado entre requisições: cada requisição traz a série completa. Qualquer
  réplica da API atende qualquer requisição, e repetir uma requisição dá o mesmo resultado. Em
  contrapartida, o limiar não se adapta ao longo do tempo e o cliente precisa reenviar o histórico.

## Passo a passo para reproduzir

1. Treinar e promover um modelo (se ainda não houver):
   ```bash
   make train
   ```
2. Subir a API local (sem Docker):
   ```bash
   make serve            # http://localhost:8000 ; usar PORT=8001 se a porta estiver ocupada
   ```
3. Enviar a série de exemplo:
   ```bash
   curl -s -X POST localhost:8000/v1/detect -H 'Content-Type: application/json' \
     -d @examples/request_synthetic.json | python -m json.tool | head -60
   ```
   Resultado esperado: entre as anomalias, eventos que começam perto das linhas 314, 805 e 1368,
   com causas `f04`, `f01` e `f07` (as variáveis em que as anomalias foram injetadas). Podem
   aparecer também eventos curtos de severidade `low`.
4. Explorar a documentação interativa em http://localhost:8000/docs.
5. Executar os testes da API:
   ```bash
   uv run pytest tests/api tests/unit/test_inference.py -v
   ```
