# X Health

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Projeto contendo Análise exploratória de dados e modelo para o problema do X-Health

| Criado por Natália de Faria
| Jan-Fev/2025

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         x_health and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── x_health   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes x_health a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── eda_utils.py            <- Scripts to help with EDA
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    ├── plots.py                <- Code to create visualizations
    │
    └── xgboost_utils           <- Script to help with data preparation and metrics for XGBoost
```

--------

- Dicionário de dados:

| nome_coluna                    | desc                                                                                               |
| --------------------------     |----------------------------------------------------------------------------------------- |
| default\_3months               |Quantidade de default nos últimos 3 meses                                                          |
| ioi\_Xmonths                   |Intervalo médio entre pedidos (em dias) nos últimos X meses                                       |
| valor\_por\_vencer             |Total em pagamentos a vencer do cliente B2B, em Reais     |
| valor\_vencido                 |Total em pagamentos vencidos do cliente B2B, em Reais                                              |
| valor\_quitado                 |Total (em Reais) pago no histórico de compras do cliente B2B                |
| quant\_protestos               |Quantidade de protestos de títulos de pagamento apresentados no Serasa|
| valor\_protestos               |Valor total (em Reais) dos protestos de títulos de pagamento apresentados no Serasa|
| quant\_acao_judicial           |Quantidade de ações judiciais apresentadas pelo Serasa|
| acao\_judicial\_valor          |Valor total das ações judiciais (Serasa) |
| participacao\_falencia\_valor  |Valor total (em Reais) de falências apresentadas pelo Serasa |
| dividas\_vencidas\_valor       |Valor total de dívidas vencidas (Serasa)|
| dividas\_vencidas\_qtd         |Quantidade total de dívidas vencidas (Serasa)|
| falencia\_concordata\_qtd      |Quantidade de concordatas (Serasa)|
| tipo\_sociedade                |Tipo de sociedade do cliente B2B |
| opcao\_tributaria              |Opção tributária do cliente B2B |
| atividade\_principal           |Atividade principal do cliente B2B|
| forma\_pagamento               |Forma de pagamento combinada para o pedido |
| valor\_total\_pedido           |Valor total (em Reais) do pedido em questão|
| month                          |Mês do pedido|
| year                           |Ano do pedido|
| default                        |Status do pedido: default = 0 (pago em dia), default = 1 (pagamento não-realizado, calote concretizado)|

