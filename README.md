# X Health

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

**X Health** é um projeto de análise exploratória de dados e modelagem preditiva para prever **risco de inadimplência** em transações B2B. Ele utiliza técnicas de Machine Learning para identificar padrões nos comportamentos de pagamento dos clientes.

| 📌 Criado por: Natália de Faria  
| 📅 Período: Jan-Fev/2025  

---

## 🚀 Objetivo do Projeto
O principal objetivo deste projeto é **prever o risco de inadimplência** de clientes B2B usando dados históricos de transações.  
Para isso, foram explorados **modelos estatísticos e de aprendizado de máquina**, principalmente utilizando **XGBoost**.

A base de dados inclui informações sobre:
- Pagamentos vencidos e quitados,
- Protestos financeiros e ações judiciais,
- Histórico de compras,
- Opção tributária e tipo de sociedade do cliente.

Com esses dados, o modelo é treinado para prever se um novo pedido terá **default = 1 (inadimplência)** ou **default = 0 (pagamento em dia)**.

---

## 📂 Estrutura do Projeto
Utilizando o Framework CookieCutter for DataScience, tem-se o seguinte esquema de arquivos e diretórios

```
├── LICENSE            <- Licença open-source (se aplicável).
├── Makefile           <- Comandos de automação (e.g., `make data`, `make train`).
├── README.md          <- Este arquivo com a documentação do projeto.
│
├── data
│   ├── external       <- Dados de terceiros.
│   ├── interim        <- Dados intermediários transformados.
│   ├── processed      <- Dados finais e prontos para modelagem.
│   └── raw            <- Dados brutos e imutáveis.
│
├── docs               <- Documentação do projeto.
│
├── models             <- Modelos treinados e serializados (pickle).
│
├── notebooks          <- Notebooks Jupyter usados para experimentação.
│
├── pyproject.toml     <- Arquivo de configuração do projeto.
│
├── references         <- Dicionário de dados e materiais explicativos.
│
├── reports            <- Relatórios gerados (HTML, PDF, etc.).
│   └── figures        <- Gráficos e visualizações geradas.
│
├── requirements.txt   <- Arquivo de dependências (bibliotecas necessárias).
│
├── setup.cfg          <- Configuração de estilo e linting (flake8).
│
└── x_health   <- Código-fonte do projeto.
    │
    ├── __init__.py             <- Torna `x_health` um módulo Python.
    │
    ├── config.py               <- Configurações do projeto.
    │
    ├── dataset.py              <- Script para manipulação de dados.
    │
    ├── eda_utils.py            <- Funções auxiliares para Análise Exploratória.
    │
    ├── features.py             <- Criação de features para modelagem.
    │
    ├── modeling                
    │   ├── __init__.py
    │   ├── predict.py          <- Script para inferência com modelos treinados.
    │   └── train.py            <- Script para treinamento de modelos.
    │
    ├── plots.py                <- Funções para geração de visualizações.
    │
    └── xgboost_utils.py        <- Funções auxiliares para XGBoost.
```

--------

---

## 📦 Instalação e Configuração

1️⃣ **Clone este repositório**  
```
git clone https://github.com/seu-usuario/x_health.git
cd x_health
```

2️⃣ **Crie um ambiente virtual e ative-o**
```
python -m venv venv
source venv/bin/activate  # No macOS/Linux
venv\Scripts\activate     # No Windows
```
3️⃣ **Instale as dependências**
```
pip install -r requirements.txt
```
▶️ Como Usar
1. Treinar o Modelo
Para treinar um modelo a partir dos dados processados, rode o comando:

```
python x_health/modeling/train.py
```
2. Fazer Previsões
O script predict.py carrega um modelo treinado e faz previsões com base nos dados fornecidos.

Exemplo de Uso
```
import json
from x_health.modeling.predict import prever_default

dados_teste = {
    "flag_valor_vencido": 1,
    "quant_protestos": 3,
    "default_3months": 0,
    "opcao_tributaria": "Simples Nacional",
    "razao_valor_vencido": 2.5,
    "forma_pagamento_agrup": "Curto prazo (16-30 dias)",
    "periodo_fiscal": "2T",
    "ioi_3months": 5,
    "historico_pagamento": 0.8
}

# Rodar a previsão
```
resultado = prever_default(dados_teste)
```

# Exibir a saída
```
print(json.dumps(resultado, indent=4))
```
Saída esperada:

```
{
    "flag_valor_vencido": 1,
    "quant_protestos": 3,
    "default_3months": 0,
    "opcao_tributaria": "Simples Nacional",
    "razao_valor_vencido": 2.5,
    "forma_pagamento_agrup": "Curto prazo (16-30 dias)",
    "periodo_fiscal": "2T",
    "ioi_3months": 5,
    "historico_pagamento": 0.8,
    "default": 1
}
```
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

