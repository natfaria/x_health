from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm


import json
import random
from x_health.config import *

app = typer.Typer()


def gerar_dados_teste():
    """
    Gera um dicionário com valores aleatórios para teste e salva em um arquivo JSON.
    """
    # Definição dos agrupamentos de forma de pagamento
    forma_pagamento_opcoes = [
        "Sem pagamento",
        "Desconhecido",
        "À vista (até 15 dias)",
        "Curto prazo (16-30 dias)",
        "Médio prazo (31-90 dias)",
        "Longo prazo (+90 dias)",
        "Outros"
    ]
    
    # Definição dos períodos fiscais por trimestre
    periodo_fiscal_opcoes = ["1T", "2T", "3T", "4T"]
    
    # Definição dos tipos de opção tributária
    opcao_tributaria_opcoes = [
        "Simples Nacional",
        "Desconhecido",
        "Lucro Real",
        "Lucro Presumido",
        "Isento"
    ]
    
    # Geração de valores aleatórios para os dados de teste
    valor_vencido = random.uniform(0, 100000)
    valor_quitado = random.uniform(0, 100000)
    
    dados_teste = {
        "flag_valor_vencido": random.choice([0, 1]),
        "quant_protestos": random.randint(0, 10),
        "default_3months": random.choice([0, 1]),
        "opcao_tributaria": random.choice(opcao_tributaria_opcoes),
        "razao_valor_vencido": round(valor_vencido / (valor_quitado + 1), 2),
        "forma_pagamento_agrup": random.choice(forma_pagamento_opcoes),
        "periodo_fiscal": random.choice(periodo_fiscal_opcoes),
        "ioi_3months": random.randint(0, 10),
        "historico_pagamento": round(valor_quitado / (valor_quitado + valor_vencido + 1), 2)
    }
    
    # Salvar em arquivo JSON
    
    
    caminho_arquivo_entrada = f"{RAW_DATA_DIR}\input_dados_random.json"
    
    with open(caminho_arquivo_entrada, "w") as file:
        json.dump(dados_teste, file, indent=4)
    
    print(f"Arquivo salvo com sucesso em {caminho_arquivo_entrada}!")

if __name__ == "__main__":
    gerar_dados_teste()