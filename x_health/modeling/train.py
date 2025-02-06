#adicionando retorno no diretório ao caminho
import sys
sys.path.append('../')

#ignorar warnings 
import warnings
warnings.filterwarnings('ignore')
#informação de diretórios
from x_health.config import *
# arquivo auxiliar
from x_health.xgboost_utils import *

from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
        
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import pickle


app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = EXTERNAL_DATA_DIR / "dataset_2021-5-26-10-14.csv.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "modelo_xgboost.pkl"
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    
    #importando a base do arquivo externo
    df = pd.read_csv(features_path, sep = '\t', encoding='utf-8', na_values="missing")
    backup = df.copy()
    #logger.info(f'Base importada com tamanho: {len(df)}')

    ############################
    #      TRANSFORMAÇÕES      #
    ############################
    ### criar flag_valor_vencido
    df["flag_valor_vencido"] = (df["valor_vencido"] > 0).astype(int)
    ## criar agrupamento da forma de pagamento
    df['forma_pagamento_agrup'] = agrupar_prazo(df, 'forma_pagamento')
    #Cria separação por trimestres
    df["periodo_fiscal"] = df["month"].apply(lambda x: "1T" if x in [1, 2, 3] else 
                                            "2T" if x in [4, 5, 6] else 
                                            "3T" if x in [7, 8, 9] else "4T")
    # Criar razão entre valor vencido e valor total pago(+1 para evitar divisão por 0)
    df["razao_valor_vencido"] = df["valor_vencido"] / (df["valor_quitado"] + 1)
    # Criar histórico de pagamento como proporção de valores pagos em relação ao vencido
    df["historico_pagamento"] = df["valor_quitado"] / (df["valor_quitado"] + df["valor_vencido"] + 1)
    
    #PREENCHIMENTO DE NAN E TRATAMENTO
    df = tratar_categoricas(df)
    
    #################################
    #      SELEÇÃO DE FEATURES      #
    #################################
    colunas = [ # lista de colunas a serem utilizadas    
    'flag_valor_vencido',
    'quant_protestos',
    'default_3months',
    'opcao_tributaria',
    'razao_valor_vencido',
    'forma_pagamento_agrup',
    'periodo_fiscal',
    'ioi_3months',
    'historico_pagamento',
]
    # Calcular scale_pos_weight
    contagem_classes = np.bincount(y_train)  # Conta os valores 0 e 1 no y_train
    scale_pos_weight = contagem_classes[0] / contagem_classes[1]
    print(f"scale_pos_weight sugerido: {scale_pos_weight:.2f}")
    
    #################################
    #           MODELO FINAL        #
    #################################
    ## preparar variaveis e separar em teste e treino
    X_train, X_test, y_train, y_test = preparar_dados(df[colunas + [var_alvo]], target=var_alvo)
    # Criando os DMatrix para XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Configuração do modelo
    params = {
    'scale_pos_weight': 4.375905217516745, 
    'max_depth': 6, 
    'learning_rate': 0.2727069825106735, 
    'n_estimators': 120, 
    'lambda': 8.087940526870096, 
    'alpha': 1.3597159615097383, 
    'min_child_weight': 8, 
    'gamma': 0.7339255157763341
    }

    # Treinando o modelo
    xgb_optimized = xgb.train(best_params, dtrain, num_boost_round=100)
    metrica_final = avaliar_XGBoost(xgb_optimized, dtrain, y_train, dtest, y_test)
    
    
    #################################
    #       SALVAR PICKLE           #
    #################################
    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    print(f"Modelo salvo em: {model_path}")

    # -----------------------------------------


if __name__ == "__main__":
    app()
