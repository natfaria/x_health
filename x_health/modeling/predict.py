from pathlib import Path
import json
import pickle
import pandas as pd
import typer
from loguru import logger

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

#informação de diretórios
from x_health.config import *

app = typer.Typer()

features_path: Path = RAW_DATA_DIR / "input_dados_random.json",
model_path: Path = MODELS_DIR / "modelo_xgboost.pkl",
predictions_path: Path = PROCESSED_DATA_DIR / "default_predicao_2.csv",

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: features_path,
    model_path: model_path / "modelo_xgboost.pkl",
    predictions_path: predictions_path / "default_predicao_2.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    prever_default()

    # -----------------------------------------

def prever_default(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: features_path,
    model_path: model_path / "modelo_xgboost.pkl",
    predictions_path: predictions_path / "default_predicao_2.csv",
    # -----------------------------------------
):
    """
    Realiza inferência utilizando um modelo treinado XGBoost para prever inadimplência.

    Parâmetros:
    -----------
    features_path : Path
        Caminho para o arquivo JSON contendo os dados de entrada para a predição.

    model_path : Path
        Caminho para o arquivo pickle (.pkl) que contém o modelo XGBoost treinado.

    predictions_path : Path
        Caminho onde a predição será salva no formato JSON.

    Funcionamento:
    --------------
    1. Carrega o modelo treinado do caminho especificado.
    2. Lê os dados de entrada a partir do JSON em `features_path`.
    3. Converte variáveis categóricas em numéricas para compatibilidade com o modelo.
    4. Cria uma matriz `DMatrix` para o XGBoost.
    5. Faz a predição com o modelo carregado.
    6. Determina se a previsão indica default (inadimplência) ou não.
    7. Salva a predição em um arquivo JSON no caminho especificado.

    Retorno:
    --------
    Nenhum retorno explícito, mas exibe o resultado da predição no console e salva em JSON.

    Exemplo de saída no arquivo JSON:
    {
        "default": 1
    }
    """
    logger.info("Carregando modelo...")
    with open(model_path, "rb") as file:
        modelo = pickle.load(file)
    
    logger.info("Carregando dados de entrada...")
    with open(features_path, "r") as file:
        input_data = json.load(file)
    
    df = pd.DataFrame([input_data])
    
    
    # Converte variáveis categóricas para numéricas
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Salva os encoders para referência futura, se necessário
    
    dmatrix = xgb.DMatrix(df)
    
    logger.info("Realizando predição...")
    predicao = modelo.predict(dmatrix)
    resultado = int(predicao[0] > 0.5)
    
    output = {"default": resultado}
    
    logger.info("Salvando predição...")
    with open(predictions_path, "w") as file:
        json.dump(output, file, indent=4)
    
    logger.success(f"Predição salva com sucesso em {predictions_path}")
    print(output)

    # -----------------------------------------



if __name__ == "__main__":
    app()
