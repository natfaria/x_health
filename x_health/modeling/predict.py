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


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = RAW_DATA_DIR / "input_dados_random.json",
    model_path: Path = MODELS_DIR / "modelo_xgboost.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "default_predicao_2.csv",
    # -----------------------------------------
):
    
    
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    """
    Realiza inferência utilizando um modelo treinado XGBoost.
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
    
    logger.success("Inferência concluída com sucesso!")
    print(output)

    # -----------------------------------------


if __name__ == "__main__":
    app()
