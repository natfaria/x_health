# arquivo com funções para uso em modelos 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from typing import Tuple, Optional


##################################
# Preparo dos dados para XGBoost #
##################################
def preparar_dados(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42):
    """
    Prepara os dados para modelagem, convertendo variáveis categóricas e separando em conjuntos de treino e teste.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo as variáveis preditoras e a variável alvo.
    target : str
        Nome da variável alvo a ser prevista.
    test_size : float, opcional (default=0.2)
        Percentual dos dados a serem separados para o conjunto de teste (entre 0 e 1).
    random_state : int, opcional (default=42)
        Semente aleatória para garantir reprodutibilidade na divisão dos dados.

    Retorno:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        - X_train : DataFrame contendo as variáveis preditoras do conjunto de treino.
        - X_test : DataFrame contendo as variáveis preditoras do conjunto de teste.
        - y_train : Series contendo a variável alvo do conjunto de treino.
        - y_test : Series contendo a variável alvo do conjunto de teste.
    """
    df = df.copy()

    # Converte variáveis categóricas para numéricas
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Separa variáveis preditoras (X) e variável alvo (y)
    X = df.drop(columns=[target])
    y = df[target]

    # Divide os dados em treino e teste com estratificação
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)



##################################
# Feature Importance do XGBoost  #
##################################

def selecionar_pelo_xgboost(df:pd.DataFrame,
                            target: str = 'default',
                            top_n=10,
                            save_path= "../reports/figures/importancia_variaveis.png"
):
    """
    Treina um modelo XGBoost e seleciona as principais variáveis com base na importância.

    Parâmetros:
    -----------
    df : pd.DataFrame
        Dataframe contendo as variáveis de interesse.
    target : pd.Series
        Variável alvo.
    top_n : int, opcional (default=10)
        Número de variáveis mais importantes a serem selecionadas.
    save_path : str, opcional (default="importancia_variaveis.png")
        Caminho para salvar o gráfico automaticamente.

    Retorno:
    --------
    list
        Lista com as variáveis mais importantes.
    """
    
    X = df.drop(columns=[target], errors="ignore")
    y = df[target]

    # Divide dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)
    
    # Importância das features
    importance = pd.DataFrame({"Variável": X_train.columns, "Importância": model.feature_importances_})
    importance = importance.sort_values(by="Importância", ascending=False).head(top_n)
    
    # Plot da importância das variáveis
    plt.figure(figsize=(10,6))
    plt.barh(importance["Variável"], importance["Importância"], color="blue")
    plt.xlabel("Importância")
    plt.ylabel("Variáveis")
    plt.title("Importância das Variáveis - XGBoost")
    plt.gca().invert_yaxis()
    
    # Salvar o gráfico automaticamente
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Gráfico salvo em: {save_path}")
    
    #Exibir o grafico
    plt.show()
    
    
    
    return importance["Variável"].tolist()

#######################################
#       Avaliação das métricas        #
#######################################

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, log_loss

def avaliar_XGBoost(model, dtrain, y_train, dtest, y_test) -> pd.DataFrame:
    """
    Avalia o desempenho de um modelo XGBoost nos conjuntos de treino e teste.

    Parâmetros:
    -----------
    model : xgboost.Booster
        Modelo treinado do XGBoost.
    dtrain : xgboost.DMatrix
        Conjunto de dados de treino no formato DMatrix.
    y_train : np.ndarray
        Valores reais do conjunto de treino.
    dtest : xgboost.DMatrix
        Conjunto de dados de teste no formato DMatrix.
    y_test : np.ndarray
        Valores reais do conjunto de teste.

    Retorno:
    --------
    pd.DataFrame
        DataFrame contendo as métricas de desempenho para os conjuntos de treino e teste.
    """

    # Previsões para treino
    y_train_pred_prob = model.predict(dtrain)  # Probabilidades para treino
    y_train_pred = (y_train_pred_prob > 0.5).astype(int)  # Classificação binária para treino

    # Previsões para teste
    y_test_pred_prob = model.predict(dtest)  # Probabilidades para teste
    y_test_pred = (y_test_pred_prob > 0.5).astype(int)  # Classificação binária para teste

    # Cálculo das métricas para treino e teste
    metrics = {
        "Métrica": ["AUC-ROC", "Acurácia", "Precisão", "Recall", "F1-score", "Log Loss"],
        "Treino": [
            roc_auc_score(y_train, y_train_pred_prob),
            accuracy_score(y_train, y_train_pred),
            precision_score(y_train, y_train_pred),
            recall_score(y_train, y_train_pred),
            f1_score(y_train, y_train_pred),
            log_loss(y_train, y_train_pred_prob),
        ],
        "Teste": [
            roc_auc_score(y_test, y_test_pred_prob),
            accuracy_score(y_test, y_test_pred),
            precision_score(y_test, y_test_pred),
            recall_score(y_test, y_test_pred),
            f1_score(y_test, y_test_pred),
            log_loss(y_test, y_test_pred_prob),
        ],
    }

    df_metrics = pd.DataFrame(metrics)
    
    #print("\n### Desempenho do Modelo XGBoost ###")
    #print(df_metrics.to_string(index=False))

    return df_metrics



###############################
#   Plot Matriz de Confusão   #
###############################
def plot_matriz_confusao(y_true_train: np.ndarray, 
                         y_pred_train: np.ndarray, 
                         y_true_test: np.ndarray, 
                         y_pred_test: np.ndarray, 
                         size: Tuple[int, int] = (12, 5), 
                         cmap_train: str = "Purples",
                         cmap_test: str = "Oranges",
                         font_size: int = 12,
                         save_path: Optional[str] = None
) -> None:
    """
    Plota a matriz de confusão para os dados de treino e teste, exibindo valores absolutos e percentuais.

    Parâmetros:
    -----------
    y_true_train : np.ndarray
        Array contendo os valores reais do conjunto de treino.
    y_pred_train : np.ndarray
        Array contendo os valores previstos pelo modelo para o conjunto de treino.
    y_true_test : np.ndarray
        Array contendo os valores reais do conjunto de teste.
    y_pred_test : np.ndarray
        Array contendo os valores previstos pelo modelo para o conjunto de teste.
    size : Tuple[int, int], opcional (default=(12,5))
        Define o tamanho da figura a ser plotada.
    cmap_train : str, opcional (default="Purples")
        Paleta de cores usada para a matriz de confusão do treino.
    cmap_test : str, opcional (default="Oranges")
        Paleta de cores usada para a matriz de confusão do teste.
    font_size : int, opcional (default=12)
        Tamanho da fonte dos valores na matriz.
    save_path : str, opcional (default=None)
        Caminho para salvar a imagem gerada. Se None, a imagem não será salva.

    Retorno:
    --------
    None
        Exibe os gráficos das matrizes de confusão para os conjuntos de treino e teste.
    """
    fig, axes = plt.subplots(1, 2, figsize=size)

    # Função auxiliar para plotar matriz
    def plot_confusion_matrix(cm, ax, title, cmap):
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        sns.heatmap(cm, annot=False, fmt='d', cmap=cmap, cbar=False, ax=ax)
        
        # Adiciona valores absolutos e percentuais no centro das células
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j + 0.5, i + 0.5,
                        f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)",
                        ha="center", va="center", color="white", fontsize=font_size)
        
        ax.set_title(title, fontsize=font_size + 2)
        ax.set_xlabel("Previsto", fontsize=font_size)
        ax.set_ylabel("Real", fontsize=font_size)

    # Matriz de confusão para treino
    cm_train = confusion_matrix(y_true_train, y_pred_train)
    plot_confusion_matrix(cm_train, axes[0], "Matriz de Confusão - Treino", cmap_train)

    # Matriz de confusão para teste
    cm_test = confusion_matrix(y_true_test, y_pred_test)
    plot_confusion_matrix(cm_test, axes[1], "Matriz de Confusão - Teste", cmap_test)

    plt.tight_layout()

    # Salvar figura, se necessário
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figura salva em: {save_path}")

    plt.show()
