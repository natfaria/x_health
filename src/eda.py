import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union, Any, Optional, Tuple


#Percentual e taxas
def taxas(
        database: pd.DataFrame,
        predictor_variables: List[str],
        target_variable: str = 'y',
        cutoff: float = 0,
        is_order_by_representation: bool = False,
        is_order_by_target_representation: bool = False
):
#-> pd.io.formats.style.Styler:
    """
    Calculates odds rations for specified variables.
    
    Parameters
    ----------
    database: pd.DataFrame
        DataFrame containing the dataset
    predictor_variables: List[str]
        List of variables to consider maximum of 3).
    target_variable: str, optional
        Response variable, by defautl 'y'.
    cutoff: float, optional
        Minimum representation value for a class to be presented, by default 0.
    is_order_by_representation: bool = bool, optional
        Sort by overall representation, by default False
    is_order_by_target_representation: bool = bool, optional
        Sort by representation in terms of the response variable, by default False

    Returns
    -------
    pd.io.formats.style.Styler
        DataFrame with class representation and odds ratio relative to the response variable
    """

    n = len(predictor_variables)
    if n == 1: 
        din_agp = pd.DataFrame(pd.crosstab(database[predictor_variables[0]], database[target_variable])).reset_index()
    elif n == 2:
        din_agp = pd.DataFrame(pd.crosstab([database[predictor_variables[0]], database[predictor_variables[1]]], database[target_variable])).reset_index()
    elif n == 3:
        din_agp = pd.DataFrame(pd.crosstab([database[predictor_variables[0]], database[predictor_variables[1]], database[predictor_variables[2]]], database[target_variable])).reset_index()

    din_agp['odds'] = din_agp[1] / din_agp[0]
    din_agp['taxa'] = din_agp[1] / (din_agp[0] + din_agp[1])

    din_agp['representação_conceito'] = (din_agp[1] / din_agp[1].sum()).round(4)

    din_agp['representacao_total'] = (din_agp[0] + din_agp[1]) / (din_agp[0] + din_agp[1].sum()).round(4)

    if is_order_by_representation==True:
        din_agp = din_agp.sort_values(by=['representacao_total'], ascending=False)
    
    if is_order_by_target_representation==True:
        din_agp = din_agp.sort_values(by=['representacao_conceito'], ascending=False)

    #Ajuste do ponto de corte de representação
    din_agp = din_agp[din_agp['representacao_total'] >= cutoff]

    #Deixa a representação em forma percentual
    format_dict = {'representacao_total': '{:.2%}', 'representacao_conceito': '{:.2%}'}
    din_agp = din_agp.style.format(format_dict).hide()

    return din_agp


def gera_taxa_por_quantis(database, var_exploratoria, var_alvo, q = 10, precisao = 2):
    database[f'FXA_{var_exploratoria}_num'] = pd.qcut(database[var_exploratoria], q, labels=False, duplicates='drop')
    database[f'FXA_{var_exploratoria}_desc'] = pd.qcut(database[var_exploratoria], q, precision=precisao, duplicates='drop')
    database[f'FXA_{var_exploratoria}'] = database[f'FXA_{var_exploratoria}_num'].astype(str) + - + database[f'FXA_{var_exploratoria}_desc'].astype(str)
    
    return taxas(database, vars_preditivas[f'FXA_{var_exploratoria}'], var_alvo = var_alvo)



def get_feature_importances(database, target_variable, feature, discrete_features = 'auto'):
    if discrete_features != 'auto':
        discrete_features = [x in discrete_features for x in features]

    importances = mutual_info_classif(
        database[features],
        database[target_variable],
        discrete_features = discrete_features
    )

    return pd.Series(importances, features)


def plot_mi(
    database: pd.DataFrame,
    features,
    target_variable: str = 'y',
    figsize: Tuple = (10,6),
    n: int = 15,
    discrete_features = 'auto'
):
    fig, ax = plt.subplots(figsize = figsize)
    
    get_feature_importances(database, target_variable, features, discrete_features)\
        .sort_values(ascending=False).head(n).sort_values()\
        .plot.barh(ax=ax, color = "#0046c0", title = f'Informação mútua para conceito: {target_variable}')

    plt.tight_layout()