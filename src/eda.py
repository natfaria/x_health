import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif
from typing import List, Union, Any, Optional, Tuple


#################################################
#                  Get WoE                      #
#################################################
def get_woe(base: pd.DataFrame, 
            woe: pd.DataFrame, 
            variavel_analise: str, 
            is_categorical: bool = False
):
    variable_filter = woe["Variável"] = variavel_analise
    woe_data = woe[variable_filter][['Corte', 'WoE']]

    if is_categorical:
        base = base.merge(woe_data, left_on=variavel_analise, right_on='Corte', how='left')
        base - base.drop(columns='Corte')
    else:
        base['Corte'] = pd.cut(
            base[variavel_analise].astype(float),
            bins = pd.IntervalIndex(woe_data['Corte']),
            labels = woe_data['Corte'].astype(str)
        )
        base = base.merge(woe_data, on='Corte', how='left')    

    base = base.rename(columns={"Corte": f'Corte_{variavel_analise}', 'WoE': f'Woe{variavel_analise}'}, errors='ignore')
    return base


#################################################
#                   IV / WOE                    #
#################################################
def iv_woe(
        data: pd.DataFrame, 
        target: str, 
        bins: int=10, 
        show_woe: bool = True,
        show_iv: bool = True
):
    """
    Parameters
    -----------
    data: DataFrame
        Dataframe where independent and dependent variables are stored
    target: str
        Target variable.
    bins: int
        Total of bins or intervals, 10 by default.
    show_woe: bool
        If the WOE values are shown in the table, True by default
    show_iv: bool
        if the IV values are shown in the table, True by default
    """   
    
    newDF, woeDF = pd.DataFrame(), pd.DataFrame()
    cols = data.columns

    #Run IV and WOE on all independent variables
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
            binned_x = pd.qcut(data[ivars], bins, duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
        
        d = d0.groupby('x', as_index=False).agg({'y': ['count', 'sum']})
        d.columns = ['Corte', 'N', 'Eventos']
        d['% de Eventos'] = np.maximum(d['Eventos'], 0.5) / d['Eventos'].sum()
        d['Nao-Eventos'] = d['N'] - d['Eventos']
        d['% de Nao-Eventos'] = np.maximum(d['Nao-Eventos'], 0.5) / d['Nao-Eventos'].sum()
        d['WoE'] = np.log(d['% de Eventos'] / d['% de Nao-Eventos'])
        d['IV'] = d['WoE'] * (d['% de Eventos'] - d['% de Nao-Eventos'])
        d.insert(loc=0, column='Variavel', value=ivars)

        if show_iv == True:
            print(f'\033[mInformation Value de \033[1m{ivars} \033[mé \033[1;34m{str(round(d['IV'].sum(),6))}')
        
        temp = pd.DataFrame({'Variavel' : [ivars], "IV": [d["IV"].sum()]}, columns = ['Variavel', 'IV'])
        newDF = pd.concat([newDF, temp], axis=0)
        woeDF = pd.concat([woeDF, d], axis=0)

        # Show WOE Table
        if show_woe == True:
            print(d)
    return newDF, woeDF




#################################################
#          MI - Mutual Information              #
#################################################

def calcular_mutual_information(df: pd.DataFrame, 
                                target: str = "default"
) -> pd.DataFrame:
    """
    Calcula a Informação Mútua (Mutual Information - MI) entre as variáveis preditoras e o target.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo variáveis de interesse.
    target : str, opcional (default="default")
        Nome da variável alvo.

    Retorno:
    --------
    pd.DataFrame
        DataFrame com os valores de MI das variáveis ordenadas por importância.
    """
    # Selecionar apenas variáveis numéricas e remover a variável alvo
    df_numerico = df.select_dtypes(include=['number']).drop(columns=[target], errors='ignore')

    # Tratamento de NaN: Preencher com a mediana de cada coluna
    df_numerico.fillna(df_numerico.median(), inplace=True)
    
     # Calcular MI
    mi_scores = mutual_info_classif(df_numerico, df[target])
    
    # Aplicar no DF
    mi_df = pd.DataFrame({"Variável": df_numerico.columns, "Mutual Information": mi_scores})
    mi_df = mi_df.sort_values(by="Mutual Information", ascending=False).reset_index(drop=True)

    return mi_df


#################################################
#          Razão de Probabilidades              #
#################################################
def odds(
        database: pd.DataFrame,
        predictor_variables: List[str],
        target_variable: str = 'y',
        cutoff: float = 0,
        is_order_by_representation: bool = False,
        is_order_by_target_representation: bool = False
):
#-> pd.io.formats.style.Styler:
    """
    Calculates odds ratios for specified variables.
    
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



#################################################
#             Gera Odds por quartis             #
#################################################
def generate_odds_by_quantis(database, exploratory_variable, target_variable, q = 10, precisao = 2):
    database[f'FXA_{exploratory_variable}_num'] = pd.qcut(database[exploratory_variable], q, labels=False, duplicates='drop')
    database[f'FXA_{exploratory_variable}_desc'] = pd.qcut(database[exploratory_variable], q, precision=precisao, duplicates='drop')
    database[f'FXA_{exploratory_variable}'] = database[f'FXA_{exploratory_variable}_num'].astype(str) + - + database[f'FXA_{exploratory_variable}_desc'].astype(str)
    
    return odds(database, predictor_variables=[f'FXA_{exploratory_variable}'], target_variable = target_variable)


#################################################
#               Feature Importance              #
#################################################
def get_feature_importances(database, target_variable, features, discrete_features = 'auto'):
    if discrete_features != 'auto':
        discrete_features = [x in discrete_features for x in features]

    importances = mutual_info_classif(
        database[features],
        database[target_variable],
        discrete_features = discrete_features
    )

    return pd.Series(importances, features)


