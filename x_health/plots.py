from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from typing import List, Union, Any, Optional, Tuple

from loguru import logger
from tqdm import tqdm

from x_health.eda_utils import get_feature_importances

#from x_health.config import FIGURES_DIR, PROCESSED_DATA_DIR


# cores personalizadas 
COR_1 = "#b11a6c" #rosa
COR_2 = "#373d7a" #azul escuro
COR_3 = "#6e446b" #roxo escuro
COR_CINZA = "#999999"


#################################################
#               Plot Mi
#################################################
def plot_mi(
    database: pd.DataFrame,
    features,
    target_variable: str = 'y',
    figsize: tuple = (10,6),
    n: int = 15,
    discrete_features = 'auto',
    color: str = COR_1  # Cor padrão: rosa
):
     
     """
     Plota a Informação Mútua (Mutual Information) para as principais variáveis.

    Parâmetros:
    -----------
    database: pd.DataFrame
        DataFrame contendo as variáveis independentes e a variável alvo.
    features: list
        Lista de colunas que representam as variáveis independentes.
    target_variable: str
        Nome da variável alvo.
    figsize: tuple
        Tamanho da figura do gráfico.
    n: int
        Número máximo de variáveis a serem exibidas no gráfico.
    discrete_features: str ou list
        Define se as variáveis devem ser tratadas como discretas ou contínuas.
    color: str
        Define a cor da barra no gráfico. Padrão: COR_1 (rosa).
    """
     fig, ax = plt.subplots(figsize = figsize)
     get_feature_importances(database, target_variable, features, discrete_features)\
        .sort_values(ascending=False).head(n).sort_values()\
        .plot.barh(ax=ax, color = color, title = f'Informação mútua para conceito: {target_variable}')

     plt.tight_layout()
    
    
    
    
    
#################################################
#            Plot Correlation Heatmap           #
#################################################
def correlation_heatmap(
    database: pd.DataFrame, method: str = 'pearson',
    numeric_only: bool = True,
    figsize: Tuple[int,int] = (10,10),
    title: str = "Mapa de Calor de Correlação",
        
    cores: List[str] = ["#ff69b4", "#e31c79", "#800040"]
) -> None:
        
    """
    Plots a heatmap of the correlation matrix for the dataframe
        
    Parameters
    ----------
    database: DataFrame
        DataFrame containing the dataset.
    method: Optional, str
        Method used to compute the correlation, 'pearson' by default.
    numeric_only: Optional, bool
        Whether to include only numeric columns in the correlation computation, True by default.
    figsize: Optional, Tuple [int,int]
        Size of the figure(width, height) in inches, (10,10) by default.

    Returns
    -------
    None
    """

    ## cria range de cores pela paleta enviada
    paleta_cores = LinearSegmentedColormap.from_list("CustomCores", cores, N=256)
        
    correlations = database.corr(method=method, numeric_only=numeric_only)
    mask = np.zeros_like(correlations)
    mask[np.triu_indices_from(mask)] = True
    

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        correlations,
        mask=mask,
        vmin=-1,  # Ajusta a escala para capturar correlações negativas corretamente
        vmax=1,   # Garante que a cor rosa será aplicada corretamente para os valores positivos
        center=0,
        fmt= '.2f',
        cmap=paleta_cores,
        square=True,
        linewidths=.5,
        annot=True,
        cbar_kws={'shrink':.70},
        annot_kws={'fontsize': 8, "fontweight": 'demibold'}
    )
    plt.title(title)
    plt.show
    
    
    
    
    
#################################################
#         Plot DISTRIBUIÇÃO BIVARIADA           #
#################################################
"""
Plota a distribuição bivariada entre uma variável de análise e um conceito 
(alvo), permitindo visualizar padrões e tendências de inadimplência.

Este gráfico combina **barras para volume de eventos** e **linha para a taxa de eventos**,
permitindo a identificação de padrões de risco e comportamento.

Parâmetros
----------
df : pd.DataFrame
    DataFrame contendo os dados a serem analisados.
variavel_analise : str
    Nome da variável independente a ser analisada.
nom_variavel_conceito : str
    Nome da variável alvo (conceito) que será analisada em relação à variável independente.
taxa_evento : float
    Taxa geral do evento de interesse (ex: taxa média de inadimplência).
convert_str : bool, opcional (default=True)
    Se True, converte a variável de análise para string.
sort_values : bool, opcional (default=False)
    Se True, ordena os valores de forma decrescente pelo volume.
figsize : Tuple[int, int], opcional (default=(16,4))
    Tamanho da figura do gráfico.
bar_width : float, opcional (default=0.3)
    Largura das barras do gráfico.
y_labelsize : int, opcional (default=10)
    Tamanho da fonte do eixo Y.
yticks_labelsize : int, opcional (default=10)
    Tamanho da fonte dos rótulos do eixo Y.
show_line_labels : bool, opcional (default=True)
    Se True, exibe rótulos na linha da taxa do evento.
show_bar_labels : bool, opcional (default=False)
    Se True, exibe rótulos sobre as barras do gráfico.
label_fontsize : int, opcional (default=9)
    Tamanho da fonte dos rótulos.
rotation : int, opcional (default=45)
    Rotação dos rótulos do eixo X.
ha : str, opcional (default='right')
    Alinhamento dos rótulos do eixo X.
is_category_axes : bool, opcional (default=False)
    Se True, trata o eixo X como categórico.
custom_xticks : list ou None, opcional (default=None)
    Permite definir rótulos personalizados para o eixo X.
ncol : int, opcional (default=4)
    Número de colunas na legenda do gráfico.
bbox_to_anchor : Tuple[float, float, float, float], opcional (default=(0, 1.08, 1., .1))
    Posição da legenda no gráfico.
leg_fontsize : int, opcional (default=10)
    Tamanho da fonte da legenda.
title_fontsize : int, opcional (default=14)
    Tamanho da fonte do título.
title_pad : int, opcional (default=40)
    Espaçamento entre o título e o gráfico.
xticks_labelsize : int, opcional (default=10)
    Tamanho da fonte dos rótulos do eixo X.
title : str, opcional (default=None)
    Título do gráfico. Se None, usa um título padrão.
label1 : str, opcional (default="Inadimplências")
    Rótulo para eventos positivos.
label2 : str, opcional (default="Não Inadimplências")
    Rótulo para eventos negativos.

Retorno
-------
None
    A função exibe um gráfico bivariado, sem retornar valores.

Exemplo de Uso
--------------
plot_distribuicao_bivariada(
    df=dados, 
    variavel_analise='tipo_sociedade', 
    nom_variavel_conceito='inadimplencia', 
    taxa_evento=0.15
)
"""

def plot_distribuicao_bivariada(
    df:pd.DataFrame,
    variavel_analise: str,
    nom_variavel_conceito: str,
    taxa_evento: float,
    convert_str: Optional[bool] = True,
    sort_values: Optional[bool] = False,
    figsize: Optional[Tuple[int,int]] = (16,4),
    bar_width:Optional[float] = 0.3,
    y_labelsize: Optional[int]= 10,
    yticks_labelsize: Optional[int] = 10,
    show_line_labels: Optional[bool] = True,
    show_bar_labels: Optional[bool] = False,
    label_fontsize: Optional[int] = 9,
    rotation: Optional[int] = 45,
    ha: Optional[str]='right',
    is_category_axes: Optional[bool] = False,
    custom_xticks: Optional[Union[None, List[str]]] = None,
    ncol: Optional[int] = 4,
    bbox_to_anchor: Optional[Tuple[float, float, float, float]] = (0, 1.08, 1., .1),
    leg_fontsize: Optional[int] = 10,
    title_fontsize: Optional[int] = 14,
    title_pad: Optional[int] = 40,
    xticks_labelsize: Optional[int] = 10,
    title: str = None,
    label1: str="Inadimplentes",
    label2: str= 'Adimplentes'

) -> None:

    if not title:
        title:f'Variação da taxa de inadimplência para a variavel: {variavel_analise}'
        df_tmp = df[[variavel_analise, nom_variavel_conceito]]
        df_tmp['quantidade'] = 1

    if convert_str:
        df_tmp[variavel_analise] = df_tmp[variavel_analise].astype(str)

    df_tmp = pd.DataFrame(df_tmp.groupby(variavel_analise)[['quantidade', nom_variavel_conceito]].sum()).reset_index()

    if sort_values:
        df_tmp.sort_values(by=['quantidade'], ascending=False, inplace=True)
    else:
        df_tmp.sort_values(by=[variavel_analise], inplace=True)

    df_tmp['qtd_nao_evento'] = (df_tmp['quantidade'] - df_tmp[nom_variavel_conceito]).astype(int)
    df_tmp['taxa_evento'] = (df_tmp[nom_variavel_conceito] / df_tmp['quantidade'])
    df_tmp['perc_vendas'] = 100* df_tmp['quantidade'] / df_tmp.quantidade.sum() # percentual de vendas
    df_tmp[nom_variavel_conceito] = df_tmp[nom_variavel_conceito].astype(int)

    fig, ax1 = plt.subplots(figsize = figsize)

    #################################
    #          PLOTA VOLUMES        #
    #################################

    bar_width = bar_width
    x_loc = np.arange(len(df_tmp))

    # Ajuste das posições para colocar as barras lado a lado
    ax1.bar(x_loc - bar_width / 2, df_tmp[nom_variavel_conceito], width=bar_width, color=COR_1,
        linestyle='-', linewidth=4., alpha=0.6, label=label1)
    y1_b1_patch = mpatches.Patch(color=COR_1, label=label1, alpha=0.5)

    ax1.bar(x_loc + bar_width / 2, df_tmp['qtd_nao_evento'], width=bar_width, color=COR_2,
        linestyle='-', linewidth=4., alpha=0.6, label=label2)
    y1_b2_patch = mpatches.Patch(color=COR_2, label=label2, alpha=0.5)


    plt.ylabel('Quantidade de vendas', fontsize=y_labelsize)

    if show_bar_labels:
        for idx, bar in enumerate(ax1.patches):
            if idx< len(df_tmp):
                bar_value = bar.get_height()
                text_x = bar.get_x() + bar.get_width() / 2
                text_y = bar.get_y + bar_value
                ax1.text(text_x, text_y, bar_value, ha='center', va='bottom', color='black', fontsize=label_fontsize)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

    ax1.yaxis.grid(True, linestyle = '--', which='major', color=COR_CINZA, alpha=.25)
    plt.gcf().autofmt_xdate(rotation=rotation, ha=ha)

    margin=(1-bar_width) + bar_width/2
    ax1.set_xlim(-margin, len(df_tmp) - 1 + margin)
    ax1.tick_params(axis='x', labelsize=xticks_labelsize)

    if is_category_axes:
        plt.xticks(df_tmp[variavel_analise])
    if custom_xticks:
        plt.xticks(np.arrange(0,len(custom_xticks)).tolist(),custom_xticks)



    #################################
    #          PLOTA TAXAS          #
    #################################
    ax2 = ax1.twinx() #instantiate a second axes that shares the same x-axis
    y_values = round(100.00 * df_tmp['taxa_evento'], 2)

    ax2.plot(
        df_tmp[variavel_analise], round(100.00 * df_tmp['taxa_evento'], 2),
        linestyle='-', marker='.',
        linewidth=1, color=COR_3
        )

    y2_patch = mpatches.Patch(color=COR_3, label="Taxa")

    plt.ylabel("Taxa do evento[%]", fontsize=y_labelsize)
    ax2.set_ylim([0, round(y_values.max()*1.1, 2)])

    if show_line_labels:
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        #props = dict(boxstyle=round(y_values.max()*1.1, 2))
        for x, y in enumerate(y_values):
            ax2.text(x, y + 0.1, f'{round(y, 2)}%',
                     ha='center', va = 'bottom',
                     color=COR_3, weight='bold', bbox=props)


    #################################
    #       PLOTA TAXA BASE         #
    #################################
    taxa_base_fmt = round(100 * taxa_evento, 2)
    plt.axhline(taxa_base_fmt, color=COR_CINZA, linestyle = '--', alpha=0.7)
    base_patch = mpatches.Patch(color=COR_CINZA, label=f'Taxa base: {taxa_base_fmt}%')

    plt.legend(
        loc='upper center',
        ncol=ncol,
        bbox_to_anchor = bbox_to_anchor,
        borderaxespad = 0.5,
        frameon = False,
        handles = [base_patch, y2_patch, y1_b1_patch, y1_b2_patch],
        fontsize = leg_fontsize
    )

    plt.title(
        title,
        fontsize = title_fontsize,
        weight = 'medium',
        pad = title_pad,
    )

    sns.despine(right=False)
    plt.show() 
