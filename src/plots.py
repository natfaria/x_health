from matplotlib.colors import Colormap
import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

import seaborn as sns
from typing import List, Union, Any, Optional, Tuple

from scipy import stats

from src.eda import *

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






#COR_1 = "#e91e63"  # Rosa vibrante
#COR_2 = "#3f51b5"  # Azul vibrante
#COR_3 = "#9c27b0"  # Roxo vibrante
#COR_CINZA = "#757575"  # Cinza para rótulos
#################################################
#         Plot Bivariate Distribution           #
#################################################

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
    label1: str="Inadimplências",
    label2: str= 'Não Inadimplências'

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


#################################################
#           Plot Distribution Graph             #
#################################################


def plot_grafico_distribuicao(
        database: pd.DataFrame,
        exploratory_variable: str,
        target_variable: str = 'y',
        label1: str = "Inadimplencia",
        label2: str = "Não Inadimplencia",
        figsize: Tuple[int, int] = (6,6), 
        title: str="Distribuição"
) -> None:
    """
    Plots a distibution chart for a variable based on the response variable

    Parameters
    database: pd.DataFrame
        DataFrame containing the dataset.
    exploratory_variable: str
        Variable to plot the distribution for.
    target_variable: str
        Response Variable.
    label1: str
        Label for the first distribution.
    label2: str
        Label for the second distribution.
    
    returns
    ---------
    None
    """
    fig, ax = plt.sublot(figsize=figsize)

    ax1 = sns.distplot(database.query(f'{target_variable} == 1')[exploratory_variable], label=label1, color = "#ff5a40")
    ax2 = sns.distplot(database.query(f'{target_variable} == 0')[exploratory_variable], label=label2, color = "#00a1fc")

    plt.title(title)
    ax1.legend()
    ax2.legend()
    plt.show()

def plot_mi(
    database: pd.DataFrame,
    features,
    target_variable: str='y',
    figsize: Tuple[int,int] = (10,6),
    n: int=15,
    discrete_features: str = 'auto'
):
    fig, ax = plt.subplots(figsize=figsize)

    get_feature_importances(database, target_variable, features, discrete_features)\
        .sort_values(ascending=False).head(n).sort_values()\
        .plot.barh(ax=ax, color='#0046c0', title=f'Informação Mútua para conceito: {target_variable}')
        
    plt.tight_layout()